"""Freeze inputs, counterbalance the three modes, resume completed trials, and export records."""

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

from buildtrace.evidence import EvidenceStore
from buildtrace.experiments.backend import GitHubTrialBackend
from buildtrace.experiments.memory import FrozenMemory
from buildtrace.experiments.models import AgentMode, ExperimentSpec, RunRecord, fingerprint
from buildtrace.experiments.openai import OpenAIClient
from buildtrace.experiments.runner import protocol_fingerprint, run_trial, trial_id
from buildtrace.github import resolve_github_token


def load_spec(path: Path) -> ExperimentSpec:
    spec = ExperimentSpec.model_validate_json(path.read_text(encoding="utf-8"))
    if spec.corpus is not None and not spec.corpus.is_absolute():
        spec = spec.model_copy(update={"corpus": (path.parent / spec.corpus).resolve()})
    return spec


def validate_local_inputs(spec: ExperimentSpec, store: EvidenceStore) -> dict:
    needs = []
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        needs.append("Configure OPENAI_API_KEY locally before live model calls.")
    if not resolve_github_token():
        needs.append("Configure GH_TOKEN/GITHUB_TOKEN or an authenticated GitHub CLI for live CI trials.")
    for task in spec.tasks:
        try:
            capture = store.load(task.capture_id)
            store.verify(task.capture_id)
            if capture["source_ref"] != task.base_sha:
                needs.append(f"{task.id}: capture revision differs from the task.")
            if task.capture_sha256 and fingerprint(capture) != task.capture_sha256:
                needs.append(f"{task.id}: initial evidence hash differs from the task.")
        except (OSError, ValueError, RuntimeError) as exc:
            needs.append(f"{task.id}: archive its exact failing CI attempt first ({exc}).")
    memory = None
    if any(task.split == "test" for task in spec.tasks):
        if spec.corpus is None:
            needs.append("Collect separate no-memory training sessions and build a frozen WikiTrace corpus.")
        else:
            try:
                memory = FrozenMemory(spec.corpus, spec.corpus_sha256)
                if memory.insights_sha256 != spec.insights_sha256:
                    raise ValueError("Injected insight hash differs")
                memory.assert_test_separation(spec)
            except (OSError, ValueError) as exc:
                needs.append(f"Resolve historical memory: {exc}")
    return {
        "schema_valid": True,
        "experiment_id": spec.id,
        "model": spec.model.model,
        "tasks_by_split": dict(Counter(task.split for task in spec.tasks)),
        "memory_verified": memory is not None,
        "local_prerequisites_ready": not needs,
        "remaining_inputs": needs,
        "online_preflight": "Before any trial, the runner also validates and pins the trusted GitHub oracle workflow.",
    }


def schedule(spec: ExperimentSpec, split: str, include_reruns: bool = False):
    rng = random.Random(spec.seed)
    tasks = sorted((task for task in spec.tasks if task.split == split), key=lambda task: task.id)
    rng.shuffle(tasks)
    planned = []
    for task in tasks:
        modes = list(AgentMode) if split == "test" else [AgentMode.NO_MEMORY]
        rng.shuffle(modes)
        for trial in range(spec.trials):
            ordered = modes[trial % len(modes):] + modes[:trial % len(modes)]
            planned.extend((task, mode, trial, "reference") for mode in ordered)
            if include_reruns and split == "test":
                planned.append((task, AgentMode.NO_MEMORY, trial, "rerun"))
    return planned


def run_experiment(
    spec: ExperimentSpec,
    store: EvidenceStore,
    output: Path,
    *,
    split: str,
    allow_publish: bool,
    include_reruns: bool = False,
    continue_on_infra: bool = False,
    model_factory=OpenAIClient,
    backend_factory=GitHubTrialBackend,
    memory_factory=FrozenMemory,
) -> list[RunRecord]:
    if not allow_publish:
        raise ValueError("Authorize experimental branches and CI runs explicitly with --allow-publish.")
    if split not in {"train", "test"} or not any(task.split == split for task in spec.tasks):
        raise ValueError("Choose a declared train or test split")
    memory = None
    if split == "test":
        if spec.corpus is None:
            raise ValueError("Three-mode evaluation requires a frozen corpus from separate training sessions")
        memory = memory_factory(spec.corpus, spec.corpus_sha256)
        if memory.insights_sha256 != spec.insights_sha256:
            raise ValueError("The experiment's insight hash differs from the frozen memory")
        memory.assert_test_separation(spec)

    with model_factory(spec.model) as model:
        pinned_tasks = []
        for task in spec.tasks:
            backend = backend_factory(task, store, allow_publish=True, namespace="preflight-" + fingerprint(task)[:20])
            try:
                pinned_tasks.append(backend.pin_oracle())
            finally:
                backend.close()
        spec = spec.model_copy(update={"tasks": pinned_tasks})
        output.mkdir(parents=True, exist_ok=True)
        spec_file = output / "experiment.json"
        if spec_file.exists():
            if fingerprint(load_spec(spec_file)) != fingerprint(spec):
                raise ValueError("This output directory belongs to a different frozen experiment")
        else:
            with spec_file.open("x", encoding="utf-8") as stream:
                stream.write(spec.model_dump_json(indent=2) + "\n")
        planned = schedule(spec, split, include_reruns)
        records = []
        for task, mode, trial, role in planned:
            identifier = trial_id(spec, task, mode, trial, role)
            existing = output / "trials" / identifier / "result.json"
            if existing.exists():
                record = RunRecord.model_validate_json(existing.read_text(encoding="utf-8"))
                if record.experiment_sha256 != fingerprint(spec) or record.protocol_sha256 != protocol_fingerprint():
                    raise ValueError("Completed trials have different configuration, code, or prompts; use a new experiment ID.")
            else:
                backend = backend_factory(task, store, allow_publish=True, namespace=identifier)
                try:
                    record = run_trial(spec, task, mode, trial, backend, model, output, memory=memory, role=role)
                finally:
                    backend.close()
            records.append(record)
            temporary = output / "results.jsonl.pending"
            temporary.write_text("".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8")
            temporary.replace(output / "results.jsonl")
            if record.outcome == "infra-error" and not continue_on_infra:
                raise ValueError(
                    f"Trial {record.id} hit an infrastructure error and was preserved. "
                    "Resolve the setup; use a new experiment ID or explicitly continue while retaining this outcome."
                )
        return records


def apply_reviews(records: list[RunRecord], review_path: Path, original_bytes: bytes) -> list[RunRecord]:
    """Attach independent annotations to a derived copy, keeping original trial records intact."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("source_results_sha256") != hashlib.sha256(original_bytes).hexdigest():
        raise ValueError("Review annotations refer to a different results file")
    by_id = {record.id: record.model_copy(deep=True) for record in records}
    seen = set()
    for annotation in review.get("hunks", []):
        if annotation.get("classification") == "unreviewed":
            continue
        key = (annotation["run_id"], annotation["hunk_id"])
        if key in seen:
            raise ValueError("Duplicate hunk review")
        seen.add(key)
        record = by_id[annotation["run_id"]]
        hunk = next((hunk for hunk in record.hunks if hunk.id == annotation["hunk_id"]), None)
        if hunk is None or not annotation.get("reviewer") or annotation["classification"] not in {"necessary", "unnecessary", "uncertain"}:
            raise ValueError("Invalid independent hunk review")
        hunk.classification, hunk.reviewer = annotation["classification"], annotation["reviewer"]
    for annotation in review.get("evidence", []):
        if annotation.get("review") == "unreviewed":
            continue
        key = (annotation["run_id"], annotation["event_id"])
        if key in seen:
            raise ValueError("Duplicate evidence-impact review")
        seen.add(key)
        record = by_id[annotation["run_id"]]
        event = next((event for event in record.evidence_decisions if event.id == annotation["event_id"]), None)
        if event is None or not annotation.get("reviewer") or annotation["review"] not in {"confirmed-change", "confirmed-no-change", "uncertain"}:
            raise ValueError("Invalid independent evidence-impact review")
        event.review, event.reviewer = annotation["review"], annotation["reviewer"]
    return [RunRecord.model_validate(by_id[record.id].model_dump()) for record in records]


def review_template(records: list[RunRecord], original_bytes: bytes) -> dict:
    return {
        "schema_version": 1,
        "source_results_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "instructions": "Review behavior and task requirements independently. Agent mode is intentionally omitted. Leave uncertain or unreviewed changes unscored.",
        "hunks": [
            {"run_id": record.id, "task_id": record.task_id, "hunk_id": hunk.id, "path": hunk.path,
             "diff": hunk.diff, "classification": "unreviewed", "reviewer": ""}
            for record in records for hunk in record.hunks
        ],
        "evidence": [
            {"run_id": record.id, "event_id": event.id, "question": event.question,
             "before": event.before.model_dump(mode="json"),
             "after": event.after.model_dump(mode="json") if event.after else None,
             "citations": [citation.model_dump(mode="json") for citation in event.citations],
             "review": "unreviewed", "reviewer": ""}
            for record in records for event in record.evidence_decisions
        ],
    }
