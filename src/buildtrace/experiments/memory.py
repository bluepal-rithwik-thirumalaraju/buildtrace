"""CI adapter around the pinned, upstream WikiTrace compiler and corpus renderers."""

import hashlib
import importlib.metadata
import json
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from wikitrace.config import CompilationInputs, ModelSpec, ModelsConfig, PromptProfile
from wikitrace.harness.llm.client import LLMClient
from wikitrace.memory.compiler import compile_trajectory, relink_compilation
from wikitrace.memory.insights import InsightItem, render_inject, render_insight_document, render_insight_index
from wikitrace.memory.store import (
    RawLayerShape, RawStepFacts, compilation_to_markdown,
    render_compilation_index, render_raw_index_from_steps,
)

from buildtrace.evidence import EvidenceError, EvidenceStore
from buildtrace.experiments.models import AgentMode, EvidenceCitation, ExperimentSpec, RunRecord, fingerprint
from buildtrace.experiments.openai import OpenAIClient
from buildtrace.experiments.prompts import BASE_SYSTEM

UPSTREAM_REVISION = "25a83679af0eb67ab3c835ab920d3a243fb80845"
CI_SHAPE = RawLayerShape(
    agent_logs=("solver.full.jsonl",), optional_agent_logs=(),
    outcome_files=("metadata.md", "verification.json"),
    step_contents="Each raw/step-NNNN/ contains metadata.md, solver.full.jsonl, verification.json, and original CI files under initial-ci/ and attempt-*/ directories.",
    read_hint="Read observable tool calls and brief decision notes in solver.full.jsonl; original CI files are listed in metadata.md",
    prediction_file="verification.json",
    metadata_description="One CI debugging session and its actual verification results.",
    metadata_keys=("task_id", "session_success"), task_key="task", outcome_key="session_success",
    metadata_artifacts="- solver.full.jsonl: observable debugging actions and results\n- verification.json: final outcome and CI verification records\n- initial-ci/: original task evidence",
    index_score="verified={success:.0f}", index_score_pattern=r"verified=(?P<success>\d+)",
    index_score_columns="(verified=0|1)", index_metric_names="verified repair",
    compilation_layer_desc="(session outcome, verified repair, one-line summary)",
    compilation_task_hint="repository, failing revision, CI run, and repair task",
    compilation_outcome_hint="Whether the candidate passed required CI checks; this does not establish causality.",
    compilation_correct_parts_hint="Only observations actually supported by the session.",
    compilation_description="CI debugging session; verified repair={success}.",
    compilation_metric_keys=("success",), compilation_fm_metric_lines=("success: 0 | 1",),
    writer_metric_header=(("success", "VERIFIED_REPAIR"),),
    env_compilation_sections=True, episode=True, no_memory=True,
    turn_log_text="Each solver.full.jsonl row is an observed model message, tool action, tool result, or session outcome. Decision notes are brief, explicit plans.",
)


def upstream_provenance() -> dict:
    distribution = importlib.metadata.distribution("wikitrace")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    if revision != UPSTREAM_REVISION:
        raise ValueError("Install BuildTrace's pinned WikiTrace dependency before compiling or using memory.")
    return {"repository": "https://github.com/yukyeongleee/wikitrace", "revision": revision, "license": "Apache-2.0"}


def _path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("Invalid corpus path")
    destination = root.joinpath(*path.parts).resolve()
    if not destination.is_relative_to(root.resolve()) or destination == root.resolve():
        raise ValueError("Corpus path escapes its root")
    return destination


class FrozenMemory:
    def __init__(self, directory: Path | str, expected_sha256: str | None = None):
        upstream_provenance()
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "corpus.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("upstream", {}).get("revision") != UPSTREAM_REVISION:
            raise ValueError("Unsupported memory corpus or upstream revision")
        self.sha256 = fingerprint(self.manifest)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("Frozen corpus hash differs from the experiment")
        self.items = [
            InsightItem(**(item | {"sections": tuple(tuple(section) for section in item["sections"]), "apps": tuple(item["apps"])}))
            for item in self.manifest["insights"]
        ]
        if not self.items or len({item.id for item in self.items}) != len(self.items):
            raise ValueError("The corpus must contain uniquely identified insights")
        self.insights_text = render_inject(self.items, source=True)
        self.insights_sha256 = hashlib.sha256(self.insights_text.encode("utf-8")).hexdigest()
        if self.insights_sha256 != self.manifest["insights_sha256"]:
            raise ValueError("The injected insight text differs from the frozen corpus")
        self.verify()

    def verify(self):
        for relative, expected in self.manifest["files"].items():
            if hashlib.sha256(_path(self.directory, relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen memory file was modified: {relative}")

    def assert_test_separation(self, spec: ExperimentSpec):
        if self.manifest.get("data_kind") == "synthetic" and spec.data_kind == "live":
            raise ValueError("Synthetic histories cannot be used as live experimental memory")
        if self.manifest.get("compiler_model") != spec.model.model_dump(mode="json"):
            raise ValueError("The corpus compiler model/settings differ from this experiment")
        tests = [task for task in spec.tasks if task.split == "test"]
        for source in self.manifest["training_tasks"]:
            if any(task.id == source["id"] or (task.repository, task.base_sha) == (source["repository"], source["base_sha"]) for task in tests):
                raise ValueError("Test task leaked into the historical memory corpus")
            if spec.hold_out_families and any(task.family == source["family"] for task in tests):
                raise ValueError("A held-out scenario family is present in memory")

    def read(self, insight_id: str, layer: str, document: str = "", offset: int = 0, limit: int = 12000):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12000:
            raise ValueError("Invalid historical evidence range")
        item = next((item for item in self.items if item.id == insight_id), None)
        if item is None:
            raise ValueError("Unknown insight ID")
        if layer == "compilation":
            if document:
                raise ValueError("A compilation has a single source document")
            relative = f"compilation/{item.source_step}.md"
        elif layer == "raw":
            relative = f"raw/{item.source_step}/{document or 'metadata.md'}"
        else:
            raise ValueError("Read compilation or raw historical evidence")
        if relative not in self.manifest["files"] or relative.endswith(".zip"):
            raise ValueError("The requested historical text document is not in the linked source run")
        content = _path(self.directory, relative).read_bytes()
        expected = self.manifest["files"][relative]
        if hashlib.sha256(content).hexdigest() != expected:
            raise EvidenceError("Historical evidence checksum mismatch")
        text = content.decode("utf-8", errors="replace")
        if offset >= len(text) and offset != 0:
            raise ValueError("Historical evidence offset is outside the document")
        excerpt = text[offset:offset + limit]
        first_line = text[:offset].count("\n") + 1
        citation = EvidenceCitation(
            insight_id=insight_id, source_run_id=item.source_step, layer=layer,
            document=f"/work/corpus/{relative}", first_line=first_line,
            last_line=first_line + excerpt.count("\n"), document_sha256=expected,
        )
        return {
            "text": excerpt, "offset": offset,
            "next_offset": offset + len(excerpt) if offset + len(excerpt) < len(text) else None,
            "citation": citation.model_dump(mode="json"),
        }, citation


def build_memory(
    spec: ExperimentSpec,
    records: list[RunRecord],
    records_root: Path,
    store: EvidenceStore,
    output: Path,
    transport: OpenAIClient,
) -> FrozenMemory:
    """Use actual upstream compilation and rendering on complete no-memory training sessions."""
    upstream = upstream_provenance()
    tasks = {task.id: task for task in spec.tasks if task.split == "train"}
    if not records or output.exists():
        raise ValueError("Supply training records and a new corpus output directory")
    model = ModelSpec(
        provider="openai", model=spec.model.model, temperature=spec.model.temperature if spec.model.temperature is not None else 0,
        max_tokens=spec.budget.max_output_tokens, reasoning_effort=spec.model.reasoning_effort or "",
    )
    llm = LLMClient(ModelsConfig(default=model), transport=transport, max_retries=0)
    profile = PromptProfile(
        task_kind="Python CI build and test failure debugging", solver_prompt=BASE_SYSTEM, has_evidence=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".pending-{uuid.uuid4().hex}"
    staging.mkdir()
    files, items, compilations, training_tasks, compiler_calls = {}, [], [], [], []

    def write(relative, content):
        if isinstance(content, str):
            content = content.encode("utf-8")
        path = _path(staging, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[relative] = hashlib.sha256(content).hexdigest()

    try:
        ordered = sorted(records, key=lambda record: (record.task_id, record.trial, record.id))
        if len({record.id for record in ordered}) != len(ordered):
            raise ValueError("Duplicate source debugging sessions")
        for number, record in enumerate(ordered):
            task = tasks.get(record.task_id)
            if (
                task is None or record.task_sha256 != fingerprint(task)
                or record.experiment_sha256 != fingerprint(spec)
                or record.model != spec.model or record.budget != spec.budget
                or record.mode != AgentMode.NO_MEMORY or record.role != "reference"
                or record.outcome not in {"solved", "unsolved"}
                or any(hunk.classification != "unreviewed" for hunk in record.hunks)
            ):
                raise ValueError("Memory sources must be original, unannotated no-memory runs on declared training tasks")
            trajectory = _path(records_root.resolve(), record.trajectory_path).read_bytes()
            step = f"step-{number:04d}"
            root = f"raw/{step}"
            write(f"{root}/solver.full.jsonl", trajectory)
            write(f"{root}/verification.json", record.model_dump_json(indent=2))
            capture = store.load(task.capture_id)
            if capture["source_ref"] != task.base_sha:
                raise ValueError("The training run's initial CI snapshot differs from its task")
            evidence_paths = []
            captures = [("initial-ci", task.capture_id)]
            for verification in record.verifications:
                if verification.capture_id:
                    captures.append((f"attempt-{verification.attempt}-target-ci", verification.capture_id))
                if verification.oracle_capture_id:
                    captures.append((f"attempt-{verification.attempt}-oracle-ci", verification.oracle_capture_id))
            for folder, capture_id in captures:
                original = store.load(capture_id)
                write(f"{root}/{folder}/manifest.json", json.dumps(original, indent=2))
                for entry in original["entries"]:
                    _, content = store.read_entry(capture_id, entry["id"])
                    relative = f"{root}/{folder}/{entry['path']}"
                    write(relative, content)
                    evidence_paths.append(relative)
            metadata = (
                f"# {task.id}\n- task: Repair the failed Python build while preserving required behavior.\n"
                f"- repository: {task.repository}\n- base_sha: {task.base_sha}\n"
                f"- initial_run: {task.initial_run_url}\n- session_success: {int(record.success)}\n"
                f"- outcome: {record.outcome}\n- source_run_id: {record.id}\n\n"
                "## Session records\n- solver.full.jsonl: observable investigation\n"
                "- verification.json: session outcome and CI verification records\n\n"
                "## Original CI evidence (initial and verification runs)\n"
                + "\n".join(f"- /work/corpus/{path}" for path in evidence_paths) + "\n"
            )
            write(f"{root}/metadata.md", metadata)
            inputs = CompilationInputs(
                sample_id=record.id, db_id=task.repository, success=record.success,
                metrics={"success": float(record.success)},
                question=f"Repair the failed Python CI run {task.initial_run_url}",
                prediction=record.verifications[-1].candidate_sha if record.verifications else "",
                task_fields=(("repository", task.repository), ("base_sha", task.base_sha), ("initial_run", task.initial_run_url)),
            )
            result = compile_trajectory(
                llm, inputs, number, trajectory_text=trajectory.decode("utf-8"),
                outcome_text=json.dumps({"outcome": record.outcome, "stop_reason": record.stop_reason,
                                         "verifications": [value.model_dump(mode="json") for value in record.verifications]}),
                profile=profile,
            )
            candidates = result.structured.parsed.candidate_insights
            if not 1 <= len(candidates) <= 3:
                raise ValueError("The compiler must return between one and three source-linked insights")
            step_items = [
                InsightItem(
                    id=f"insight-{len(items) + index:04d}", title=candidate.title,
                    sections=(("Insight", candidate.guideline), ("Applicability", candidate.applicability)),
                    source_step=step, outcome="success" if record.success else "failure",
                    task=inputs.question, task_type=task.category,
                )
                for index, candidate in enumerate(candidates)
            ]
            items.extend(step_items)
            compilation = result.compilation
            compilation.body = relink_compilation(compilation.body, step_items)
            compilations.append(compilation)
            write(f"compilation/{step}.md", compilation_to_markdown(compilation, raw_shape=CI_SHAPE))
            for item in step_items:
                write(f"insight/{item.id}.md", render_insight_document(item, task_type=True))
            training_tasks.append({"id": task.id, "family": task.family, "repository": task.repository, "base_sha": task.base_sha, "source_run_id": record.id})
            compiler_calls.append(result.structured.record.model_dump(mode="json"))
        write("insight/_index.md", render_insight_index(items))
        write("compilation/_index.md", render_compilation_index(compilations, entry_source="title", raw_shape=CI_SHAPE))
        write("raw/_index.md", render_raw_index_from_steps([
            (comp.step_id, RawStepFacts(outcome=comp.outcome.value, tail=comp.question, metrics=comp.metrics))
            for comp in compilations
        ], raw_shape=CI_SHAPE))
        manifest = {
            "schema_version": 1, "data_kind": spec.data_kind, "upstream": upstream, "compiler_model": spec.model.model_dump(mode="json"),
            "training_tasks": training_tasks, "insights": [asdict(item) for item in items],
            "insights_sha256": hashlib.sha256(render_inject(items, source=True).encode("utf-8")).hexdigest(),
            "compiler_calls": compiler_calls, "files": files,
        }
        (staging / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return FrozenMemory(output)
