"""CLI wiring for training, frozen memory, paired experiments, and all six measurements."""

import json
import os
import sys
from pathlib import Path

from buildtrace.evidence import EvidenceError, EvidenceStore
from buildtrace.experiments.backend import BackendError
from buildtrace.experiments.memory import FrozenMemory, build_memory
from buildtrace.experiments.metrics import build_report, read_records, render_markdown
from buildtrace.experiments.openai import OpenAIClient, OpenAIError
from buildtrace.experiments.workflow import apply_reviews, load_spec, review_template, run_experiment, validate_local_inputs
from buildtrace.experiments.models import fingerprint
from buildtrace.github import GitHubActionsClient, GitHubError
from buildtrace.models import FAILED_CONCLUSIONS, RunReference


def add_commands(commands):
    experiment = commands.add_parser("experiment", help="Compare no-memory, insights-only, and WikiTrace")
    actions = experiment.add_subparsers(dest="experiment_action", required=True)
    validate = actions.add_parser("validate", help="Check local experiment inputs without model calls or GitHub writes")
    run = actions.add_parser("run", help="Run shared agents on the train or held-out test split")
    memory = actions.add_parser("build-memory", help="Compile actual no-memory training sessions with upstream WikiTrace")
    report = actions.add_parser("report", help="Compute paired metrics and write JSON and Markdown reports")
    review = actions.add_parser("review", help="Attach independent hunk/evidence reviews to a derived results file")
    template = actions.add_parser("review-template", help="Export blinded hunk/evidence annotation forms")
    capture = actions.add_parser("capture-inputs", help="Archive the exact declared failing runs without model calls")
    attach = actions.add_parser("attach-memory", help="Bind verified frozen memory hashes to an evaluation spec")
    for action in (validate, run, memory, report, capture, attach):
        action.add_argument("--spec", type=Path, required=True)
    for action in (validate, run, memory, capture):
        action.add_argument("--data-dir", type=Path, default=Path(os.environ.get("BUILDTRACE_DATA_DIR", "data")))
    run.add_argument("--split", choices=["train", "test"], required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--allow-publish", action="store_true", help="Authorize trial commits/branches and paid CI/model execution")
    run.add_argument("--include-reruns", action="store_true", help="Also collect independent no-memory reruns on test tasks")
    run.add_argument("--continue-on-infra", action="store_true", help="Continue while retaining recorded infrastructure failures")
    for action in (memory, report, review, template):
        action.add_argument("--records", type=Path, required=True)
        action.add_argument("--output", type=Path, required=True)
    report.add_argument("--bootstrap", type=int, default=2000)
    review.add_argument("--annotations", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    attach.add_argument("--corpus", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)


def dispatch(args) -> int:
    try:
        if args.experiment_action == "review-template":
            if args.output.exists():
                raise ValueError("Choose a new annotation file")
            form = review_template(read_records(args.records), args.records.read_bytes())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(form, indent=2) + "\n")
            print(f"Saved independent review template to {args.output}")
            return 0
        if args.experiment_action == "review":
            records = apply_reviews(read_records(args.records), args.annotations, args.records.read_bytes())
            if args.output.exists():
                raise ValueError("Choose a new derived results file; original records are never overwritten")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write("".join(record.model_dump_json() + "\n" for record in records))
            print(json.dumps({"reviewed_records": len(records), "output": str(args.output)}, indent=2))
            return 0
        spec = load_spec(args.spec)
        if args.experiment_action in {"capture-inputs", "attach-memory"}:
            if args.output.exists():
                raise ValueError("Choose a new output specification file")
            if args.experiment_action == "capture-inputs":
                store, tasks = EvidenceStore(args.data_dir), []
                with GitHubActionsClient() as client:
                    for task in spec.tasks:
                        collected = client.collect(RunReference.parse(task.initial_run_url))
                        if collected.source_ref != task.base_sha or collected.run.get("conclusion") not in FAILED_CONCLUSIONS or collected.log_archive is None:
                            raise ValueError(f"{task.id}: the exact failing source revision and original logs are required")
                        archived = store.save(collected)
                        tasks.append(task.model_copy(update={"capture_id": archived["id"], "capture_sha256": fingerprint(archived)}))
                spec = spec.model_copy(update={"tasks": tasks})
            else:
                corpus = FrozenMemory(args.corpus)
                corpus.assert_test_separation(spec)
                spec = spec.model_copy(update={"corpus": corpus.directory, "corpus_sha256": corpus.sha256, "insights_sha256": corpus.insights_sha256})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(spec.model_dump_json(indent=2) + "\n")
            print(json.dumps({"specification": str(args.output), "experiment_sha256": fingerprint(spec)}, indent=2))
        elif args.experiment_action == "validate":
            result = validate_local_inputs(spec, EvidenceStore(args.data_dir))
            print(json.dumps(result, indent=2))
            return 0 if result["local_prerequisites_ready"] else 2
        elif args.experiment_action == "run":
            records = run_experiment(
                spec, EvidenceStore(args.data_dir), args.output, split=args.split,
                allow_publish=args.allow_publish, include_reruns=args.include_reruns,
                continue_on_infra=args.continue_on_infra,
            )
            print(json.dumps({"completed_runs": len(records), "results": str(args.output / "results.jsonl"), "frozen_spec": str(args.output / "experiment.json")}, indent=2))
        elif args.experiment_action == "build-memory":
            records = read_records(args.records)
            with OpenAIClient(spec.model) as transport:
                corpus = build_memory(spec, records, args.records.parent.resolve(), EvidenceStore(args.data_dir), args.output, transport)
            print(json.dumps({"corpus": str(args.output.resolve()), "corpus_sha256": corpus.sha256, "insights_sha256": corpus.insights_sha256, "insights": len(corpus.items)}, indent=2))
        elif args.experiment_action == "report":
            result = build_report(spec, read_records(args.records), resamples=args.bootstrap)
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            (args.output / "summary.md").write_text(render_markdown(result), encoding="utf-8")
            print(render_markdown(result))
            print(f"Saved reports to {args.output}")
        return 0
    except (ValueError, OSError, KeyError, EvidenceError, BackendError, OpenAIError, GitHubError) as exc:
        print(f"buildtrace experiment: {exc}", file=sys.stderr)
        return 2
