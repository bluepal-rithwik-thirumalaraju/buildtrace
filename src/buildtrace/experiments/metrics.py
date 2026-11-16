"""Paired, denominator-explicit measurements; no LLM grades task success."""

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from buildtrace.experiments.models import AgentMode, ExperimentSpec, RunRecord, fingerprint


def read_records(path: Path | str) -> list[RunRecord]:
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(RunRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid run record on line {number}: {exc}") from exc
    return records


def _rate(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "percent": 100 * numerator / denominator if denominator else None,
    }


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def validate_cohort(spec: ExperimentSpec, records: list[RunRecord]) -> dict:
    tasks = {task.id: task for task in spec.tasks if task.split == "test"}
    if not tasks:
        raise ValueError("Reporting requires held-out test tasks")
    seen = set()
    ids = set()
    verification_runs = set()
    returned_models = {model for record in records for model in record.returned_models}
    if len(returned_models) > 1:
        raise ValueError("The comparison contains multiple served model versions")
    if len({record.protocol_sha256 for record in records}) > 1:
        raise ValueError("The comparison mixes different agent implementations or prompts")
    for record in records:
        key = (record.task_id, record.mode, record.trial, record.role)
        if key in seen or record.id in ids:
            raise ValueError(f"Duplicate run record: {record.id}")
        seen.add(key)
        ids.add(record.id)
        if spec.data_kind == "live":
            if record.outcome in {"solved", "unsolved"} and (record.model_calls < 1 or not record.returned_models):
                raise ValueError("Live agent outcomes need recorded model calls and the served model version")
            for verification in record.verifications:
                urls = [verification.run_url, verification.oracle_run_url]
                if not all(urls) or not verification.capture_id or not verification.oracle_capture_id:
                    raise ValueError("Live verification needs archived target and trusted-oracle runs")
                if any(url in verification_runs for url in urls):
                    raise ValueError("Independent agent trials cannot reuse the same CI verification run")
                verification_runs.update(urls)
        if record.experiment_id != spec.id or record.experiment_sha256 != fingerprint(spec):
            raise ValueError(f"Run {record.id} belongs to a different experiment configuration")
        if record.task_id not in tasks or record.task_sha256 != fingerprint(tasks[record.task_id]):
            raise ValueError(f"Run {record.id} has a different task snapshot or is not held out")
        if record.model != spec.model or record.budget != spec.budget:
            raise ValueError(f"Run {record.id} has a different model or repair budget")
        if record.trial >= spec.trials:
            raise ValueError(f"Run {record.id} has an out-of-range trial index")
        if record.mode != AgentMode.NO_MEMORY and (
            record.corpus_sha256 != spec.corpus_sha256
            or record.insights_sha256 != spec.insights_sha256
        ):
            raise ValueError(f"Run {record.id} did not receive the experiment's identical frozen insights")
    expected = {
        (task_id, mode, trial, "reference")
        for task_id in tasks
        for mode in AgentMode
        for trial in range(spec.trials)
    }
    missing = expected - seen
    if missing:
        task_id, mode, trial, _ = sorted(missing)[0]
        raise ValueError(
            f"Incomplete comparison: {len(missing)} missing run(s), including "
            f"{task_id}/{mode.value}/trial-{trial}. Run all three modes before reporting."
        )
    reruns = {key for key in seen if key[3] == "rerun"}
    expected_reruns = {(task_id, AgentMode.NO_MEMORY, trial, "rerun") for task_id in tasks for trial in range(spec.trials)}
    if reruns and reruns != expected_reruns:
        raise ValueError("The no-memory rerun reference must cover every task and trial")
    return tasks


def _summarize(runs: list[RunRecord], baselines: dict[str, bool | None], *, reference: bool = False) -> dict:
    gained = [run for run in runs if baselines[run.task_id] is False]
    regressed = [run for run in runs if baselines[run.task_id] is True]
    hunks = [hunk for run in runs for hunk in run.hunks]
    reviewed = [hunk for hunk in hunks if hunk.classification in {"necessary", "unnecessary"}]
    unnecessary = [hunk for hunk in reviewed if hunk.classification == "unnecessary"]
    full_review = len(reviewed) == len(hunks)
    events = [event for run in runs for event in run.evidence_decisions]
    assessed = [event for event in events if event.changed is not None]
    confirmed = [event for event in events if event.review in {"confirmed-change", "confirmed-no-change"}]
    outcomes = Counter(run.outcome for run in runs)
    solved = [run for run in runs if run.success]
    per_task = defaultdict(list)
    for run in runs:
        per_task[run.task_id].append(run.success)
    return {
        "attempts": len(runs),
        "successful_fixes": _rate(len(solved), len(runs)),
        "tasks_solved_by_majority": _rate(sum(sum(values) > len(values) / 2 for values in per_task.values()), len(per_task)),
        "outcomes": {key: outcomes[key] for key in ("solved", "unsolved", "infra-error", "not-reproduced")},
        "gain": None if reference else _rate(sum(run.success for run in gained), len(gained)),
        "regression": None if reference else _rate(sum(not run.success for run in regressed), len(regressed)),
        "changes": {
            "changed_hunks": len(hunks),
            "changed_lines": sum(hunk.added_lines + hunk.deleted_lines for hunk in hunks),
            "reviewed_hunks": len(reviewed),
            "review_coverage": _rate(len(reviewed), len(hunks)),
            "known_unnecessary_hunks": len(unnecessary),
            "unnecessary_hunks": len(unnecessary) if full_review else None,
            "unnecessary_lines": sum(hunk.added_lines + hunk.deleted_lines for hunk in unnecessary) if full_review else None,
            "uncertain_or_unreviewed_hunks": len(hunks) - len(reviewed),
        },
        "time_seconds": {
            "all_attempts_wall_median": _median([run.timing.wall_seconds for run in runs]),
            "successful_fixes_wall_median": _median([run.timing.wall_seconds for run in solved]),
            "successful_fixes_active_median": _median([run.timing.active_seconds for run in solved]),
            "ci_wait_total": sum(run.timing.ci_wait_seconds for run in runs),
            "approval_wait_total": sum(run.timing.approval_wait_seconds for run in runs),
        },
        "evidence_impact": {
            "lookups": len(events),
            "completed_before_after": len(assessed),
            "decision_changes": _rate(sum(event.changed for event in assessed), len(assessed)),
            "independently_reviewed": len(confirmed),
            "review_confirmed_changes": _rate(sum(event.review == "confirmed-change" for event in confirmed), len(confirmed)),
            "runs_with_decision_change": sum(any(event.changed for event in run.evidence_decisions) for run in runs),
        },
        "usage": {
            "model_calls": sum(run.model_calls for run in runs),
            "input_tokens": sum(run.input_tokens for run in runs),
            "output_tokens": sum(run.output_tokens for run in runs),
            "reasoning_tokens_included_in_output": sum(run.reasoning_tokens for run in runs),
            "cached_input_tokens": sum(run.cached_input_tokens for run in runs),
            "patch_attempts": sum(len(run.verifications) for run in runs),
        },
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _paired_intervals(groups: dict, task_ids: list[str], baselines: dict, resamples: int, seed: int) -> dict:
    rng = random.Random(seed)
    result = {}
    for competitor in (AgentMode.NO_MEMORY, AgentMode.INSIGHTS_ONLY):
        draws = defaultdict(list)
        for _ in range(resamples):
            sampled = rng.choices(task_ids, k=len(task_ids))
            for metric in ("successful_fixes", "gain", "regression"):
                if competitor == AgentMode.NO_MEMORY and metric != "successful_fixes":
                    # The reference trials define eligibility; they are not an independent noise floor.
                    continue
                eligible = sampled
                if metric != "successful_fixes":
                    want = metric == "regression"
                    eligible = [task_id for task_id in sampled if baselines[task_id] is want]
                if not eligible:
                    continue
                means = {}
                for mode in (AgentMode.WIKITRACE, competitor):
                    observations = [run.success for task_id in eligible for run in groups[mode][task_id]]
                    if metric == "regression":
                        observations = [not success for success in observations]
                    means[mode] = sum(observations) / len(observations)
                draws[metric].append(100 * (means[AgentMode.WIKITRACE] - means[competitor]))
        result[f"wikitrace-minus-{competitor.value}"] = {
            metric: {
                "unit": "percentage-points",
                "ci95_low": _percentile(draws[metric], 0.025),
                "ci95_high": _percentile(draws[metric], 0.975),
                "eligible_resamples": len(draws[metric]),
            }
            for metric in ("successful_fixes", "gain", "regression")
        }
    return result


def build_report(spec: ExperimentSpec, records: list[RunRecord], *, resamples: int = 2000) -> dict:
    tasks = validate_cohort(spec, records)
    if resamples < 0:
        raise ValueError("Bootstrap resamples must be nonnegative")
    groups = {mode: defaultdict(list) for mode in AgentMode}
    reruns = []
    for record in records:
        if record.role == "rerun":
            reruns.append(record)
        else:
            groups[record.mode][record.task_id].append(record)
    baselines = {}
    for task_id in tasks:
        reference = groups[AgentMode.NO_MEMORY][task_id]
        baselines[task_id] = (
            sum(run.success for run in reference) > spec.trials // 2
            if all(run.outcome in {"solved", "unsolved"} for run in reference)
            else None
        )
    report = {
        "schema_version": 1,
        "experiment_id": spec.id,
        "experiment_sha256": fingerprint(spec),
        "protocol_sha256": records[0].protocol_sha256,
        "data_kind": spec.data_kind,
        "tasks": len(tasks),
        "trials_per_mode": spec.trials,
        "baseline_solved_tasks": sum(value is True for value in baselines.values()),
        "baseline_unsolved_tasks": sum(value is False for value in baselines.values()),
        "baseline_unscorable_tasks": [task_id for task_id, value in baselines.items() if value is None],
        "methods": {
            mode.value: _summarize(
                [run for task_id in sorted(tasks) for run in groups[mode][task_id]],
                baselines,
                reference=mode == AgentMode.NO_MEMORY,
            )
            for mode in AgentMode
        },
        "paired_bootstrap": {
            "resamples": resamples,
            "seed": spec.seed,
            "task_level_differences": _paired_intervals(groups, sorted(tasks), baselines, resamples, spec.seed),
        },
        "notes": [
            "Gain and Regression use the majority of the same task's no-memory reference trials.",
            "Infrastructure errors remain visible and count as unsuccessful attempted fixes; tasks with invalid no-memory references have no Gain/Regression label.",
            "No eligible tasks means N/A, not zero. No-memory reference Gain/Regression is omitted; independent reruns estimate ordinary variation.",
            "Patch size is a proxy only. Unnecessary changes require independent hunk review; incomplete review is not reported as zero unnecessary changes.",
            "Resolution time is reported separately for successful fixes and all attempts, with CI and human waiting separated.",
            "Evidence impact counts recorded actionable before/after plan changes, not wording changes or proven causal effects. Independent review is reported separately.",
            "All repeated trials stay together when bootstrapping paired task-level differences. Higher Gain/success and lower Regression are favorable.",
        ],
    }
    if reruns:
        report["no_memory_rerun_reference"] = _summarize(reruns, baselines)
    if spec.data_kind == "synthetic":
        report["notes"].insert(0, "SYNTHETIC DATA: this report tests the measurement pipeline and is not evidence of real agent performance.")
    return report


def render_markdown(report: dict) -> str:
    def rate(value):
        if value is None or value["percent"] is None:
            return "N/A"
        return f"{value['percent']:.1f}% ({value['numerator']}/{value['denominator']})"

    def number(value):
        return "N/A" if value is None else str(round(value, 2))

    lines = [f"# BuildTrace comparison: {report['experiment_id']}", ""]
    if report["data_kind"] == "synthetic":
        lines += ["**Synthetic data: pipeline verification only; these are not real agent results.**", ""]
    lines += [
        f"Tasks: {report['tasks']}; trials per mode: {report['trials_per_mode']}.",
        "",
        "| Mode | Successful fixes | Gain | Regression | Unnecessary hunks | Median successful resolution (s) | Evidence changes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, method in report["methods"].items():
        lines.append(
            f"| {name} | {rate(method['successful_fixes'])} | {rate(method['gain'])} | "
            f"{rate(method['regression'])} | {number(method['changes']['unnecessary_hunks'])} | "
            f"{number(method['time_seconds']['successful_fixes_wall_median'])} | "
            f"{rate(method['evidence_impact']['decision_changes'])} |"
        )
    lines += ["", "## Measurement notes", "", *(f"- {note}" for note in report["notes"])]
    return "\n".join(lines) + "\n"
