"""The same solver loop in all three modes; only historical access differs."""

import hashlib
import json
import time
from pathlib import Path

from buildtrace.evidence import EvidenceError
from buildtrace.experiments.backend import BackendError
from buildtrace.experiments.memory import FrozenMemory
from buildtrace.experiments.models import (
    AgentMode, Decision, EvidenceDecision, ExperimentSpec, RunRecord, TaskSpec, Timing, fingerprint,
)
from buildtrace.experiments.openai import OpenAIError
from buildtrace.experiments.prompts import BASE_SYSTEM, CI_DESCENT, tool_definitions


def protocol_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_text(encoding="utf-8").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def trial_id(spec: ExperimentSpec, task: TaskSpec, mode: AgentMode, trial: int, role="reference") -> str:
    return "run-" + fingerprint([fingerprint(spec), task.id, mode.value, trial, role])[:24]


def _reject_json_constant(value):
    raise ValueError(f"Use finite JSON values; {value} is not a valid tool argument")


def run_trial(
    spec: ExperimentSpec,
    task: TaskSpec,
    mode: AgentMode,
    trial: int,
    backend,
    model,
    output: Path,
    *,
    memory: FrozenMemory | None = None,
    role: str = "reference",
) -> RunRecord:
    if not 0 <= trial < spec.trials:
        raise ValueError("Trial index is outside the experiment")
    if mode != AgentMode.NO_MEMORY:
        if memory is None or (memory.sha256, memory.insights_sha256) != (spec.corpus_sha256, spec.insights_sha256):
            raise ValueError("Memory modes must receive the same frozen corpus and identical insights")
        if task.split == "test":
            memory.assert_test_separation(spec)
    if task.split == "train" and mode != AgentMode.NO_MEMORY:
        raise ValueError("Collect training sessions without historical memory")
    if model.settings != spec.model:
        raise ValueError("The solver model settings differ from the experiment")
    observation = backend.initial_observation()

    identifier = trial_id(spec, task, mode, trial, role)
    relative_trajectory = f"trials/{identifier}/trajectory.jsonl"
    directory = output / "trials" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    journal = (directory / "trajectory.jsonl").open("x", encoding="utf-8", newline="\n")
    start = time.monotonic()
    deadline = start + spec.budget.wall_seconds
    ci_wait = 0.0
    verifications, hunks, evidence_decisions = [], [], []
    model_calls = input_tokens = output_tokens = 0
    reasoning_tokens = cached_input_tokens = 0
    returned_models = set()
    current_plan = None
    pending_evidence = None
    outcome, stop_reason = "unsolved", "agent-stopped"

    def log(row):
        journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        journal.flush()

    def finish_evidence(after=None):
        nonlocal pending_evidence
        if pending_evidence is not None:
            pending_evidence.after = after
            evidence_decisions.append(pending_evidence)
            pending_evidence = None

    tools = tool_definitions(mode == AgentMode.WIKITRACE)
    allowed = {tool["function"]["name"] for tool in tools}
    system = BASE_SYSTEM + ("\n" + CI_DESCENT if mode == AgentMode.WIKITRACE else "")
    user = {
        "task": "Repair this CI failure while preserving required behavior and the original tests.",
        "current": observation,
        "budget": spec.budget.model_dump(mode="json"),
        "editable_paths": task.editable_paths,
        "protected_paths": task.protected_paths,
    }
    initial_text = json.dumps(user, ensure_ascii=False)
    if mode != AgentMode.NO_MEMORY:
        initial_text += "\n<memory_documents>\n" + memory.insights_text + "\n</memory_documents>"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": initial_text}]
    for message in messages:
        log({"type": "message", **message})

    try:
        while model_calls < spec.budget.model_calls:
            remaining = deadline - time.monotonic()
            tokens_left = spec.budget.total_tokens - input_tokens - output_tokens
            if remaining <= 0 or tokens_left < 128:
                stop_reason = "time-budget" if remaining <= 0 else "token-budget"
                break
            model_calls += 1
            turn = model.complete(
                messages, tools=tools,
                max_tokens=min(spec.budget.max_output_tokens, tokens_left),
                timeout=min(120, remaining),
            )
            input_tokens += turn.input_tokens
            output_tokens += turn.output_tokens
            reasoning_tokens += turn.reasoning_tokens
            cached_input_tokens += turn.cached_input_tokens
            returned_models.add(turn.returned_model)
            if len(returned_models) > 1:
                raise OpenAIError("The served model version changed during this trial")
            raw_calls = turn.message.get("tool_calls") or []
            assistant = {"role": "assistant", "content": turn.message.get("content")}
            if raw_calls:
                assistant["tool_calls"] = raw_calls
            if turn.continuation_items:
                messages.extend(turn.continuation_items)
            else:
                messages.append(assistant)
            log({"type": "message", **assistant, "usage": {"input_tokens": turn.input_tokens, "output_tokens": turn.output_tokens, "reasoning_tokens": turn.reasoning_tokens, "cached_input_tokens": turn.cached_input_tokens}, "returned_model": turn.returned_model})
            if turn.stop_reason:
                stop_reason = "model-output-" + turn.stop_reason
                break
            if not raw_calls:
                stop_reason = "agent-stopped"
                break
            if len(raw_calls) != 1:
                raise OpenAIError("The model returned parallel actions despite the shared serial-action protocol")
            call = raw_calls[0]
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not isinstance(call.get("function"), dict):
                raise OpenAIError("The model returned an invalid tool-call envelope")
            name = call.get("function", {}).get("name", "")
            is_error = False
            arguments = {}
            try:
                parsed = json.loads(call["function"]["arguments"], parse_constant=_reject_json_constant)
                json.dumps(parsed, allow_nan=False)
                arguments = parsed
                if not isinstance(arguments, dict) or name not in allowed:
                    raise ValueError("This tool is not available in the assigned agent mode")
                if time.monotonic() >= deadline:
                    stop_reason = "time-budget"
                    break
                if name == "set_plan":
                    current_plan = Decision.model_validate(arguments)
                    finish_evidence(current_plan)
                    result = {"recorded": current_plan.model_dump(mode="json")}
                elif name == "read_historical_evidence":
                    if mode != AgentMode.WIKITRACE or memory is None:
                        raise ValueError("Historical source evidence is not accessible in this mode")
                    if current_plan is None:
                        raise ValueError("Use set_plan before inspecting historical evidence")
                    result, citation = memory.read(
                        arguments["insight_id"], arguments["layer"], arguments.get("document", ""),
                        arguments.get("offset", 0), arguments.get("limit", 12000),
                    )
                    if pending_evidence is None:
                        pending_evidence = EvidenceDecision(
                            id=f"read-{len(evidence_decisions):04d}",
                            question=arguments["question"], trigger=arguments["trigger"],
                            before=current_plan.model_copy(deep=True), citations=[citation],
                        )
                    else:
                        pending_evidence.citations.append(citation)
                elif name == "submit_patch":
                    if current_plan is None or pending_evidence is not None:
                        raise ValueError("Record the current actionable plan with set_plan before patching")
                    if len(verifications) >= spec.budget.patch_attempts:
                        stop_reason = "patch-budget"
                        break
                    patch = backend.submit_patch(arguments["changes"], timeout_seconds=max(0.01, deadline - time.monotonic()))
                    verifications.append(patch.verification)
                    ci_wait += patch.verification.elapsed_seconds
                    hunks = patch.hunks
                    result = patch.observation
                elif name == "list_current_files":
                    finish_evidence()
                    result = {"files": backend.list_files()}
                elif name == "read_current_file":
                    finish_evidence()
                    result = {"path": arguments["path"], "content": backend.read_file(arguments["path"])}
                elif name == "read_current_log":
                    finish_evidence()
                    result = backend.read_log(arguments["entry_id"], arguments.get("offset", 0), arguments.get("limit", 12000))
                else:
                    raise ValueError("Unknown solver tool")
            except (ValueError, KeyError, TypeError) as exc:
                is_error = True
                result = {"error": str(exc)}
            log({"type": "function_call", "name": name, "arguments": arguments, "result": result, "is_error": is_error})
            if turn.continuation_items:
                messages.append({"type": "function_call_output", "call_id": call["id"], "output": json.dumps(result, ensure_ascii=False)})
            else:
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
            if verifications and verifications[-1].passed:
                outcome, stop_reason = "solved", "verified-by-target-ci-and-trusted-oracle"
                break
        else:
            stop_reason = "model-call-budget"
    except (OpenAIError, BackendError, EvidenceError, OSError) as exc:
        outcome, stop_reason = "infra-error", str(exc)
        log({"type": "infrastructure_error", "message": str(exc)})
    finally:
        finish_evidence()
        wall = time.monotonic() - start
        record = RunRecord(
            id=identifier, experiment_id=spec.id, experiment_sha256=fingerprint(spec), protocol_sha256=protocol_fingerprint(),
            task_id=task.id, task_sha256=fingerprint(task), mode=mode, trial=trial, role=role,
            model=spec.model, budget=spec.budget,
            corpus_sha256=memory.sha256 if mode != AgentMode.NO_MEMORY else None,
            insights_sha256=memory.insights_sha256 if mode != AgentMode.NO_MEMORY else None,
            outcome=outcome, stop_reason=stop_reason,
            timing=Timing(wall_seconds=wall, ci_wait_seconds=ci_wait),
            model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens, cached_input_tokens=cached_input_tokens,
            returned_models=sorted(returned_models), verifications=verifications,
            hunks=hunks, evidence_decisions=evidence_decisions, trajectory_path=relative_trajectory,
        )
        log({"type": "session_outcome", "record": record.model_dump(mode="json")})
        journal.close()
    with (directory / "result.json").open("x", encoding="utf-8") as stream:
        stream.write(record.model_dump_json(indent=2) + "\n")
    return record
