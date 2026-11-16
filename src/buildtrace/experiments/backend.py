"""Revision-scoped GitHub trial tools and an independent, trusted CI verification run."""

import base64
import copy
import difflib
import fnmatch
import hashlib
import io
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote

import httpx
import yaml

from buildtrace.evidence import EvidenceStore
from buildtrace.experiments.models import ChangedHunk, TaskSpec, Verification, fingerprint
from buildtrace.github import API_ROOT, GitHubActionsClient, GitHubError, resolve_github_token
from buildtrace.models import FAILED_CONCLUSIONS, RunReference, SourceFile


class BackendError(RuntimeError):
    """An infrastructure failure that cannot be counted as a verified repair."""


@dataclass
class PatchResult:
    verification: Verification
    hunks: list[ChangedHunk]
    observation: dict


def safe_repository_path(path: str) -> str:
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ".." in parsed.parts or "\\" in path or ":" in path or path.startswith(".git/"):
        raise ValueError("Use a relative project-file path")
    return parsed.as_posix()


def changed_hunks(originals: dict[str, str | None], overlay: dict[str, str | None]) -> list[ChangedHunk]:
    hunks = []
    for path, content in sorted(overlay.items()):
        before = (originals[path] or "").splitlines(keepends=True)
        after = (content or "").splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        for group in matcher.get_grouped_opcodes(0):
            old_start, old_end = group[0][1], group[-1][2]
            new_start, new_end = group[0][3], group[-1][4]
            lines = [f"--- a/{path}\n+++ b/{path}\n@@ -{old_start + 1},{old_end - old_start} +{new_start + 1},{new_end - new_start} @@\n"]
            added = deleted = 0
            for tag, a0, a1, b0, b1 in group:
                if tag in {"delete", "replace"}:
                    lines.extend("-" + line for line in before[a0:a1])
                    deleted += a1 - a0
                if tag in {"insert", "replace"}:
                    lines.extend("+" + line for line in after[b0:b1])
                    added += b1 - b0
            diff = "".join(lines)
            hunks.append(ChangedHunk(
                id="hunk-" + hashlib.sha256((path + "\0" + diff).encode()).hexdigest()[:20],
                path=path, added_lines=added, deleted_lines=deleted, diff=diff,
            ))
    return hunks


def required_checks_passed(task: TaskSpec, capture: dict) -> bool:
    if capture["run"].get("conclusion") != "success":
        return False
    for name, required in task.required_steps.items():
        jobs = [job for job in capture["jobs"] if job.get("name") == name]
        if not jobs or any(job.get("conclusion") != "success" for job in jobs):
            return False
        for job in jobs:
            for step_name in required:
                matches = [step for step in job.get("steps", []) if step.get("name") == step_name]
                if not matches or any(step.get("conclusion") != "success" for step in matches):
                    return False
    return True


def junit_passed(archive_bytes: bytes, expected_tests: int) -> bool:
    """Require all original tests, no failures/errors/skips, from the trusted oracle run."""
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            reports = [info for info in archive.infolist() if PurePosixPath(info.filename).name == "pytest.xml"]
            if len(reports) != 1 or reports[0].file_size > 2 * 1024 * 1024:
                return False
            root = ET.fromstring(archive.read(reports[0]))
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
        counts = [int(suite.attrib.get(key, "0")) for suite in suites for key in ("failures", "errors", "skipped")]
        return tests == expected_tests and all(value == 0 for value in counts) and len(root.findall(".//testcase")) == expected_tests
    except (zipfile.BadZipFile, ET.ParseError, KeyError, ValueError, RuntimeError):
        return False


def validate_workflow_environment_change(before: str, after: str, allowed_keys: list[str]):
    """Configuration cases can repair declared env settings while preserving CI commands and checks."""
    def project(text):
        try:
            document = yaml.load(text, Loader=yaml.BaseLoader)
        except yaml.YAMLError as exc:
            raise ValueError("The workflow must remain valid YAML") from exc
        if not isinstance(document, dict):
            raise ValueError("The workflow must remain a YAML mapping")
        document = copy.deepcopy(document)
        environments = {}

        def remove_env(mapping, location):
            environment = mapping.pop("env", {})
            if not isinstance(environment, dict):
                raise ValueError("The controlled CI lab requires explicit environment mappings")
            environments[location] = environment

        remove_env(document, "workflow")
        jobs = document.get("jobs", {})
        if not isinstance(jobs, dict):
            raise ValueError("The workflow must retain its job mappings")
        for name, job in jobs.items():
            if not isinstance(job, dict) or not isinstance(job.get("steps", []), list):
                raise ValueError("The workflow must retain its job and step mappings")
            remove_env(job, "job:" + name)
            for number, step in enumerate(job.get("steps", [])):
                if not isinstance(step, dict):
                    raise ValueError("The workflow must retain its step mappings")
                remove_env(step, f"job:{name}/step:{number}")
        return document, environments

    original, old_env = project(before)
    proposed, new_env = project(after)
    if original != proposed:
        raise ValueError("Workflow repairs may change declared environment settings, not commands, triggers, or checks")
    for location in old_env.keys() | new_env.keys():
        old, new = old_env.get(location, {}), new_env.get(location, {})
        for key in old.keys() | new.keys():
            if old.get(key) != new.get(key) and key not in allowed_keys:
                raise ValueError(f"The task does not allow changing workflow environment key {key}")


class GitHubTrialBackend:
    def __init__(self, task: TaskSpec, store: EvidenceStore, *, token: str | None = None, client: httpx.Client | None = None, allow_publish: bool = False, namespace: str):
        self.task, self.store = task, store
        self.token = resolve_github_token() if token is None else token
        if not self.token:
            raise BackendError("Configure GitHub authentication for trial branches and Actions runs.")
        if not allow_publish:
            raise BackendError("Live experiments create trial branches and run CI. Pass --allow-publish to authorize them.")
        if not re.fullmatch(r"[a-zA-Z0-9-]{1,50}", namespace):
            raise ValueError("Invalid trial namespace")
        self.namespace = namespace
        self.client = client or httpx.Client(timeout=20)
        self._owns_client = client is None
        self.headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        self.base = f"/repos/{task.repository}"
        self.overlay, self.originals = {}, {}
        self.attempt = 0
        self.allowed_captures = {task.capture_id}
        capture = store.load(task.capture_id)
        if task.capture_sha256 is not None and fingerprint(capture) != task.capture_sha256:
            raise ValueError("The task's initial evidence manifest was modified")
        store.verify(task.capture_id)
        reference = RunReference.parse(task.initial_run_url)
        if (
            capture["source_ref"] != task.base_sha
            or capture["reference"]["run_id"] != reference.run_id
            or capture["reference"]["attempt"] != reference.attempt
            or capture["reference"]["owner"] + "/" + capture["reference"]["repository"] != task.repository
        ):
            raise ValueError("The archived CI attempt does not match the task snapshot")
        if capture["run"].get("event") not in {"push", "workflow_dispatch"}:
            raise ValueError("The initial CI benchmark supports push and manual-dispatch snapshots")
        if capture["run"].get("conclusion") not in FAILED_CONCLUSIONS:
            raise ValueError("The input must be a known failing CI run")
        self.initial_capture = capture

    def close(self):
        if self._owns_client:
            self.client.close()

    def _api(self, method, path, payload=None):
        try:
            response = self.client.request(method, API_ROOT + self.base + path, headers=self.headers, json=payload)
            GitHubActionsClient._check_status(response)
            return response.json() if response.content else {}
        except (httpx.RequestError, GitHubError, ValueError) as exc:
            raise BackendError(f"GitHub trial operation failed ({method} {path.split('?')[0]}).") from exc

    def pin_oracle(self) -> TaskSpec:
        ref = self._api("GET", "/git/ref/heads/" + quote(self.task.oracle_ref, safe=""))
        sha = ref["object"]["sha"]
        if self.task.oracle_sha is not None and sha != self.task.oracle_sha:
            raise BackendError("The trusted verification branch moved; create a new, explicitly pinned experiment.")
        workflow = self._api("GET", "/contents/.github/workflows/" + quote(self.task.workflow, safe="") + f"?ref={sha}")
        document = yaml.safe_load(base64.b64decode(workflow["content"]))
        # PyYAML's YAML 1.1 parser may read the GitHub key 'on' as True.
        triggers = document.get("on", document.get(True, {}))
        inputs = triggers.get("workflow_dispatch", {}).get("inputs", {}) if isinstance(triggers, dict) else {}
        if not {"source_sha", "oracle_mode"}.issubset(inputs):
            raise BackendError("Publish the lab workflow's trusted source_sha/oracle_mode inputs before live experiments.")
        self.task = self.task.model_copy(update={"oracle_sha": sha, "capture_sha256": fingerprint(self.initial_capture)})
        return self.task

    def initial_observation(self) -> dict:
        return {
            "repository": self.task.repository, "base_sha": self.task.base_sha,
            "initial_run_url": self.task.initial_run_url,
            "failed_jobs": [{"name": job.get("name"), "steps": [step.get("name") for step in job.get("steps", []) if step.get("conclusion") in FAILED_CONCLUSIONS]} for job in self.initial_capture["jobs"] if job.get("conclusion") in FAILED_CONCLUSIONS],
            "current_logs": self._log_catalog(self.initial_capture),
        }

    @staticmethod
    def _log_catalog(capture):
        return [{"entry_id": f"{capture['id']}/{entry['id']}", "name": entry["label"], "bytes": entry["byte_count"]} for entry in capture["entries"] if entry["kind"] == "log"]

    def list_files(self):
        tree = self._api("GET", f"/git/trees/{self.task.base_sha}?recursive=1")
        if tree.get("truncated"):
            raise BackendError("The task's file tree was truncated by GitHub")
        base_paths = {item["path"] for item in tree["tree"] if item["type"] == "blob"}
        base_paths.update(path for path, content in self.overlay.items() if content is not None)
        base_paths.difference_update(path for path, content in self.overlay.items() if content is None)
        return sorted(base_paths)

    def _read_base(self, path):
        if path not in self.originals:
            try:
                result = self.client.get(API_ROOT + self.base + "/contents/" + quote(path, safe="/") + f"?ref={self.task.base_sha}", headers=self.headers)
                if result.status_code == 404:
                    self.originals[path] = None
                    return None
                GitHubActionsClient._check_status(result)
                body = result.json()
            except (httpx.RequestError, GitHubError, ValueError) as exc:
                raise BackendError("Could not read a file from the fixed task snapshot") from exc
            if isinstance(body, list):
                raise ValueError("This path is a directory. Use list_current_files and read an exact file path.")
            if not isinstance(body, dict):
                raise BackendError("GitHub returned an invalid source-file response")
            try:
                if body.get("type") != "file" or body.get("encoding") != "base64" or body.get("size", 0) > 100000:
                    raise ValueError("Unsupported source file")
                self.originals[path] = base64.b64decode(body["content"]).decode("utf-8")
            except (ValueError, KeyError) as exc:
                raise BackendError("Could not read a file from the fixed task snapshot") from exc
        return self.originals[path]

    def read_file(self, path):
        path = safe_repository_path(path)
        content = self.overlay[path] if path in self.overlay else self._read_base(path)
        if content is None:
            raise ValueError("This file does not exist in the current task state")
        if len(content) > 30000:
            raise ValueError("This source file exceeds the current agent's 30,000-character limit")
        return content

    def read_log(self, entry_id, offset=0, limit=12000):
        if not isinstance(entry_id, str):
            raise ValueError("Choose a string entry ID from the current-log catalog")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12000:
            raise ValueError("Invalid current-log range")
        capture_id, separator, file_id = entry_id.partition("/")
        if not separator or capture_id not in self.allowed_captures:
            raise ValueError("Only this trial's current CI evidence is accessible")
        entry, content = self.store.read_entry(capture_id, file_id)
        if entry["kind"] != "log":
            raise ValueError("Choose an entry from the current-log catalog")
        text = content.decode("utf-8", errors="replace")
        if offset > len(text):
            raise ValueError("Current-log offset is outside the document")
        excerpt = text[offset:offset + limit]
        return {"text": excerpt, "offset": offset, "next_offset": offset + len(excerpt) if offset + len(excerpt) < len(text) else None, "sha256": entry["sha256"]}

    def _check_changes(self, changes):
        if not isinstance(changes, list) or not 1 <= len(changes) <= 6:
            raise ValueError("Submit between one and six changed files")
        updated, seen = self.overlay.copy(), set()
        for change in changes:
            if not isinstance(change, dict) or set(change) != {"path", "content"}:
                raise ValueError("Each change needs only path and content")
            path = safe_repository_path(change["path"])
            if path in seen:
                raise ValueError("A patch cannot repeat a file path")
            seen.add(path)
            protected = any(path.startswith(pattern) if pattern.endswith("/") else fnmatch.fnmatchcase(path, pattern) for pattern in self.task.protected_paths)
            if protected or not any(fnmatch.fnmatchcase(path, pattern) for pattern in self.task.editable_paths):
                raise ValueError(f"The task does not allow changing {path}")
            content = change["content"]
            if content is not None and (not isinstance(content, str) or len(content.encode("utf-8")) > 100000):
                raise ValueError("Each new file must contain at most 100,000 UTF-8 bytes")
            self._read_base(path)
            if path.startswith(".github/workflows/"):
                if path != f".github/workflows/{self.task.workflow}" or self.originals[path] is None or content is None:
                    raise ValueError("Only the existing target workflow's declared environment settings are editable")
                validate_workflow_environment_change(self.originals[path], content, self.task.editable_environment_keys)
            updated[path] = content
        if not any(content != self.originals[path] for path, content in updated.items()):
            raise ValueError("The proposed patch does not change the failing snapshot")
        return updated

    def _dispatch_and_capture(self, ref, expected_head, inputs, deadline, expected_title=None, archive_oracle=False):
        if time.monotonic() >= deadline:
            raise BackendError("The trial's CI verification deadline was reached")
        path = "/actions/workflows/" + quote(self.task.workflow, safe="") + "/runs"
        query = "?event=workflow_dispatch&branch=" + quote(ref, safe="") + f"&head_sha={expected_head}&per_page=100"
        existing = {run["id"] for run in self._api("GET", path + query)["workflow_runs"]}
        self._api("POST", "/actions/workflows/" + quote(self.task.workflow, safe="") + "/dispatches", {"ref": ref, "inputs": inputs})
        while time.monotonic() < deadline:
            runs = self._api("GET", path + query)
            candidates = [run for run in runs["workflow_runs"] if run["id"] not in existing and run["head_sha"] == expected_head and (expected_title is None or run.get("display_title") == expected_title)]
            if candidates and candidates[0]["status"] == "completed":
                run = candidates[0]
                reference = RunReference.parse(run["html_url"] + f"/attempts/{run.get('run_attempt', 1)}")
                with GitHubActionsClient(token=self.token, client=self.client) as client:
                    collected = client.collect(reference)
                    if archive_oracle:
                        self._attach_oracle_evidence(collected, client)
                    capture = self.store.save(collected)
                self.allowed_captures.add(capture["id"])
                return capture
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        raise BackendError("The trial's CI verification deadline was reached")

    def _attach_oracle_evidence(self, collected, client):
        artifacts = [artifact for artifact in collected.artifacts if artifact.get("name") == f"invoice-ci-{collected.reference.run_id}-{collected.reference.attempt}" and not artifact.get("expired")]
        if len(artifacts) != 1:
            collected.warnings.append("A unique current-attempt oracle artifact was not available.")
            return
        try:
            archive = client._download_logs(f"{self.base}/actions/artifacts/{artifacts[0]['id']}/zip")
        except GitHubError as exc:
            raise BackendError("Could not download the oracle's original verification artifact") from exc
        collected.attachments.append(SourceFile("oracle-evidence.zip", archive))
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                for name in ("pytest.xml", "environment-before-install.json"):
                    matches = [info for info in bundle.infolist() if PurePosixPath(info.filename).name == name and info.file_size <= 2 * 1024 * 1024]
                    if len(matches) == 1:
                        collected.attachments.append(SourceFile(name, bundle.read(matches[0])))
        except (zipfile.BadZipFile, RuntimeError):
            collected.warnings.append("The oracle artifact could not be decoded.")

    def _oracle_passed(self, capture, candidate_sha):
        if not required_checks_passed(self.task, capture):
            return False
        entries = [entry for entry in capture["entries"] if entry["kind"] == "artifact" and entry["label"] == "oracle-evidence.zip"]
        if len(entries) != 1:
            return False
        _, archive = self.store.read_entry(capture["id"], entries[0]["id"])
        if not junit_passed(archive, self.task.expected_test_count):
            return False
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                matches = [info for info in bundle.infolist() if PurePosixPath(info.filename).name == "environment-before-install.json" and info.file_size < 100000]
                if len(matches) != 1:
                    return False
                return json.loads(bundle.read(matches[0]))["checked_out_sha"] == candidate_sha
        except (zipfile.BadZipFile, ValueError, KeyError):
            return False

    def submit_patch(self, changes, *, timeout_seconds):
        if self.task.oracle_sha is None:
            raise BackendError("Pin and validate the trusted oracle before starting a trial")
        updated = self._check_changes(changes)
        current_ref = self._api("GET", "/git/ref/heads/" + quote(self.task.oracle_ref, safe=""))
        if current_ref["object"]["sha"] != self.task.oracle_sha:
            raise BackendError("The trusted verification branch moved during the experiment")
        start = time.monotonic()
        deadline = start + timeout_seconds
        base_commit = self._api("GET", "/git/commits/" + self.task.base_sha)
        tree_entries = []
        for path, content in updated.items():
            if content == self.originals[path]:
                continue
            blob = self._api("POST", "/git/blobs", {"content": base64.b64encode(content.encode()).decode(), "encoding": "base64"}) if content is not None else None
            tree_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"] if blob else None})
        tree = self._api("POST", "/git/trees", {"base_tree": base_commit["tree"]["sha"], "tree": tree_entries})
        self.attempt += 1
        commit = self._api("POST", "/git/commits", {"message": f"BuildTrace experimental patch {self.namespace} attempt {self.attempt}", "tree": tree["sha"], "parents": [self.task.base_sha]})
        branch = f"buildtrace-trials/{self.namespace}-p{self.attempt}"
        self._api("POST", "/git/refs", {"ref": "refs/heads/" + branch, "sha": commit["sha"]})
        self.overlay = updated
        target = self._dispatch_and_capture(branch, commit["sha"], {}, deadline)
        oracle = self._dispatch_and_capture(self.task.oracle_ref, self.task.oracle_sha, {"source_sha": commit["sha"], "oracle_mode": True}, deadline, expected_title=f"Oracle {commit['sha']}", archive_oracle=True)
        verification = Verification(
            attempt=self.attempt, candidate_sha=commit["sha"], run_url=target["run_url"],
            oracle_run_url=oracle["run_url"], workflow_conclusion=target["run"].get("conclusion", "unknown"),
            required_checks_passed=required_checks_passed(self.task, target),
            protected_files_unchanged=True, oracle_passed=self._oracle_passed(oracle, commit["sha"]),
            elapsed_seconds=time.monotonic() - start, capture_id=target["id"],
            oracle_capture_id=oracle["id"],
        )
        return PatchResult(verification, changed_hunks(self.originals, self.overlay), {
            "verification": verification.model_dump(mode="json"),
            "current_logs": self._log_catalog(target), "oracle_logs": self._log_catalog(oracle),
        })
