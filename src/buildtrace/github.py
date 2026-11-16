"""Read GitHub Actions evidence; preserve the selected run attempt."""

import base64
import binascii
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import PurePosixPath
from urllib.parse import quote, urlsplit

import httpx

from buildtrace.models import ApiDocument, CollectedRun, RunReference, SourceFile

API_ROOT = "https://api.github.com"
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
SOURCE_PATHS = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-ci.txt",
    "ci/constraints.txt",
    ".python-version",
)


class GitHubError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def resolve_github_token() -> str | None:
    """Use an environment token, or the user's existing GitHub CLI sign-in."""
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        if token := os.environ.get(name, "").strip():
            return token
    if executable := shutil.which("gh"):
        try:
            result = subprocess.run(
                [executable, "auth", "token", "--hostname", "github.com"],
                capture_output=True,
                text=True,
                timeout=5,
                env=os.environ | {"GH_PROMPT_DISABLED": "1"},
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


class GitHubActionsClient:
    def __init__(self, token: str | None = None, client: httpx.Client | None = None):
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(20, connect=10))
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "BuildTrace/0.1.0",
        }
        token = resolve_github_token() if token is None else token
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._documents: list[ApiDocument] = []

    def __enter__(self) -> "GitHubActionsClient":
        return self

    def __exit__(self, *_args) -> None:
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _check_status(response: httpx.Response) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        if status == 401:
            message = "GitHub rejected the credentials. Sign in with gh or configure GH_TOKEN."
        elif status in {403, 429} and response.headers.get("x-ratelimit-remaining") == "0":
            message = "GitHub's API rate limit was reached. Configure authentication or retry after the reset."
        elif status == 403:
            message = "GitHub denied access. Check repository access and Actions read permission."
        elif status == 404:
            message = "GitHub could not find the requested run or file, or the account cannot access it."
        elif status in {409, 410}:
            message = "GitHub's execution logs are not available yet or have expired."
        elif status >= 500:
            message = "GitHub is temporarily unavailable. Please retry."
        else:
            message = f"GitHub returned HTTP {status}."
        raise GitHubError(message, status)

    def _get_json(self, path: str) -> dict:
        try:
            response = self._client.get(API_ROOT + path, headers=self._headers, follow_redirects=False)
            self._check_status(response)
            value = response.json()
        except httpx.RequestError as exc:
            raise GitHubError("Could not reach GitHub. Check the connection and retry.") from exc
        except ValueError as exc:
            raise GitHubError("GitHub returned an invalid JSON response.") from exc
        if not isinstance(value, dict):
            raise GitHubError("GitHub returned an unexpected response format.")
        self._documents.append(ApiDocument(endpoint=path, content=response.content))
        return value

    def _list(self, path: str, key: str) -> list[dict]:
        records = []
        for page in range(1, 51):
            result = self._get_json(f"{path}?per_page=100&page={page}")
            batch = result.get(key)
            if not isinstance(batch, list) or any(not isinstance(item, dict) for item in batch):
                raise GitHubError(f"GitHub returned an invalid {key} list.")
            records.extend(batch)
            if len(batch) < 100:
                return records
        raise GitHubError(f"This run exceeds the viewer's 5,000-item {key} limit.")

    @staticmethod
    def _validate_download_url(url: str) -> None:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        allowed = (
            host == "api.github.com"
            or host.endswith(".blob.core.windows.net")
            or host.endswith(".githubusercontent.com")
        )
        if parsed.scheme != "https" or not allowed or parsed.username or parsed.password:
            raise GitHubError("GitHub returned an unsupported log download location.")
        try:
            if parsed.port not in {None, 443}:
                raise GitHubError("GitHub returned an unsupported log download port.")
        except ValueError as exc:
            raise GitHubError("GitHub returned an invalid log download location.") from exc

    def _download_logs(self, path: str) -> bytes:
        url = API_ROOT + path
        headers = self._headers.copy()
        try:
            for _ in range(4):
                request = self._client.build_request("GET", url, headers=headers)
                # A client's default headers must not leak credentials to signed storage URLs.
                if not url.startswith(API_ROOT + "/"):
                    request.headers.pop("Authorization", None)
                response = self._client.send(request, stream=True, follow_redirects=False)
                try:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        url = response.headers.get("location", "")
                        self._validate_download_url(url)
                        headers = {"User-Agent": self._headers["User-Agent"]}
                        continue
                    self._check_status(response)
                    chunks = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > MAX_ARCHIVE_BYTES:
                            raise GitHubError("The log archive exceeds the viewer's 20 MiB download limit.")
                        chunks.append(chunk)
                    return b"".join(chunks)
                finally:
                    response.close()
        except httpx.RequestError as exc:
            raise GitHubError("The log download was interrupted. Please retry.") from exc
        raise GitHubError("GitHub returned too many log download redirects.")

    def _get_source(self, reference: RunReference, path: str, sha: str) -> SourceFile | None:
        endpoint = f"/repos/{reference.full_name}/contents/{quote(path, safe='/')}?ref={sha}"
        try:
            result = self._get_json(endpoint)
        except GitHubError as exc:
            if exc.status_code == 404:
                return None
            raise
        if result.get("type") != "file" or result.get("encoding") != "base64":
            raise GitHubError(f"Could not archive {path}: unsupported file representation.")
        if result.get("size", 0) > MAX_SOURCE_BYTES:
            raise GitHubError(f"Could not archive {path}: source file exceeds 1 MiB.")
        try:
            encoded = "".join(result["content"].split())
            content = base64.b64decode(encoded, validate=True)
        except (KeyError, ValueError, TypeError, binascii.Error) as exc:
            raise GitHubError(f"Could not decode the source file {path}.") from exc
        if len(content) > MAX_SOURCE_BYTES:
            raise GitHubError(f"Could not archive {path}: source file exceeds 1 MiB.")
        return SourceFile(path=path, content=content)

    def collect(self, reference: RunReference) -> CollectedRun:
        self._documents = []
        latest = self._get_json(reference.api_base)
        attempt = reference.attempt or latest.get("run_attempt", 1)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise GitHubError("GitHub returned an invalid run attempt.")
        reference = replace(reference, attempt=attempt)
        attempt_base = f"{reference.api_base}/attempts/{attempt}"
        run = self._get_json(attempt_base)
        if run.get("id") != reference.run_id or run.get("run_attempt", attempt) != attempt:
            raise GitHubError("GitHub returned a different workflow run attempt than requested.")
        sha = run.get("head_sha", "")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", sha):
            raise GitHubError("GitHub returned an invalid source revision.")
        capture = CollectedRun(
            reference=reference,
            run=run,
            jobs=self._list(attempt_base + "/jobs", "jobs"),
            source_ref=sha,
        )
        if run.get("event") in {"pull_request", "pull_request_target"}:
            capture.warnings.append(
                "Pull-request workflows can check out a merge revision. Source files below use "
                "the reported head SHA; inspect checkout evidence before treating it as the tested revision."
            )
        try:
            capture.log_archive = self._download_logs(attempt_base + "/logs")
        except GitHubError as exc:
            capture.warnings.append(f"Log archive unavailable: {exc}")
        try:
            capture.artifacts = self._list(reference.api_base + "/artifacts", "artifacts")
        except GitHubError as exc:
            capture.warnings.append(f"Artifact listing unavailable: {exc}")

        workflow_path = str(run.get("path", "")).split("@", 1)[0]
        workflow = PurePosixPath(workflow_path)
        paths = list(SOURCE_PATHS)
        if workflow.parent == PurePosixPath(".github/workflows") and workflow.suffix in {".yml", ".yaml"}:
            paths.insert(0, workflow_path)
        else:
            capture.warnings.append("The workflow path could not be resolved to a repository YAML file.")
        for path in paths:
            try:
                if source := self._get_source(reference, path, sha):
                    capture.sources.append(source)
            except GitHubError as exc:
                capture.warnings.append(str(exc))
        capture.api_documents = self._documents.copy()
        return capture
