"""A small, server-rendered browser for archived Actions evidence."""

import io
import os
import re
from collections.abc import Callable
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from buildtrace import __version__
from buildtrace.evidence import EvidenceError, EvidenceStore
from buildtrace.github import GitHubActionsClient, GitHubError
from buildtrace.models import FAILED_CONCLUSIONS
from buildtrace.service import capture_run

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def create_app(
    config: dict | None = None,
    *,
    github_factory: Callable[[], GitHubActionsClient] = GitHubActionsClient,
) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        DATA_DIR=os.environ.get("BUILDTRACE_DATA_DIR", "data"),
        MAX_CONTENT_LENGTH=8 * 1024,
    )
    if config:
        app.config.update(config)
    store = EvidenceStore(app.config["DATA_DIR"])
    app.extensions["evidence_store"] = store

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        status = 200
        run_url = ""
        if request.method == "POST":
            run_url = request.form.get("run_url", "").strip()
            try:
                manifest = capture_run(run_url, store, github_factory)
                return redirect(url_for("capture_detail", capture_id=manifest["id"]), code=303)
            except ValueError as exc:
                error, status = str(exc), 400
            except GitHubError as exc:
                error, status = str(exc), 502
            except OSError:
                error, status = "The evidence archive could not be saved. Check the data directory.", 500
        return render_template("index.html", recent=store.recent(), error=error, run_url=run_url), status

    @app.get("/captures/<capture_id>")
    def capture_detail(capture_id: str):
        manifest = store.load(capture_id)
        failed_jobs = [job for job in manifest["jobs"] if job.get("conclusion") in FAILED_CONCLUSIONS]
        failed_names = [
            re.sub(r"[^a-z0-9]", "", step.get("name", "").lower())
            for job in failed_jobs
            for step in job.get("steps", [])
            if step.get("conclusion") in FAILED_CONCLUSIONS and step.get("name")
        ]
        previews = []
        for entry in manifest["entries"]:
            if entry["kind"] not in {"source", "log"}:
                continue
            _, content = store.read_entry(capture_id, entry["id"])
            lines = _ANSI.sub("", content.decode("utf-8-sig", errors="replace")).splitlines()
            normalized = re.sub(r"[^a-z0-9]", "", entry["label"].lower())
            previews.append(
                entry
                | {
                    "preview": "\n".join(lines[:140]),
                    "line_count": len(lines),
                    "truncated": len(lines) > 140,
                    "failure_match": any(name in normalized for name in failed_names),
                }
            )
        logs = sorted(
            (entry for entry in previews if entry["kind"] == "log"),
            key=lambda entry: (not entry["failure_match"], entry["label"]),
        )
        return render_template(
            "capture.html",
            capture=manifest,
            failed_jobs=failed_jobs,
            logs=logs,
            sources=[entry for entry in previews if entry["kind"] == "source"],
            api_entries=[entry for entry in manifest["entries"] if entry["kind"] == "api"],
            archive=next((entry for entry in manifest["entries"] if entry["kind"] == "archive"), None),
        )

    @app.get("/captures/<capture_id>/files/<entry_id>")
    def evidence_file(capture_id: str, entry_id: str):
        entry, content = store.read_entry(capture_id, entry_id)
        kind = entry["kind"]
        binary_archive = kind == "archive" or entry["path"].endswith(".zip")
        mimetype = "application/zip" if binary_archive else "application/json" if kind == "api" else "text/plain"
        return send_file(
            io.BytesIO(content),
            mimetype=mimetype,
            as_attachment=binary_archive or request.args.get("download") == "1",
            download_name=Path(entry["path"]).name,
            max_age=0,
        )

    @app.get("/api/captures/<capture_id>")
    def capture_manifest(capture_id: str):
        return jsonify(store.load(capture_id))

    @app.get("/health")
    def health():
        return jsonify(status="ok", version=__version__)

    @app.errorhandler(FileNotFoundError)
    def missing_capture(_error):
        return render_template("error.html", message="That evidence capture or file does not exist."), 404

    @app.errorhandler(EvidenceError)
    def invalid_evidence(error):
        return render_template("error.html", message=str(error)), 409

    @app.errorhandler(413)
    def oversized_request(_error):
        return render_template("error.html", message="Submit a single GitHub Actions run URL."), 413

    return app
