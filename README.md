# BuildTrace

Evidence-first CI failure investigation, inspired by WikiTrace.

[![BuildTrace CI](https://github.com/bluepal-rithwik-thirumalaraju/buildtrace/actions/workflows/ci.yml/badge.svg)](https://github.com/bluepal-rithwik-thirumalaraju/buildtrace/actions/workflows/ci.yml)

## The two repositories

| Repository | Role |
| --- | --- |
| **This repository: BuildTrace** | Investigate failed runs, preserve evidence, and build the debugging agent |
| [buildtrace-ci-lab](https://github.com/bluepal-rithwik-thirumalaraju/buildtrace-ci-lab) | Invoice calculator, tests, GitHub Actions, and controlled failure revisions |

## Current milestone: failure viewer and evidence capture

Paste a GitHub Actions run URL into the local webpage, or use the CLI, to:

- Identify the exact workflow run attempt and its failing jobs and steps.
- Archive the original log ZIP and API response bodies.
- Read log previews and open the complete original log files.
- Archive the workflow and supported dependency files at the reported head SHA.
- Preserve immutable captures with SHA-256 checksums for every evidence file.
- Reopen captures offline and verify their recorded file integrity.

Each capture keeps the run attempt separate. For pull-request events, the viewer
labels the reported head SHA explicitly because the actual checkout can be a
merge revision. Run-level artifact listings are also labeled separately from
attempt-specific logs.

The next milestone is agent diagnosis and repair, followed by the linked
insight/compilation/raw-trajectory memory and the three-mode comparison.
See the complete [implementation plan](BUILDTRACE_PLAN.md).

## Quick start

Use Python 3.11 or newer; CI uses Python 3.12.

```sh
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# macOS / Linux
source .venv/bin/activate
```

Install and run:

```sh
python -m pip install --editable ".[dev]" --constraint ci/constraints.txt
python -m buildtrace serve
```

Open <http://127.0.0.1:8000> and paste a workflow run URL. The server binds to
localhost by default.

### GitHub authentication

BuildTrace uses `GH_TOKEN` or `GITHUB_TOKEN` when set, or an existing
`gh auth login` sign-in when the GitHub CLI is installed. Authentication is useful
for API limits and required for private resources. The viewer needs repository
contents and Actions read access.

Credentials remain in the process environment or credential provider; they are
not written into captures or sent to the log storage host on redirects.

### Try the first real failure

Paste this run into the webpage:

<https://github.com/bluepal-rithwik-thirumalaraju/buildtrace-ci-lab/actions/runs/43810145176>

Or capture it from the command line:

```sh
python -m buildtrace inspect https://github.com/bluepal-rithwik-thirumalaraju/buildtrace-ci-lab/actions/runs/43810145176
```

The command prints a capture ID. Verify that capture's recorded checksums:

```sh
python -m buildtrace verify CAPTURE_ID
```

An `inspect` command succeeds when evidence was collected; its JSON output
separately reports whether the CI run succeeded or failed.

## Evidence layout

```text
data/captures/<capture-id>/
  manifest.json           Run metadata, provenance, warnings, and file checksums
  raw/api-0001.json       Original GitHub API response body
  raw/logs.zip            Original attempt log archive
  logs/log-0001.txt       Original member bytes under generated safe filenames
  repository/...         Source files retrieved at the reported head SHA
```

Capturing a run again creates a new capture. Existing captures are never
overwritten by the application. Log previews are limited to 140 lines; complete
files and the original ZIP remain available. Collection limits are 20 MiB for a
downloaded log archive, 1 MiB per source file, 5 MiB per expanded log preview,
50 MiB total expanded logs, and 500 log files. Missing, expired, or oversized
evidence is reported in capture notes.

`data/`, local credentials, virtual environments, and the nested local lab
checkout are Git-ignored. The lab has its own Git history and remote.

## Development

```sh
python -m pip check
python -m pytest
python -m build
```

Tests cover attempt-specific ingestion, pagination, download credential handling,
missing logs, evidence integrity, URL and archive-path handling, and the browser
and CLI flows. Tests use simulated HTTP responses; the documented lab run is the
live integration input.

The [scenario catalog](docs/LAB_SCENARIOS.md) is operator-only evaluation material
and must stay outside the target-project context supplied to a debugging agent.

## Design references

- [BuildTrace implementation plan](BUILDTRACE_PLAN.md).
- [WikiTrace source paper](sources/2611.14320v1.pdf), Sections 3-5 and Appendices B and F.
- [GitHub Actions run logs](https://docs.github.com/en/actions/how-tos/monitor-workflows/using-workflow-run-logs).
- [GitHub Actions rerun behavior](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).
