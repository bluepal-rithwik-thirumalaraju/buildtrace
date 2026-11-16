# CI lab scenarios

This catalog is for the experiment operator. Keep it, reference repairs, and
other agents' repair branches outside the debugging agent's task context.

Target repository:
<https://github.com/bluepal-rithwik-thirumalaraju/buildtrace-ci-lab>

## Healthy baseline

- Branch: `main`
- Initial revision: `c6906b43c1d27598d52da2ecc4950018aa2215ad`
- Initial passing workflow: <https://github.com/bluepal-rithwik-thirumalaraju/buildtrace-ci-lab/actions/runs/43809926047>
- Local verification: 36 tests pass; package builds; example total is `99.00`.

## case-001

- Category: application test failure.
- Branch: `case-001`
- Failing revision: `c7de58d8094fe4c2c5a952c0eb9888c417eb39f6`
- Workflow: <https://github.com/bluepal-rithwik-thirumalaraju/buildtrace-ci-lab/actions/runs/43810145176>
- Mutation: the final total uses the subtotal instead of the amount remaining
  after the discount. Tax calculation, dependencies, workflow, and tests are unchanged.
- Expected failure: 5 tests fail, 31 pass; the example's total is `109.00`
  instead of `99.00`.
- Success condition: restore correct invoice behavior and pass all 36 original
  tests without weakening them.

### Reproduce locally

In a checkout of the **lab** repository, with its development environment installed:

```sh
git switch case-001
python -m pytest
```

Return to the healthy branch:

```sh
git switch main
```

### Capture the failure in BuildTrace

From the **agent** repository:

```sh
python -m buildtrace inspect https://github.com/bluepal-rithwik-thirumalaraju/buildtrace-ci-lab/actions/runs/43810145176
python -m buildtrace serve
```

Open <http://127.0.0.1:8000> and select the saved capture. Record its capture ID
and verify the archive with `python -m buildtrace verify CAPTURE_ID`.

For a repair attempt, branch from the fixed failing revision into a new repair
branch. Run CI on the new patched commit; re-running the original failed run
would execute the original code.

## Next cases

Add dependency conflicts, import failures, and missing configuration as isolated
revisions. Then add held-out lookalike failures whose historical solutions have
different applicability conditions. Pin source revisions and retain the same
tests and budgets across agent modes.
