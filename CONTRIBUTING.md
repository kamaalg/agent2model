# Contributing

Thanks for your interest in `agent2model`. This guide covers the conventions CI
enforces and the release process.

## Ways to contribute

- **New examples** — a small `flowchart.yaml` + README for a procedure (it must pass
  `agent2model compile` and `agent2model show`). No API key or GPU needed — the easiest
  first PR.
- **Docs, troubleshooting, FAQ** — if something tripped you up, fix the doc.
- **Bug fixes** — see the open [good first issues](https://github.com/kamaalg/agent2model/issues).

### Adapters wanted

The biggest force-multiplier is **new framework adapters** — each one lets a whole
ecosystem compile their existing agents with zero YAML. The LangGraph adapter
(`src/agent2model/adapters/langgraph.py`) is the worked reference; an adapter recovers a
framework's procedure *structure* (nodes, edges, decision branches, terminals) into the
Flowchart IR and emits honest lossy-conversion warnings. On the wishlist:

- **CrewAI Flows** → IR ([#5](https://github.com/kamaalg/agent2model/issues/5))
- **OpenAI Agents SDK** → IR ([#6](https://github.com/kamaalg/agent2model/issues/6))
- Anything else with a declared graph/flow (LlamaIndex Workflows, Pydantic AI, …)

Open an issue first to align on the IR mapping, then mirror the LangGraph adapter's
shape (structure recovery + `TODO:` prompt placeholders + warnings). See
[`docs/adapters.md`](https://github.com/kamaalg/agent2model/blob/main/docs/adapters.md).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tooling/tests + langgraph
pip install -e ".[dev,docs]"     # add the docs site toolchain
```

## Before you push

CI runs these and fails on any of them; run them locally first:

```bash
ruff check .
black --check src tests
mypy src
pytest tests/unit -q
```

Test tiers (see `pyproject.toml` markers):

- **unit** — fast, mocked, no network/GPU; runs on every PR.
- **integration** — real Anthropic API, tiny budgets; nightly (`-m integration`).
- **e2e** — full reproduction, compared to the paper; release candidates
  (`-m e2e`). Skipped unless `AGENT2MODEL_E2E=1` and a built model are present.

Coding conventions (see `CLAUDE.md` for the full list): Python 3.11+, modern type
hints, Pydantic v2 for user-facing schemas, Typer for CLI, Loguru for logging,
async + semaphores for batched API work, Google-style docstrings, no bare
`# type: ignore`.

## Conventional commits

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
so the changelog and version bumps are automatic. Examples:

```
feat(ir): add scenario-variable range sampling
fix(generation): resume from the correct checkpoint after a 429
docs(eval): document the Holm-Bonferroni correction
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`. A `!`
after the type/scope (or a `BREAKING CHANGE:` footer) marks a breaking change.

## Cutting a release

Releases are driven by [commitizen](https://commitizen-tools.github.io/commitizen/):

```bash
cz bump                 # bumps version, updates CHANGELOG.md, creates a vX.Y.Z tag
git push --follow-tags
```

Pushing the tag triggers `.github/workflows/release.yml`, which builds the sdist
and wheel and publishes to PyPI via Trusted Publishing (OIDC — no token secret).
Benchmark numbers in `benchmarks/` are updated on every minor release; if they
regress beyond the 5% gate, that blocks the release.
