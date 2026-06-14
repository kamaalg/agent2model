"""Typer CLI — the primary user entry point.

The canonical user journey is ``compile`` → ``generate`` → ``train`` →
``eval``/``serve``. Each command prints the expected LLM cost before running and
the actual cost after, and exits with an actionable message on typed failures.

A ``cloud`` subcommand group wraps the generic Modal entrypoint:
``agent2model cloud run my_workflow.yaml --size 3b`` simply ``subprocess``-
invokes ``modal run -m agent2model.cloud.modal_app::run -- ...`` with the
typed flags mapped through, so users don't have to remember the ``modal run
-m`` incantation.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

import typer

from agent2model.exceptions import (
    FlowchartValidationError,
    GenerationBudgetExceeded,
    ServingError,
    TrainingDivergedError,
)
from agent2model.generation.formatter import write_dataset
from agent2model.generation.generator import (
    DEFAULT_MODEL,
    ConversationGenerator,
    GenerationConfig,
    estimate_cost,
)
from agent2model.ir.loader import load_flowchart
from agent2model.ir.render import to_mermaid, to_summary
from agent2model.ir.schema import Flowchart
from agent2model.ir.validator import validate
from agent2model.logging import configure_logging, logger
from agent2model.training.config import DENNIS_2026B, TrainingConfig

_JOURNEY_EPILOG = (
    "The canonical journey: init -> compile -> show -> generate -> train -> eval -> serve. "
    "init, compile, and show are free and offline. generate and eval call the Anthropic API "
    "(set ANTHROPIC_API_KEY; --budget caps spend, --dry-run previews cost for free). "
    "train and serve need a GPU (install the 'train' / 'serve' extras). "
    "No GPU? 'agent2model cloud run ...' runs the whole pipeline on Modal."
)

app = typer.Typer(
    name="agent2model",
    help="Turn your LangGraph agent into a small model that runs with no orchestrator.",
    epilog=_JOURNEY_EPILOG,
    no_args_is_help=True,
    add_completion=False,
)

cloud_app = typer.Typer(
    name="cloud",
    help="Run the pipeline on Modal (generic + paper reproductions).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(cloud_app, name="cloud")


def _version_callback(value: bool) -> None:
    """Print the installed version and exit (eager ``--version``)."""
    if value:
        from agent2model import __version__

        typer.echo(f"agent2model {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging.")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the agent2model version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Configure global logging before any command runs."""
    configure_logging(verbose=verbose)


def _require_anthropic_key() -> None:
    """Exit early with a clear message if no Anthropic API key is configured.

    Both ``generate`` and ``eval`` make Anthropic calls; without a key the SDK
    raises a raw traceback only *after* we have printed a cost estimate and
    started a progress bar. Fail fast and friendly instead.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set. Export your Anthropic API key first, e.g.\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Get a key at https://console.anthropic.com/."
        )
        raise typer.Exit(code=1)


#: Bundled example workflows shipped inside the wheel under ``agent2model/_examples``.
_BUNDLED_EXAMPLES = ("travel_booking", "zoom_support", "insurance_claims", "langgraph_demo")


def _examples_root() -> Path | None:
    """Locate the bundled examples directory.

    Examples ship *inside* the wheel at ``agent2model/_examples`` (see the
    ``force-include`` in ``pyproject.toml``), not in the user's working
    directory. In an editable/dev checkout that directory does not exist, so fall
    back to the repo's top-level ``examples/``. Returns ``None`` if neither is
    found.
    """
    packaged = resources.files("agent2model") / "_examples"
    if packaged.is_dir():
        return Path(str(packaged))
    # Editable/dev fallback: <repo>/examples, two levels up from this package.
    dev = Path(__file__).resolve().parent.parent.parent / "examples"
    return dev if dev.is_dir() else None


def _available_examples_str() -> str:
    """Comma-separated names of bundled examples, for error/help messages."""
    root = _examples_root()
    if root is None:
        return ", ".join(_BUNDLED_EXAMPLES)
    return ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir()))


def _validate_positive_budget(budget: float) -> None:
    """Reject a non-positive ``--budget`` with a friendly Typer error.

    ``GenerationConfig``/``EvalConfig`` constrain ``budget_usd > 0``; constructing
    them with ``--budget 0`` (a natural "spend nothing" guard) would otherwise
    surface a raw pydantic ``ValidationError`` traceback. Catch it at the CLI
    boundary instead.
    """
    if budget <= 0:
        raise typer.BadParameter(
            f"--budget must be greater than 0 (got {budget:g}). "
            "Use --dry-run to preview the estimated cost without spending."
        )


@app.command()
def compile(
    source: Annotated[
        Path, typer.Argument(help="Flowchart YAML, or a .py file defining a LangGraph graph.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Build directory for the compiled IR.")],
) -> None:
    """Validate a workflow and emit the canonical IR.

    Loads a YAML flowchart or a LangGraph ``.py`` source, enforces every graph
    invariant, and writes the normalised IR to ``<out>/flowchart.json``. IR
    derived from LangGraph contains TODO placeholder prompts (LangGraph nodes
    carry no natural-language instructions); these still validate and should be
    filled in before generating data.
    """
    if not source.exists():
        logger.error(
            f"No such file: {source}. Bundled examples ship inside the package, not your "
            "working directory — copy one first with `agent2model init <example>` (e.g. "
            "`agent2model init travel_booking`), then compile the copied path. "
            f"Available examples: {_available_examples_str()}."
        )
        raise typer.Exit(code=1)
    if not source.is_file():
        logger.error(
            f"Expected a flowchart YAML or a LangGraph .py file, but {source} is a directory. "
            "Point `compile` at the flowchart file itself (e.g. "
            f"`{source}/flowchart.yaml`), not the directory."
        )
        raise typer.Exit(code=1)
    try:
        if source.suffix == ".py":
            # Imported lazily so the CLI works without the optional langgraph extra.
            from agent2model.adapters.langgraph import (
                flowchart_from_stategraph,
                load_stategraph_from_pyfile,
            )

            graph = load_stategraph_from_pyfile(source)
            flowchart = flowchart_from_stategraph(graph, name=source.stem)
        else:
            flowchart = load_flowchart(source)
        validate(flowchart)
    except FlowchartValidationError as exc:
        for line in exc.errors:
            logger.error(line)
        raise typer.Exit(code=1) from exc

    out.mkdir(parents=True, exist_ok=True)
    ir_path = out / "flowchart.json"
    ir_path.write_text(
        json.dumps(flowchart.model_dump(mode="json"), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    # Emit a Mermaid diagram next to the IR — GitHub/Markdown render it inline, so
    # the user gets a shareable picture of their procedure for free.
    mmd_path = out / "flowchart.mmd"
    mmd_path.write_text(to_mermaid(flowchart) + "\n", encoding="utf-8")

    n_nodes = len(flowchart.nodes)
    n_terminals = len(flowchart.terminals)
    logger.info(
        f"Compiled '{flowchart.name}': {n_nodes} nodes, {n_terminals} terminals → {ir_path}"
    )
    # A short coverage summary so `compile` feels like it understood the procedure.
    for line in to_summary(flowchart).splitlines()[1:]:
        logger.info(line.strip())
    logger.info(f"Diagram written to {mmd_path} — view it with `agent2model show {out}`.")

    # Surface data-quality gaps the user must fix before generating (especially
    # for LangGraph-derived IR): placeholder TODO prompts and missing user turns.
    n_todo = sum(
        1
        for node in flowchart.nodes.values()
        if node.prompt is not None and node.prompt.strip().startswith("TODO:")
    )
    if n_todo:
        logger.warning(
            f"{n_todo} node(s) still have placeholder 'TODO:' prompts. Replace them with "
            f"real instructions in {ir_path} before running `agent2model generate`."
        )
    if not any(node.role == "user" for node in flowchart.nodes.values()):
        logger.warning(
            "No `role: user` nodes found; generated conversations would be agent-only "
            "monologue. Add user nodes where the customer speaks."
        )


@app.command()
def init(
    example: Annotated[
        str,
        typer.Argument(
            help="Bundled example to copy: travel_booking, zoom_support, "
            "insurance_claims, or langgraph_demo."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Destination directory. Defaults to ./<example>."),
    ] = None,
) -> None:
    """Copy a bundled example workflow into your working directory.

    The examples referenced in the quickstart ship *inside* the installed
    package, not in your current directory, so a literal ``pip install`` user has
    nothing to ``compile`` yet. This materialises one so the four-command journey
    works from a fresh install:

    Example:
        $ agent2model init travel_booking
        $ agent2model compile travel_booking/flowchart.yaml --out build/travel

    Args:
        example: Name of a bundled example.
        out: Destination directory; defaults to ``./<example>``.
    """
    root = _examples_root()
    if root is None:
        logger.error(
            "Could not locate the bundled examples. Reinstall agent2model, or copy an "
            "example from the source repository's examples/ directory."
        )
        raise typer.Exit(code=1)
    src = root / example
    if not src.is_dir():
        logger.error(f"Unknown example '{example}'. Available: {_available_examples_str()}.")
        raise typer.Exit(code=2)
    dest = out if out is not None else Path(example)
    if dest.exists():
        logger.error(f"Destination {dest} already exists; remove it or pass a different --out.")
        raise typer.Exit(code=1)
    shutil.copytree(src, dest)
    flowchart = dest / "flowchart.yaml"
    next_src = flowchart if flowchart.exists() else dest
    logger.info(
        f"Copied example '{example}' → {dest}. "
        f"Next: agent2model compile {next_src} --out build/{example}"
    )


@app.command()
def show(
    target: Annotated[
        Path,
        typer.Argument(
            help="A build dir (with flowchart.json), a flowchart YAML, or a LangGraph .py file."
        ),
    ],
    fmt: Annotated[
        str,
        typer.Option("--format", help="Output format: 'mermaid' (default) or 'summary'."),
    ] = "mermaid",
) -> None:
    """Visualise a procedure as a Mermaid diagram or a text summary.

    Renders the compiled IR so you can *see* the procedure instead of reading
    JSON. ``--format mermaid`` (default) prints a Mermaid ``flowchart`` — paste it
    into a ```` ```mermaid ```` fenced block and GitHub/Markdown render it inline.
    ``--format summary`` prints node/terminal/path counts. This is free, offline,
    and needs no API key or GPU.

    Example:
        $ agent2model show build/travel              # Mermaid diagram to stdout
        $ agent2model show build/travel --format summary
    """
    if not target.exists():
        logger.error(f"No such path: {target}.")
        raise typer.Exit(code=1)

    try:
        if target.is_dir():
            flowchart = _load_compiled_flowchart(target)
        elif target.suffix == ".py":
            from agent2model.adapters.langgraph import (
                flowchart_from_stategraph,
                load_stategraph_from_pyfile,
            )

            flowchart = flowchart_from_stategraph(
                load_stategraph_from_pyfile(target), name=target.stem
            )
            validate(flowchart)
        else:
            flowchart = load_flowchart(target)
            validate(flowchart)
    except FlowchartValidationError as exc:
        for line in exc.errors:
            logger.error(line)
        raise typer.Exit(code=1) from exc

    if fmt == "summary":
        typer.echo(to_summary(flowchart))
    elif fmt == "mermaid":
        typer.echo(to_mermaid(flowchart))
    else:
        raise typer.BadParameter(f"--format must be 'mermaid' or 'summary' (got {fmt!r}).")


def _load_compiled_flowchart(build_dir: Path) -> Flowchart:
    """Load and graph-validate the compiled ``flowchart.json`` from a build dir."""
    ir_path = build_dir / "flowchart.json"
    if not ir_path.exists():
        logger.error(f"No compiled flowchart at {ir_path}. Run `agent2model compile` first.")
        raise typer.Exit(code=1)
    flowchart = Flowchart.model_validate(json.loads(ir_path.read_text(encoding="utf-8")))
    validate(flowchart)
    return flowchart


@app.command()
def generate(
    build_dir: Annotated[
        Path, typer.Argument(help="Build directory holding the compiled flowchart.json.")
    ],
    n: Annotated[
        int, typer.Option("--n", min=1, help="Number of conversations to generate.")
    ] = 100,
    model: Annotated[str, typer.Option("--model", help="Anthropic model id.")] = DEFAULT_MODEL,
    budget: Annotated[
        float, typer.Option("--budget", help="Hard USD spending cap; generation stops if hit.")
    ] = 50.0,
    seed: Annotated[int, typer.Option("--seed", help="Base RNG seed for reproducibility.")] = 0,
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", min=1, help="Maximum in-flight API calls.")
    ] = 10,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes/--no-yes",
            "-y",
            help="Skip the cost-confirmation prompt (use for non-interactive runs).",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the estimated cost and exit without an API key or any API calls.",
        ),
    ] = False,
) -> None:
    """Generate synthetic training data by walking the compiled flowchart.

    Reads ``<BUILD_DIR>/flowchart.json``, samples ``--n`` conversations via
    Claude, prints the expected cost before starting and the actual cost after,
    and writes the HF chat-template dataset to ``<BUILD_DIR>/dataset.jsonl``.
    Generation is resumable and stops if the ``--budget`` cap is reached. Pass
    ``--dry-run`` to see the cost estimate with no API key and no spending.
    """
    _validate_positive_budget(budget)
    try:
        flowchart = _load_compiled_flowchart(build_dir)
    except FlowchartValidationError as exc:
        for line in exc.errors:
            logger.error(line)
        raise typer.Exit(code=1) from exc

    # Refuse to spend money turning placeholder prompts into garbage data.
    todo_nodes = [
        nid
        for nid, node in flowchart.nodes.items()
        if node.prompt is not None and node.prompt.strip().startswith("TODO:")
    ]
    if todo_nodes:
        logger.error(
            f"{len(todo_nodes)} node(s) still have placeholder 'TODO:' prompts "
            f"({', '.join(todo_nodes[:5])}{'…' if len(todo_nodes) > 5 else ''}). "
            "Replace them with real instructions before generating data."
        )
        raise typer.Exit(code=1)

    config = GenerationConfig(
        n=n, model=model, budget_usd=budget, seed=seed, max_concurrent=max_concurrent
    )
    # Estimate depends only on n/model/flowchart — no key needed — so print it
    # before the API-key check so cost is visible even without credentials.
    expected = estimate_cost(config)
    logger.info(f"Expected cost for {n} conversations with {model}: ~${expected:.2f}")
    if expected > budget:
        logger.warning(
            f"Expected cost ~${expected:.2f} exceeds the ${budget:.2f} budget; "
            "generation may stop before completing all conversations."
        )
    if dry_run:
        logger.info("Dry run: estimate only — no API key required, no API calls, no data written.")
        raise typer.Exit(code=0)

    _require_anthropic_key()
    if not yes and not typer.confirm(
        f"Proceed with generation (~${expected:.2f}, hard cap ${budget:.2f})?", default=True
    ):
        logger.info("Aborted before spending.")
        raise typer.Exit(code=0)

    generator = ConversationGenerator(flowchart, config)
    try:
        conversations = asyncio.run(generator.run(build_dir))
    except GenerationBudgetExceeded as exc:
        logger.error(str(exc))
        logger.error(f"Actual cost when stopped: ${generator.cost.cost_usd:.4f}")
        raise typer.Exit(code=1) from exc

    dataset_path = build_dir / "dataset.jsonl"
    written = write_dataset(conversations, dataset_path)
    logger.info(f"Actual cost: ${generator.cost.cost_usd:.4f}")
    logger.info(f"Wrote {written} conversations to {dataset_path}")


@app.command()
def train(
    build_dir: Annotated[
        Path, typer.Argument(help="Build directory holding the generated dataset.jsonl.")
    ],
    base: Annotated[
        str | None,
        typer.Option("--base", help="HF base model id. Defaults to the size preset's model."),
    ] = None,
    size: Annotated[
        str, typer.Option("--size", help="Model size preset: '3b' (single-GPU) or '8b' (ZeRO-3).")
    ] = "3b",
    epochs: Annotated[
        int | None,
        typer.Option("--epochs", help="Training epochs. Defaults to the preset (3B: 20, 8B: 10)."),
    ] = None,
    lora: Annotated[
        bool,
        typer.Option(
            "--lora/--no-lora",
            help="LoRA is NOT supported; full fine-tuning only. Passing --lora is refused.",
        ),
    ] = False,
) -> None:
    """Fine-tune a base model on generated data with the paper's recipe.

    Reads ``<BUILD_DIR>/dataset.jsonl`` (HF chat-template JSONL from
    ``agent2model generate``), builds a :class:`TrainingConfig` from the chosen
    ``--size`` preset, and runs full-parameter SFT, saving the best checkpoint
    (by held-out eval loss) to ``<BUILD_DIR>/model/best``.

    Full fine-tuning only: ``--lora`` is refused with a link to the companion
    paper. The heavy ML stack is GPU-only and not installed locally; if it is
    missing the command exits with an install hint rather than crashing.
    """
    size = size.lower()
    if size not in {"3b", "8b"}:
        logger.error(f"Unknown --size '{size}'. Use '3b' or '8b'.")
        raise typer.Exit(code=2)

    if lora:
        logger.error(
            "LoRA is not supported in agent2model v1: it fails to internalise procedural "
            f"workflows. See {DENNIS_2026B}. Re-run without --lora to use full fine-tuning."
        )
        raise typer.Exit(code=2)

    dataset_path = build_dir / "dataset.jsonl"
    if not dataset_path.exists():
        logger.error(f"No dataset at {dataset_path}. Run `agent2model generate {build_dir}` first.")
        raise typer.Exit(code=1)

    output_dir = str(build_dir / "model")
    overrides: dict[str, Any] = {}
    if base is not None:
        overrides["base_model"] = base
    if epochs is not None:
        overrides["epochs"] = epochs
    if size == "3b":
        config = TrainingConfig.for_3b(output_dir, **overrides)
    else:
        config = TrainingConfig.for_8b(output_dir, **overrides)

    logger.info(
        f"Training plan: {config.base_model} ({config.size}), {config.epochs} epochs, "
        f"effective batch size {config.effective_batch_size}, lr {config.learning_rate} "
        f"({config.lr_scheduler_type}). Best checkpoint by held-out eval loss "
        f"({config.eval_split:.0%} split) → {output_dir}/best."
    )
    if size == "8b":
        logger.info(
            f"8B uses DeepSpeed ZeRO-3 across {config.num_gpus} GPUs, launched via "
            "`accelerate launch` (run this on an 8x A100-class host)."
        )

    # Lazy import: keeps `agent2model train --help` working without the ML stack.
    # launch_training trains 3B in-process and launches 8B under accelerate+ZeRO-3.
    from agent2model.training.launch import launch_training as run_training

    try:
        best = run_training(config, dataset_path)
    except RuntimeError as exc:
        # Raised when the optional [train] extra / GPU host is unavailable.
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except TrainingDivergedError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    logger.info(f"Done. Best checkpoint (eval_loss={best.eval_loss}) saved to {best.path}.")


@app.command()
def eval(
    build_dir: Annotated[
        Path, typer.Argument(help="Build directory holding the compiled flowchart.json.")
    ],
    baselines: Annotated[
        str,
        typer.Option(
            "--baselines",
            help="Comma-separated baselines: in_context, langgraph, same_model_orch. "
            "Note: the 'langgraph' baseline needs the [langgraph] extra.",
        ),
    ] = "in_context",
    n: Annotated[int, typer.Option("--n", min=1, help="Scenarios per condition.")] = 200,
    judge_model: Annotated[
        str, typer.Option("--judge-model", help="Anthropic model id for the LLM judge.")
    ] = DEFAULT_MODEL,
    budget: Annotated[
        float, typer.Option("--budget", help="Hard USD spending cap across all LLM calls.")
    ] = 50.0,
    served_url: Annotated[
        str | None,
        typer.Option(
            "--served-url",
            help="OpenAI-compatible base URL of a `agent2model serve` endpoint; "
            "adds the served compiled model as a condition.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Base RNG seed.")] = 0,
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", min=1, help="Concurrent scenario evaluations.")
    ] = 10,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes/--no-yes",
            "-y",
            help="Skip the cost-confirmation prompt (use for non-interactive runs).",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the estimated cost and exit without an API key or any API calls.",
        ),
    ] = False,
) -> None:
    """Evaluate a compiled model against baselines with the paper's rubric.

    Samples ``--n`` scenarios, runs each condition (the baselines, plus the served
    ``compiled`` model when ``--served-url`` is given) against a flowchart-blind
    user simulator, judges every conversation on the 5-criterion rubric, computes
    bootstrap CIs / Wilcoxon + Holm-Bonferroni significance / failure rates / cost,
    and writes ``<BUILD_DIR>/eval_report.pdf`` and ``eval_report.json``. Prints
    the expected cost before starting and the actual cost after.
    """
    import asyncio

    from agent2model.eval.baselines import make_condition
    from agent2model.eval.judge import Judge, JudgeConfig
    from agent2model.eval.report import write_json_report, write_pdf_report
    from agent2model.eval.runner import EvalConfig, EvalRunner, estimate_eval_cost
    from agent2model.exceptions import EvalBudgetExceeded, EvalError

    _validate_positive_budget(budget)
    try:
        flowchart = _load_compiled_flowchart(build_dir)
    except FlowchartValidationError as exc:
        for line in exc.errors:
            logger.error(line)
        raise typer.Exit(code=1) from exc

    names = [b.strip() for b in baselines.split(",") if b.strip()]
    if served_url:
        names.append("compiled")
    else:
        logger.warning(
            "No --served-url given: this run evaluates only the baselines and does NOT "
            "score your compiled model. Start it with `agent2model serve <build_dir>` "
            "(needs a GPU) and pass --served-url to include the 'compiled' condition."
        )
    try:
        conditions = [make_condition(name, flowchart, served_url=served_url) for name in names]
    except EvalError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=2) from exc

    config = EvalConfig(
        n=n,
        budget_usd=budget,
        seed=seed,
        max_concurrent=max_concurrent,
        judge=JudgeConfig(model=judge_model),
    )
    expected = estimate_eval_cost(config, len(conditions))
    logger.info(
        f"Evaluating {len(conditions)} conditions ({', '.join(names)}) x {n} scenarios. "
        f"Expected cost: ~${expected:.2f}"
    )
    if expected > budget:
        logger.warning(
            f"Expected cost ~${expected:.2f} exceeds the ${budget:.2f} budget; "
            "the run may stop before completing."
        )
    if dry_run:
        logger.info(
            "Dry run: estimate only — no API key required, no API calls, no report written."
        )
        raise typer.Exit(code=0)

    _require_anthropic_key()
    if not yes and not typer.confirm(
        f"Proceed with evaluation (~${expected:.2f}, hard cap ${budget:.2f})?", default=True
    ):
        logger.info("Aborted before spending.")
        raise typer.Exit(code=0)

    runner = EvalRunner(flowchart, conditions, config, judge=Judge(config.judge))
    try:
        result = asyncio.run(runner.run())
    except EvalBudgetExceeded as exc:
        logger.error(str(exc))
        logger.error(f"Actual cost when stopped: ${runner.cost.cost_usd:.4f}")
        raise typer.Exit(code=1) from exc

    logger.info(f"Actual cost: ${result.total_cost_usd:.4f}")
    json_path = write_json_report(result, build_dir / "eval_report.json")
    logger.info(f"Wrote {json_path}")
    try:
        pdf_path = write_pdf_report(result, build_dir / "eval_report.pdf")
        logger.info(f"Wrote {pdf_path}")
    except EvalError as exc:
        logger.warning(str(exc))


@app.command()
def serve(
    build_dir: Annotated[
        Path,
        typer.Argument(help="Build directory holding the compiled model (or a model dir)."),
    ],
    port: Annotated[int, typer.Option("--port", help="TCP port to bind.")] = 8000,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Interface to bind. Defaults to 127.0.0.1 (local only); the endpoint "
            "is unauthenticated, so use 0.0.0.0 only on a trusted network.",
        ),
    ] = "127.0.0.1",
    model_name: Annotated[
        str | None,
        typer.Option("--model-name", help="Public model id exposed via the API."),
    ] = None,
) -> None:
    """Serve a compiled model via an OpenAI-compatible vLLM endpoint.

    Resolves the servable checkpoint under ``<BUILD_DIR>`` (prefers
    ``<BUILD_DIR>/best`` from ``agent2model train``, falling back to the
    directory itself), prints what it is about to serve and on what address,
    then launches vLLM's OpenAI-compatible API server (``/v1/chat/completions``,
    ``/v1/models``). vLLM is GPU/CUDA-only; if it is not installed the command
    exits with an actionable install hint rather than crashing.
    """
    from agent2model.serve.vllm_server import resolve_model_path
    from agent2model.serve.vllm_server import serve as run_server

    try:
        model_path = resolve_model_path(build_dir)
    except ServingError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    logger.info(
        f"Serving '{model_name or model_path}' on http://{host}:{port} "
        "(OpenAI-compatible: /v1/chat/completions, /v1/models)."
    )
    try:
        run_server(model_path, port=port, host=host, served_model_name=model_name)
    except ServingError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc


#: Default modal-run target for ``agent2model cloud run``. The ``::run`` suffix
#: selects the generic ``@app.local_entrypoint`` in ``modal_app.py``.
MODAL_RUN_TARGET = "agent2model.cloud.modal_app::run"


def _build_modal_run_argv(
    flowchart_path: Path,
    *,
    name: str | None,
    size: str,
    n: int,
    epochs: int,
    eval_n: int,
    base_model: str | None,
    skip_eval: bool,
    serve_after: bool,
    yes: bool = False,
    modal_bin: str = "modal",
) -> list[str]:
    """Build the argv to invoke the generic Modal entrypoint via ``modal run -m``.

    Pure helper kept separate so unit tests can assert on the constructed command
    without spawning Modal. Mirrors :func:`agent2model.cloud.modal_app.run`'s
    parameters one-for-one, dasherising them for the ``modal run`` CLI.

    Args:
        flowchart_path: Path to the flowchart (resolved by the caller).
        name: Optional recipe name override.
        size: Training size preset (``"3b"`` / ``"8b"``).
        n: Number of conversations.
        epochs: Training epochs.
        eval_n: Eval scenarios per condition.
        base_model: Optional HF base model override.
        skip_eval: If True, append ``--skip-eval``.
        serve_after: If True, append ``--serve-after``.
        yes: If True, append ``--yes`` to skip the modal entrypoint's
            cost-confirmation prompt (required for non-interactive use).
        modal_bin: The ``modal`` executable path (default ``"modal"``).

    Returns:
        The argv list to hand to :mod:`subprocess`.

    Example:
        >>> _build_modal_run_argv(Path("/tmp/wf.yaml"), name=None, size="3b",
        ...     n=2000, epochs=20, eval_n=200, base_model=None,
        ...     skip_eval=False, serve_after=False)
        ['modal', 'run', '-m', 'agent2model.cloud.modal_app::run', '--',
         '--flowchart-path', '/tmp/wf.yaml', '--size', '3b',
         '--n', '2000', '--epochs', '20', '--eval-n', '200']
    """
    argv: list[str] = [
        modal_bin,
        "run",
        "-m",
        MODAL_RUN_TARGET,
        "--",
        "--flowchart-path",
        str(flowchart_path),
        "--size",
        size,
        "--n",
        str(n),
        "--epochs",
        str(epochs),
        "--eval-n",
        str(eval_n),
    ]
    if name is not None:
        argv += ["--name", name]
    if base_model is not None:
        argv += ["--base-model", base_model]
    if skip_eval:
        argv.append("--skip-eval")
    if serve_after:
        argv.append("--serve-after")
    if yes:
        argv.append("--yes")
    return argv


@cloud_app.command("run")
def cloud_run(
    flowchart_path: Annotated[
        Path, typer.Argument(help="Flowchart YAML, or a .py file defining a LangGraph graph.")
    ],
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help="Recipe name (volume subdir). Defaults to the YAML's name or the file stem.",
        ),
    ] = None,
    size: Annotated[str, typer.Option("--size", help="Training preset: '3b' or '8b'.")] = "3b",
    n: Annotated[int, typer.Option("--n", help="Number of conversations to generate.")] = 2000,
    epochs: Annotated[int, typer.Option("--epochs", help="Training epochs.")] = 20,
    eval_n: Annotated[
        int, typer.Option("--eval-n", help="Scenarios per evaluation condition.")
    ] = 200,
    base_model: Annotated[
        str | None,
        typer.Option(
            "--base-model",
            help="HF base model id. Defaults to the size preset's model.",
        ),
    ] = None,
    skip_eval: Annotated[
        bool,
        typer.Option("--skip-eval/--no-skip-eval", help="Skip the evaluation step."),
    ] = False,
    serve_after: Annotated[
        bool,
        typer.Option(
            "--serve-after/--no-serve-after",
            help="Launch the autoscaling serve endpoint after training.",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes/--no-yes",
            "-y",
            help="Skip the cost-confirmation prompt on the Modal entrypoint (use for CI).",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the modal-run command that would be invoked and exit.",
        ),
    ] = False,
) -> None:
    """Run the generic Modal pipeline on a user-supplied flowchart.

    Builds and invokes ``modal run -m agent2model.cloud.modal_app::run -- ...``
    in a subprocess, mapping the typed flags through. Modal must be installed
    (``pip install agent2model[cloud]``) and authenticated.

    Args:
        flowchart_path: Path to a ``.yaml`` / ``.yml`` flowchart or LangGraph
            ``.py``.
        name: Recipe name; defaults to the YAML's ``name`` field or file stem.
        size: ``"3b"`` (single GPU) or ``"8b"`` (8x A100 ZeRO-3).
        n: Number of conversations to generate.
        epochs: Training epochs.
        eval_n: Scenarios per evaluation condition.
        base_model: HF base model id; defaults to the size preset.
        skip_eval: Skip the evaluation step.
        serve_after: Launch the autoscaling serve endpoint after training.
        yes: Pass ``--yes`` through to the Modal entrypoint so it skips the
            interactive cost-confirmation prompt. Required for non-interactive
            invocations.
        dry_run: Print the constructed command and exit without running it.
    """
    resolved = flowchart_path.expanduser().resolve()
    if not resolved.exists():
        logger.error(f"No such flowchart: {resolved}")
        raise typer.Exit(code=1)

    if name is not None:
        from agent2model.cloud._recipes import validate_recipe_name

        try:
            validate_recipe_name(name)
        except ValueError as exc:
            logger.error(str(exc))
            raise typer.Exit(code=2) from exc

    argv = _build_modal_run_argv(
        resolved,
        name=name,
        size=size,
        n=n,
        epochs=epochs,
        eval_n=eval_n,
        base_model=base_model,
        skip_eval=skip_eval,
        serve_after=serve_after,
        yes=yes,
    )

    rendered = " ".join(argv)
    if dry_run:
        typer.echo(rendered)
        return

    logger.info(f"Launching: {rendered}")
    try:
        result = subprocess.run(argv, check=False)
    except FileNotFoundError as exc:
        logger.error(
            "Could not find the `modal` executable on PATH. Install the cloud "
            "extra (`pip install agent2model[cloud]`) and run `modal setup`."
        )
        raise typer.Exit(code=1) from exc
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@cloud_app.command("doctor")
def cloud_doctor() -> None:
    """Run the cloud-preflight checklist and print a green/red summary.

    Validates, in order: ``modal`` is installed; the local Modal token exists;
    the ``anthropic-secret`` Modal Secret resolves; the local
    ``ANTHROPIC_API_KEY`` bills (informational); a Hugging Face token is valid
    if one is configured (informational). Exits with code 0 when every critical
    check passes, 1 otherwise. Informational red lines never flip the code.
    """
    from rich.console import Console

    from agent2model.cloud.doctor import overall_exit_code, run_all_checks

    console = Console()
    results = run_all_checks()
    console.print("[bold]agent2model cloud doctor[/bold]")
    for r in results:
        mark = "[green]+[/green]" if r.ok else "[red]x[/red]"
        sev = "" if r.severity == "critical" else " [dim](info)[/dim]"
        console.print(f"  {mark} {r.name}{sev}: {r.message}")
        if not r.ok and r.fix_command:
            console.print(f"      [dim]fix:[/dim] {r.fix_command}")
    code = overall_exit_code(results)
    if code == 0:
        console.print("[green]All critical checks passed.[/green]")
    else:
        console.print("[red]One or more critical checks failed; see fixes above.[/red]")
    raise typer.Exit(code=code)


class _TyperWizardIO:
    """:class:`WizardIO` implementation that delegates to Typer prompts."""

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        """Yes/no prompt via :func:`typer.confirm`."""
        return bool(typer.confirm(prompt, default=default))

    def prompt_hidden(self, prompt: str) -> str:
        """Hidden-input prompt via :func:`typer.prompt`."""
        return str(typer.prompt(prompt, hide_input=True))

    def echo(self, message: str) -> None:
        """Forward a one-line status message to stdout."""
        typer.echo(message)


@cloud_app.command("setup")
def cloud_setup() -> None:
    """First-time cloud wizard: account check, Modal token, Anthropic Secret.

    Idempotent — every step inspects current state and only acts when something
    is missing. After the steps complete, runs the doctor checklist as a final
    summary. Designed so a user can re-run it at any time without breaking
    anything.
    """
    from rich.console import Console

    from agent2model.cloud.setup import run_setup

    console = Console()
    console.print("[bold]agent2model cloud setup[/bold]")
    io = _TyperWizardIO()
    results = run_setup(io)
    for r in results:
        marker = {
            "already_done": "[green]+[/green]",
            "completed": "[green]+[/green]",
            "skipped": "[yellow]-[/yellow]",
            "user_declined": "[yellow]-[/yellow]",
            "failed": "[red]x[/red]",
        }[r.outcome]
        console.print(f"  {marker} {r.step}: {r.message}")

    console.print(
        "\nSetup complete. Running `agent2model cloud doctor` to verify your environment.\n"
    )
    # Reuse the same doctor command so output stays in lock-step.
    from agent2model.cloud.doctor import overall_exit_code, run_all_checks

    doctor_results = run_all_checks()
    for check in doctor_results:
        mark = "[green]+[/green]" if check.ok else "[red]x[/red]"
        sev = "" if check.severity == "critical" else " [dim](info)[/dim]"
        console.print(f"  {mark} {check.name}{sev}: {check.message}")
        if not check.ok and check.fix_command:
            console.print(f"      [dim]fix:[/dim] {check.fix_command}")

    if overall_exit_code(doctor_results) == 0:
        console.print("\n[green]Ready.[/green] Try: agent2model cloud run my_workflow.yaml")
    else:
        console.print(
            "\n[yellow]Some checks still failing.[/yellow] Address them above and re-run "
            "`agent2model cloud setup` or `agent2model cloud doctor`."
        )


if __name__ == "__main__":
    app()
