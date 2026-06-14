"""Render a Flowchart IR as a Mermaid diagram or a text summary.

The compiled IR is otherwise an invisible JSON blob; these renderers turn it into
something a user can *see*. :func:`to_mermaid` emits a `Mermaid
<https://mermaid.js.org/>`_ ``flowchart`` that GitHub, GitLab, and most Markdown
viewers render inline — so a user (or a launch README) gets a shareable diagram of
the procedure for free. :func:`to_summary` is a dependency-free text overview for
the terminal.

Neither renderer makes network or LLM calls; both operate purely on a validated
:class:`~agent2model.ir.schema.Flowchart`.
"""

from __future__ import annotations

import re

from agent2model.ir.schema import Flowchart, Node
from agent2model.ir.validator import enumerate_paths

#: Mermaid ``classDef`` name per terminal kind, used to colour terminal nodes.
_TERMINAL_CLASS: dict[str, str] = {
    "success": "success",
    "abandonment": "abandon",
    "escalation": "escalate",
}

#: Colour palette injected as Mermaid ``classDef`` directives.
_CLASS_DEFS = (
    "classDef start fill:#cce5ff,stroke:#004085,color:#004085;",
    "classDef success fill:#d4edda,stroke:#28a745,color:#155724;",
    "classDef abandon fill:#f8d7da,stroke:#dc3545,color:#721c24;",
    "classDef escalate fill:#fff3cd,stroke:#ffc107,color:#856404;",
)


def _escape_label(text: str) -> str:
    """Make ``text`` safe inside a Mermaid node/edge label."""
    text = text.replace('"', "&quot;").replace("\n", "<br/>")
    return text


def _short(text: str, limit: int = 48) -> str:
    """Collapse whitespace and truncate a label so diagrams stay legible."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) > limit:
        return collapsed[: limit - 1].rstrip() + "…"
    return collapsed


def _safe_ids(flowchart: Flowchart) -> dict[str, str]:
    """Map each (possibly arbitrary) node id to a unique Mermaid-safe identifier."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for node_id in flowchart.nodes:
        base = re.sub(r"\W", "_", node_id) or "n"
        if base[0].isdigit():
            base = f"n_{base}"
        candidate = base
        suffix = 0
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        mapping[node_id] = candidate
    return mapping


def _node_decl(node_id: str, node: Node, sid: str) -> str:
    """Emit the Mermaid shape declaration for a single node.

    Shapes encode the role: rectangle = agent turn, rounded = user turn,
    diamond = decision, stadium = terminal (coloured by outcome).
    """
    label = _escape_label(node_id)
    if node.is_terminal:
        return f'{sid}(["{label}"])'
    if node.role == "decision":
        return f'{sid}{{"{label}"}}'
    if node.role == "user":
        return f'{sid}("{label}")'
    return f'{sid}["{label}"]'  # agent (default)


def to_mermaid(flowchart: Flowchart) -> str:
    """Render ``flowchart`` as a Mermaid ``flowchart TD`` definition.

    Args:
        flowchart: A validated flowchart.

    Returns:
        The Mermaid source as a string (no surrounding code fence). Wrap it in a
        ```` ```mermaid ```` fence to render it in Markdown.

    Example:
        >>> print(to_mermaid(fc))  # doctest: +SKIP
        flowchart TD
            greet["greet"]
            ...
    """
    ids = _safe_ids(flowchart)
    lines: list[str] = ["flowchart TD"]

    for node_id, node in flowchart.nodes.items():
        lines.append(f"    {_node_decl(node_id, node, ids[node_id])}")

    lines.append("")
    for node_id, node in flowchart.nodes.items():
        for edge in node.next:
            if edge.to not in ids:
                continue  # dangling edge; the validator reports it separately
            if edge.when:
                lines.append(
                    f'    {ids[node_id]} -->|"{_escape_label(_short(edge.when))}"| {ids[edge.to]}'
                )
            else:
                lines.append(f"    {ids[node_id]} --> {ids[edge.to]}")

    lines.append("")
    for definition in _CLASS_DEFS:
        lines.append(f"    {definition}")

    # Assign classes: start node + terminals coloured by outcome.
    if flowchart.start in ids:
        lines.append(f"    class {ids[flowchart.start]} start;")
    for node_id, node in flowchart.terminals.items():
        cls = _TERMINAL_CLASS.get(node.terminal or "", "")
        if cls:
            lines.append(f"    class {ids[node_id]} {cls};")

    return "\n".join(lines)


def to_summary(flowchart: Flowchart, *, max_paths: int = 1000) -> str:
    """Build a dependency-free text overview of the procedure's shape.

    Reports node counts by role, the terminal breakdown by outcome, the number of
    distinct ``start`` → terminal paths (bounded by ``max_paths``), and the
    longest such path — enough for a user to feel the CLI *understood* their graph.

    Args:
        flowchart: A validated flowchart.
        max_paths: Cap on path enumeration (see :func:`enumerate_paths`).

    Returns:
        A multi-line summary string.
    """
    roles = {"agent": 0, "user": 0, "decision": 0}
    for node in flowchart.nodes.values():
        if node.role in roles:
            roles[node.role] += 1

    terminals_by_kind: dict[str, int] = {}
    for node in flowchart.terminals.values():
        kind = node.terminal or "?"
        terminals_by_kind[kind] = terminals_by_kind.get(kind, 0) + 1

    paths = list(enumerate_paths(flowchart, max_paths=max_paths))
    n_paths = len(paths)
    longest = max((len(p) for p in paths), default=0)
    capped = " (capped)" if n_paths >= max_paths else ""

    terminal_str = ", ".join(f"{k}: {v}" for k, v in sorted(terminals_by_kind.items())) or "none"
    lines = [
        f"Procedure '{flowchart.name}': {len(flowchart.nodes)} nodes "
        f"({roles['agent']} agent, {roles['user']} user, {roles['decision']} decision, "
        f"{len(flowchart.terminals)} terminal)",
        f"  Terminals by outcome: {terminal_str}",
        f"  Distinct start→terminal paths: {n_paths}{capped}",
        f"  Longest path: {longest} nodes",
    ]
    return "\n".join(lines)
