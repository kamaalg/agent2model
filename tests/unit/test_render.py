"""Unit tests for the Mermaid / summary renderers (``agent2model.ir.render``)."""

from __future__ import annotations

from agent2model.ir.render import to_mermaid, to_summary
from agent2model.ir.schema import Flowchart


def test_mermaid_has_header_and_all_nodes(travel_flowchart: Flowchart) -> None:
    """Every node id appears and the diagram opens with the flowchart header."""
    mermaid = to_mermaid(travel_flowchart)
    assert mermaid.startswith("flowchart TD")
    for node_id in travel_flowchart.nodes:
        assert node_id in mermaid


def test_mermaid_shapes_encode_roles(travel_flowchart: Flowchart) -> None:
    """Decision nodes render as diamonds and terminals as coloured stadiums."""
    mermaid = to_mermaid(travel_flowchart)
    # assess_readiness is a decision node -> {"..."}
    assert 'assess_readiness{"assess_readiness"}' in mermaid
    # booking_confirmed is a success terminal -> (["..."]) + a class assignment
    assert 'booking_confirmed(["booking_confirmed"])' in mermaid
    assert "class booking_confirmed success;" in mermaid
    assert "classDef success" in mermaid


def test_mermaid_edge_labels_are_short_and_escaped() -> None:
    """``when`` labels are truncated and embedded quotes are escaped."""
    fc = Flowchart.model_validate(
        {
            "name": "q",
            "start": "a",
            "nodes": {
                "a": {
                    "role": "agent",
                    "prompt": "x",
                    "next": [{"to": "done", "when": 'user said "yes" ' + "z" * 100}],
                },
                "done": {"terminal": "success"},
            },
        }
    )
    mermaid = to_mermaid(fc)
    assert "&quot;yes&quot;" in mermaid  # quotes escaped
    assert "…" in mermaid  # long label truncated


def test_mermaid_sanitises_unsafe_ids() -> None:
    """Node ids with spaces/punctuation become valid Mermaid identifiers."""
    fc = Flowchart.model_validate(
        {
            "name": "q",
            "start": "first step",
            "nodes": {
                "first step": {"role": "agent", "prompt": "x", "next": ["done!"]},
                "done!": {"terminal": "success"},
            },
        }
    )
    mermaid = to_mermaid(fc)
    # Original ids survive as labels; declarations use sanitised identifiers.
    assert 'first_step["first step"]' in mermaid
    assert "first_step --> done_" in mermaid


def test_summary_counts(travel_flowchart: Flowchart) -> None:
    """Summary reports node role counts, terminals, and path stats."""
    summary = to_summary(travel_flowchart)
    assert "14 nodes" in summary
    assert "3 decision" in summary
    assert "3 terminal" in summary
    assert "abandonment: 1" in summary
    assert "Distinct start→terminal paths: 3" in summary
