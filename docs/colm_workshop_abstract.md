# COLM 2026 — Lifelong Agents Workshop: Draft Abstract

**Status:** Draft — not yet submitted.
**Deadline:** July 3, 2026 (AoE)
**Portal:** OpenReview — https://openreview.net/group?id=colmweb.org%2FCOLM%2F2026%2FWorkshop%2FLLA
**Workshop site:** https://lifelongagent.github.io/ (returned 403 during automated fetch — verify page/format limits directly in a browser before submitting)

---

## CFP snapshot (from web search — verify against the live page)

| Field | Value |
|---|---|
| Deadline | **July 3, 2026 (AoE)** |
| Notification | July 24, 2026 (AoE) |
| Workshop date | October 9, 2026 |
| Archival | **Non-archival** |
| Portal | OpenReview (link above) |
| Format | COLM 2026 LaTeX template; single PDF; refs + appendices outside page limit |
| Page limit | **Not confirmed** — CFP page returned 403 during this run; check the live site |
| Topics | Agent post-training, agentic RL, user-agent alignment, self-evolving agents, lifelong/embodied agents |

> **Scope fit:** The strongest fit is "agent post-training" — compiling a declared procedure into a small model's weights via SFT — and the eval harness as a reusable contribution for measuring agent behaviour. The "lifelong" angle: compiled models sustain stable procedural behaviour across long deployments without continuous orchestration overhead.

---

## Proposed title

**Procedure Compilation for Small Models: An Open Pipeline and Reusable Procedure-Adherence Evaluation Harness**

---

## Draft abstract (~265 words)

Running conversational agents via external orchestrators incurs per-turn calls to
frontier models — expensive, privacy-sensitive, and latency-bound. Dennis et al. (2026)
showed that compiling a declared agent procedure directly into a small model's weights,
via supervised fine-tuning on synthetically generated conversations, can deliver
near-frontier quality at two orders of magnitude lower inference cost. We present
**agent2model**, an open-source library that makes this compilation pipeline reproducible
and accessible for arbitrary procedural workflows.

Our contributions are twofold. First, we introduce a **Flowchart IR**: a minimal YAML
specification for declarative agent procedures, paired with a LangGraph adapter that
imports existing `StateGraph` workflows automatically. A four-stage CLI pipeline
(`compile`, `generate`, `train`, `eval`) converts any flowchart into a fine-tuned small
model. The generation stage uses Claude Sonnet with prompt caching to produce thousands
of synthetic training conversations; critically, the flowchart structure never appears in
the training data, so the model must internalise the procedure rather than pattern-match
on schema tokens.

Second, we release a **procedure-adherence evaluation harness** that is independently
useful regardless of how the model was trained. The harness combines: (1) a dynamic user
simulator that is explicitly flowchart-blind, so evaluation scores reflect generalisation;
(2) a five-criterion LLM-judge rubric (task success, information accuracy, consistency,
graceful handling, naturalness) with fine-grained behavioural anchors; (3) in-context
frontier, LangGraph-orchestrated, and same-base-model-orchestrated baselines in the same
harness, isolating the effect of compilation; and (4) bootstrap 95% confidence intervals,
Wilcoxon/Mann-Whitney tests, and Holm-Bonferroni correction across criteria.

We target the three procedural domains from Dennis et al. (travel booking, Zoom customer
support, insurance claims processing). Independent reproduction of the paper's quality
and cost figures is ongoing; current reproduction status is tracked in
[`benchmarks/`](https://github.com/kamaalg/agent2model/tree/main/benchmarks). The
library and eval harness are released under Apache 2.0 at
https://github.com/kamaalg/agent2model.

---

## Honest positioning reminders

- The 128–462× cost reduction and ~98% quality figures are **Dennis et al. 2026's published results**, not independently reproduced numbers from this library. Never present them as ours in the submission.
- The eval harness *design* is a novel contribution; the rubric criteria are modelled on the paper's criteria but agent2model's behavioural anchors are our own.
- State clearly in the paper that reproduction is in progress and link to `benchmarks/` for the live status.

---

## Prep checklist — what's left before July 3

- [ ] **Verify page/format limit** — open https://lifelongagent.github.io/ in a browser and confirm page count (likely 4–6 pp excl. refs) and any template requirements
- [ ] **Expand abstract to a full paper** (intro, method, eval harness design, preliminary/demo results, related work, conclusion)
- [ ] **Run at least one reproduction pass** (travel_booking) and report whatever numbers you have, marked as preliminary — even a single condition is stronger than no numbers
- [ ] **Author list and affiliations** — finalize; ensure all co-authors have current OpenReview profiles (incomplete profiles → desk rejection)
- [ ] **Figures** — the pipeline diagram from the README and a sample eval-report bar chart would each carry significant visual weight; confirm these render in LaTeX
- [ ] **Scope framing** — the abstract currently leads with "agent post-training"; if the "lifelong" framing matters for acceptance, add a sentence connecting stable procedure adherence to deployment longevity
- [ ] **Related work paragraph** — τ²-bench, JourneyBench, DSPy, GEPA; be precise about what distinguishes this harness (flowchart-blind simulator + full stat pipeline on a user-supplied flowchart)
- [ ] **OpenReview submission** — register submission, add author list, upload PDF before July 3 AoE (= end of July 3 in UTC+12, i.e. July 4 12:00 UTC)
- [ ] **Commit final version** on this branch and push before deadline
