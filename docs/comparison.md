# How agent2model compares

The first question people ask is "isn't this just DSPy / LangGraph / distillation?"
Short answer: **no — they optimize or orchestrate a procedure at runtime; agent2model
removes the runtime machinery by baking the procedure into a small model's weights.**
This page lays out the differences honestly, including where the other tools are the
better choice.

## The one-paragraph version

If your agent procedure is **stable** (a support flow, a booking flow, an onboarding
script), agent2model compiles it — from a LangGraph `StateGraph` or a YAML flowchart —
into the weights of a small open model (Qwen 3B/8B), so at runtime there's **no
orchestrator and no per-turn frontier call**. The win is **cost, latency, and privacy**:
a self-contained model you own and run offline. Prompt-optimizers (DSPy/GEPA) make a
frontier model follow a procedure *better* but keep a runtime program; orchestrators
(LangGraph/CrewAI) run the procedure *live* every turn. agent2model is the only one that
deletes the runtime layer.

## At a glance

| | **agent2model** | **DSPy / GEPA** | **LangGraph / CrewAI** | **Generic distillation** |
|---|---|---|---|---|
| What it optimizes | the model's **weights** | **prompts / programs** | nothing — it **runs** the program | the model's weights |
| Runtime orchestrator | **none** | yes (the compiled program) | yes (every turn) | n/a |
| Per-turn frontier calls | **none** | yes | yes | n/a |
| Input | LangGraph graph or YAML flowchart | a DSPy program + metric | your graph/crew | a teacher + dataset |
| Runs offline / private | **yes** | no (calls a provider) | usually no | yes |
| Target | one **stable, multi-turn procedure** | any task with a metric | any agent app | general capability |
| Ships procedure-adherence eval | **yes** | no (task metrics) | no | no |

## Tool by tool

### vs DSPy / GEPA
DSPy and GEPA *compile* too — but they compile to **prompts and program structure**,
found by optimization/reflection, and the LM + program stay in the loop at runtime. They
shine when you want a frontier model to perform a task better and you're fine paying
per call. agent2model compiles to **weights** and drops the runtime program entirely.
They're complementary: you could use DSPy-style optimization to design a strong
procedure, then agent2model to bake the *stable* result into a cheap local model.

### vs LangGraph / CrewAI
These are **runtime orchestrators** — they execute the procedure live, re-sending it to
the model every turn. That's exactly the cost agent2model removes. **agent2model is
additive, not a replacement**: you can point `compile` straight at an existing LangGraph
`StateGraph` and it extracts the nodes/edges/decisions for you. Keep LangGraph for
procedures that change often or need live tool-use; compile the *stable* ones into
weights. (Tool-use during compiled inference is on the v2 roadmap, not v1.)

### vs generic distillation / fine-tuning
Classic distillation clones broad capability from a teacher. agent2model is narrower and
that's the point: it internalizes **one declared, multi-turn conversational procedure**,
generating synthetic conversations by traversing the flowchart so the model learns the
*behavior* — and the flowchart never appears in the training data. It deliberately uses
**full-parameter SFT, not LoRA** (the source paper's companion shows LoRA fails to
internalize procedures at any rank), and ships an eval harness to **prove** the procedure
stuck.

## When NOT to use agent2model

Be honest with yourself — reach for something else if:

- Your procedure **changes frequently** — you'd recompile constantly; keep it in
  LangGraph.
- It needs **live tool-use / external API calls mid-conversation** — that's v2, not v1.
- It's **multi-agent with handoffs** — out of scope for v1.
- You have **no stable procedure yet** — design it first (DSPy can help), then compile.

## The honesty caveat

The headline **128–462× cheaper / near-frontier quality** figures are **the paper's**
([Dennis et al. 2026](https://arxiv.org/abs/2605.22502)) and have **not yet been
reproduced in this repository** — reproduction is in progress; status lives in
[`benchmarks/`](https://github.com/kamaalg/agent2model/tree/main/benchmarks). And note a
nuance: a same-author follow-up shows in-context prompting alone nearly matches frontier
*quality*, so quality parity isn't the pitch — **cost, latency, and privacy** are. What
this repo ships today is the open pipeline plus a procedure-adherence eval harness; the
numbers will be filled in as the reproduction lands.
