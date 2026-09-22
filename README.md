# call-center-circuit

A support-line triage agent built on [decision circuits](https://github.com/Barneyjm/decision-circuits).
A transcript comes in; one request asks a decision model eight typed questions about it;
a circuit, plain code, turns the probabilities into decisions; the agent acts on them and
writes down every number it relied on.

```
$ callcenter route calls/03_double_charge.json --backend jev


== 03_double_charge.json  (phone)
   You charged my card twice for the same order, 149 dollars each time, and the second charge overdrew my account. This is the third time I have called about it. I...
-> CC-1001: queue=billing priority=P1 person=True deflect=False repeat=True translate=False
   dept billing 1.00 | angry 0.97 | pii 0.12 | urgency {'0': 0.0, '1': 0.02, '2': 0.02, '3': 0.96} | self-service 0.04 | one topic 0.95
   route      decided   dept -> billing conf=1.00; check single_topic p=0.95 (tau 0.6)
   human_now  decided   angry p=0.97; urgency[3] p=0.96; or under independence -> p=1.00
   redact     decided   pii p=0.12
   deflect    decided   self_service p=0.04; gate _deflect_1 p=0.03; and under independence -> p=0.00
   reply: Thanks, I have passed this straight to a member of the billing team as P1, and they will contact you within the hour. Your reference is CC-1001.
   jev-1.13.0 in 538 ms
```

No prose is generated anywhere. The model answers questions with probabilities; the
decisions, the reply, and the audit are code.

## Run it

```bash
git clone https://github.com/Barneyjm/call-center-circuit && cd call-center-circuit
uv sync
cp .env.example .env            # put one key in it
uv run callcenter batch calls --backend jev
```

Four commands:

| | |
|---|---|
| `callcenter route <call.json>` | one call, the ticket, the reasons |
| `callcenter batch calls/` | every sample call as a table |
| `callcenter shuffle calls/` | the same calls with the department list reordered four ways: does the queue change? |
| `callcenter diagram` | the circuit as Mermaid |

`--json` prints the full ticket with its audit record: every probability, every gate's
value, outcome and trace, the model version and request id.

## The model is one flag

The agent does not care which model answers, as long as it speaks the System One
contract (typed questions in, probabilities out, one pass). `--backend` picks it:

| `--backend` | model | key |
|---|---|---|
| `jev` (default) | TypeSafe's hosted Jev | `TYPESAFE_API_KEY` |
| `circuits` | the open-weights circuit family, hosted at decisioncircuits.com | `DECISIONCIRCUITS_API_KEY`, or a free key is issued on first run |
| `local` | the same open weights on your own machine | none |
| `semif` | SemIf on the LangSmith gateway | `LANGSMITH_API_KEY` |
| `openai`, `anthropic` | a chat model, read through logprobs or tool use | the vendor's key |
| `fake` | the hand-written answers in `calls/*.json` | none |

### Swapping in the open weights

Hosted, no download:

```bash
uv run callcenter batch calls --backend circuits                    # circuit-1.7b
uv run callcenter batch calls --backend circuits --model circuit-8b
```

On your own machine (an Apple-silicon Mac or any GPU with 8 GB):

```bash
git clone https://github.com/Barneyjm/circuit && cd circuit && uv sync
uv run hf download jbarney/circuit-1.7b --local-dir runs/circuit-1.7b
S1_MODEL=lora:runs/circuit-1.7b uv run python -m s1proto        # serves :8901
cd ../call-center-circuit && uv run callcenter batch calls --backend local
```

Weights: [`jbarney/circuit-1.7b`](https://huggingface.co/jbarney/circuit-1.7b),
[`jbarney/circuit-8b`](https://huggingface.co/jbarney/circuit-8b), Apache 2.0.

## What the circuit decides

`callcenter/circuit.py` is the whole policy, about sixty lines. Eight questions go out in
one request:

| question | type | used for |
|---|---|---|
| `dept` | choice of 7 | the queue |
| `urgency` | score, 4 levels | P1 to P4 |
| `angry` | yes/no | a person takes it now; never deflect an angry caller |
| `pii` | yes/no | redact before logging |
| `repeat` | yes/no | attach the previous ticket, skip the back of the queue |
| `self_service` | yes/no | send a help article instead of opening a ticket |
| `single_topic` | yes/no | the checker: a route is only trusted when the contact is about one thing |
| `english` | yes/no | the translated queue |

And the gates:

```python
c.gate("route", verify("dept", check=Q("single_topic"), tau=0.6, min_confidence=0.35), on_uncertain="escalate")
c.gate("priority", order("urgency", [0.5, 1.5, 2.5]))
c.gate("human_now", (Q("angry") | Q("urgency")[3]) >= 0.7, band=0.1, on_uncertain="escalate")
c.gate("redact", Q("pii") >= 0.5, band=0.15, on_uncertain="default", default=True)
c.gate("deflect", (Q("self_service") & ~Q("angry")) >= 0.7, band=0.1, on_uncertain="default", default=False)
```

Three things a classifier at a fixed 0.5 does not give you:

- **A band.** `human_now` at 0.7 with a band of 0.1 means 0.6 to 0.8 is "not sure", and
  not-sure has its own action (here: a person looks). A threshold alone silently rounds.
- **A checker.** `verify` trusts the department pick only when a second question agrees
  the contact is about one issue. Call 08 raises two unrelated problems; the pick is a coin
  toss between billing and technical, and the circuit sends it to triage instead of guessing.
- **Asymmetric defaults.** When the model is unsure whether a transcript holds personal
  data, `redact` defaults to yes. A wrongly hidden digit costs nothing; a leaked card
  number does.

Every ticket carries the answers and traces, so a decision can be replayed months later
with `c.evaluate(audit["answers"])` and no model at all.

## The shuffle demo

```
$ callcenter shuffle calls --backend jev
```

asks each call four times with the seven departments listed in different orders. A
causal decoder reads options in sequence, so its answer can depend on the order they were
typed in. On these twelve calls neither Jev nor `circuit-1.7b` changes a queue: they are
clear-cut, and the demo shows it in the numbers instead. Our model returns the same
probabilities to the second decimal in every order; Jev's move (.68 to .83 on the
two-issue call). Measured on 981 harder questions, 8% of Jev's answers change with the
order, 23% of SemIf's, 30% of Laya's, 14% of our own v1.1. `circuit-1.7b` v1.2 encodes
the options side by side and does not depend on the order at all: 0.5%, the bf16 floor.
[The write-up and the test.](https://github.com/Barneyjm/circuit/blob/main/docs/cold-eval.md)

## What each model did with these calls

Same twelve calls, same circuit:

| | Jev | circuit-1.7b v1.2 (hosted) |
|---|---|---|
| queue right | 12 of 12 | 10 of 12: the wrong-number call went to `cancellation`, the cancellation to `billing` |
| two-issue call sent to triage | yes | yes |
| angry double-charge to a person | yes | no: it read the caller as calm |
| card number redacted | yes | yes |
| Spanish flagged | yes | yes |
| latency per call, 8 questions | 290 to 590 ms | 1.3 to 1.6 s |

The 1.7B is the smaller, free, open model and it shows on the two judgment calls. Its
latency here is high because the side-by-side layout cannot share the state across the
eight questions the way the ordinary layout does; a one-question request is 190 ms.

## Inside an agent framework

This repo keeps the agent loop as plain Python so it can be read top to bottom. The
same circuit drops into the frameworks through the adapters in decision-circuits:
[`CircuitToolGuard`](https://github.com/Barneyjm/decision-circuits/blob/main/docs/integrations.md)
for LangChain, `circuit_input_guardrail` for the OpenAI Agents SDK, `circuit_pre_tool_use`
for the Claude Agent SDK. `build_circuit()` is the object they all take.

## Tests

`uv run pytest` runs the circuit against the hand-written answers in each call file. No
network, no model, and every decision above is asserted.

## License

MIT.
