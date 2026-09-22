"""callcenter route|batch|shuffle|diagram

callcenter route calls/03_double_charge.json --backend jev
callcenter batch calls --backend circuits
callcenter shuffle calls --backend jev          # does the queue change when the department list is reordered?
callcenter diagram                              # the circuit as Mermaid
"""

from __future__ import annotations

import argparse
import random
import sys

from .agent import triage
from .backends import BACKENDS, pick_backend
from .calls import load_calls
from .circuit import DEPARTMENTS, build_circuit


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="callcenter", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["route", "batch", "shuffle", "diagram"])
    ap.add_argument("path", nargs="?", default="calls")
    ap.add_argument("--backend", default="jev", help=", ".join(BACKENDS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-options", type=int, default=None, help="trim the language list for a backend that caps options (semif, laya: 16)")
    ap.add_argument("--json", action="store_true", help="print the full ticket with its audit record")
    args = ap.parse_args(argv)

    if args.command == "diagram":
        print(build_circuit(args.max_options).to_mermaid())
        return
    backend = pick_backend(args.backend, args.model)
    calls = load_calls(args.path)
    circuit = build_circuit(args.max_options)

    if args.command == "route":
        for call in calls:
            t = triage(call, backend, model=args.model, circuit=circuit)
            if args.json:
                print(t.to_json())
            else:
                _print_ticket(call, t)
        return

    if args.command == "batch":
        print(f"{'call':28s} {'queue':16s} {'prio':5s} {'person':7s} {'deflect':8s} {'redact':7s} {'ms':>6s}   route trace")
        for call in calls:
            t = triage(call, backend, model=args.model, circuit=circuit)
            route = t.audit["gates"]["route"]
            redacted = "yes" if t.audit["gates"]["redact"]["value"] else ""
            trace = "; ".join(route["trace"] or [])[:70]
            print(
                f"{call['file'][:28]:28s} {t.queue:16s} {t.priority:5s} {'yes' if t.assign_to_person else '':7s} "
                f"{'yes' if t.deflected else '':8s} {redacted:7s} {t.audit['latency_ms']:6.0f}   {route['outcome']}: {trace}"
            )
        return

    if args.command == "shuffle":
        _shuffle(calls, backend, args.model)


def _print_ticket(call, t) -> None:
    print(f"\n== {call['file']}  ({call.get('channel', 'phone')}{', recording' if call.get('audio') else ''})")
    if call["transcript"]:
        print("   " + call["transcript"][:160].replace("\n", " ") + ("..." if len(call["transcript"]) > 160 else ""))
    print(f"-> {t.ref}: queue={t.queue} priority={t.priority} person={t.assign_to_person} deflect={t.deflected} repeat={t.repeat_contact} translate={t.needs_translation}")
    a = t.audit["answers"]
    top = max(a["dept"], key=a["dept"].get)
    print(f"   dept {top} {a['dept'][top]:.2f} | angry {a['angry']:.2f} | pii {a['pii']:.2f} | urgency {a['urgency']} | self-service {a['self_service']:.2f} | one topic {a['single_topic']:.2f}")
    for gid in ("route", "human_now", "redact", "deflect"):
        g = t.audit["gates"][gid]
        print(f"   {gid:10s} {g['outcome']:9s} {'; '.join(g['trace'] or [])}")
    print(f"   reply: {t.reply}")
    if t.audit["gates"]["redact"]["value"]:
        print(f"   logged as: {t.transcript_for_log[:120]}")
    print(f"   {t.audit['model']} in {t.audit['latency_ms']:.0f} ms")


def _shuffle(calls, backend, model) -> None:
    """The same call, the department list in four orders. A model that reads the options
    gives the same queue every time; one that reads the position does not."""
    keys = list(DEPARTMENTS)
    orders = [keys, keys[::-1], random.Random(1).sample(keys, len(keys)), random.Random(2).sample(keys, len(keys))]
    flips, n = 0, 0
    print(f"{'call':28s} " + " | ".join(f"order {i + 1}" for i in range(4)))
    for call in calls:
        picks = []
        for order in orders:
            c = build_circuit()
            c.questions["dept"]["criteria"] = {k: DEPARTMENTS[k] for k in order}
            t = triage(call, backend, model=model, circuit=c)
            picks.append(f"{t.audit['gates']['dept_raw']['value']} {t.audit['gates']['dept_raw']['p']:.2f}")
        n += 1
        changed = len({p.split()[0] for p in picks}) > 1
        flips += changed
        print(f"{call['file'][:28]:28s} " + " | ".join(f"{p:20s}" for p in picks) + ("   <- changed" if changed else ""))
    print(f"\n{flips} of {n} calls changed queue when only the option order changed ({getattr(backend, 'model', type(backend).__name__)}).")


if __name__ == "__main__":
    main(sys.argv[1:])
