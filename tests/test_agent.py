"""The circuit and the agent, exercised with the hand-written answers in calls/, no network."""

from pathlib import Path

import pytest

from callcenter import triage
from callcenter.agent import redact
from callcenter.backends import FakeBackend
from callcenter.calls import load_calls
from callcenter.circuit import build_circuit

CALLS = {c["id"]: c for c in load_calls(Path(__file__).resolve().parents[1] / "calls")}


def run(call_id):
    return triage(CALLS[call_id], FakeBackend())


def test_self_service_is_deflected_with_an_article():
    t = run("01_password_reset")
    assert t.deflected and t.queue == "account_access" and "reset-password" in t.reply


def test_outage_goes_to_a_person_at_p1():
    t = run("02_outage")
    assert t.queue == "technical" and t.priority == "P1" and t.assign_to_person and not t.deflected


def test_angry_repeat_billing_contact():
    t = run("03_double_charge")
    assert t.queue == "billing" and t.assign_to_person and t.repeat_contact and not t.deflected


def test_card_number_is_redacted_before_logging_and_not_deflected_blindly():
    t = run("04_card_number")
    assert "4532" not in t.transcript_for_log and "[redacted]" in t.transcript_for_log
    assert "Elm Street" not in t.transcript_for_log


def test_two_unrelated_issues_go_to_triage_not_a_guess():
    t = run("08_two_issues")
    assert t.queue == "triage" and t.audit["gates"]["route"]["outcome"] == "escalate"


def test_wrong_number_is_other_and_low_priority():
    t = run("09_wrong_number")
    assert t.queue == "other" and t.priority == "P4"


def test_spanish_is_flagged_for_translation():
    assert run("11_spanish").needs_translation


def test_repeat_contact_does_not_sit_at_p4():
    t = run("07_rude_agent")
    assert t.repeat_contact and t.priority != "P4"


def test_every_ticket_carries_a_full_audit():
    for cid in CALLS:
        t = run(cid)
        assert set(t.audit["answers"]) == {"dept", "urgency", "angry", "pii", "repeat", "self_service", "single_topic", "english"}
        assert all("trace" in g for g in t.audit["gates"].values())


def test_redaction_patterns():
    assert redact("card 4532 0151 1283 0366 please") == "card [redacted] please"
    assert redact("mail me at a.b@example.com") == "mail me at [redacted]"
    assert redact("order 149 dollars") == "order 149 dollars"


def test_circuit_compiles_and_draws():
    c = build_circuit()
    assert set(c.compile()) >= {"route", "priority", "human_now", "redact", "deflect"}
    assert "route" in c.to_mermaid()


@pytest.mark.parametrize("name", ["jev", "circuits", "local", "semif", "openai", "anthropic"])
def test_backend_names_are_known(name):
    from callcenter.backends import BACKENDS

    assert name in BACKENDS


def test_priority_near_a_cutpoint_rounds_toward_faster():
    call = dict(CALLS["05_cancel"])
    call["expected_answers"] = dict(call["expected_answers"], urgency={"0": 0.55, "1": 0.45, "2": 0.0, "3": 0.0})  # score 0.45, cutpoint 0.5
    t = triage(call, FakeBackend())
    assert t.priority == "P3" and "rounded up" in " ".join(t.audit["gates"]["priority"]["trace"])


def test_a_recording_is_routed_from_the_audio_and_never_transcribed_into_the_log():
    audio_calls = {c["id"]: c for c in load_calls(Path(__file__).resolve().parents[1] / "calls" / "audio")}
    call = audio_calls["04_card_number"]
    assert call["audio"].endswith("04_card_number.wav") and call["transcript"]  # the twin .json supplies the text for display only
    answers = CALLS["04_card_number"]["expected_answers"]
    t = triage(call, FakeBackend(answers))
    assert t.queue == "billing" and t.audit["model"] == "circuit-audio-7b"
    assert "4532" not in t.transcript_for_log and "withheld" in t.transcript_for_log
    calm = triage(audio_calls["05_cancel"], FakeBackend(CALLS["05_cancel"]["expected_answers"]))
    assert calm.transcript_for_log.startswith("recording: ")
