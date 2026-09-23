"""The questions and the gates. This file is the whole policy.

Nothing here generates text. The model answers typed questions with probabilities; every
decision below is code over those numbers, so it can be read, tested with hand-written
answers, and audited per call.
"""

from __future__ import annotations

from decision_circuits import Circuit, Q, argmax, at_least, order, verify

DEPARTMENTS = {
    "billing": "Charges, refunds, invoices, payment methods, subscription changes",
    "technical": "Something does not work: errors, outages, bugs, devices, connectivity",
    "account_access": "Cannot log in, password or two-factor problems, locked or compromised accounts",
    "cancellation": "Wants to cancel, downgrade, or is asking what happens if they leave",
    "sales": "Wants to buy, upgrade, ask about plans or pricing before purchase",
    "complaint": "Unhappy with how a previous contact or a person was handled",
    "other": "None of the above, or not about our service at all",
}

# The twenty most spoken languages in US homes after English (American Community Survey),
# because a support line takes the calls the country makes. Twenty-two options is nothing for
# Jev or the circuit family and more than SemIf or Laya can be asked at all; see the README.
LANGUAGES = dict.fromkeys(
    [
        "English",
        "Spanish",
        "Chinese",
        "Tagalog",
        "Vietnamese",
        "Arabic",
        "French",
        "Korean",
        "Russian",
        "Haitian Creole",
        "German",
        "Hindi",
        "Portuguese",
        "Italian",
        "Polish",
        "Urdu",
        "Japanese",
        "Persian",
        "Gujarati",
        "Telugu",
        "Bengali",
    ]
) | {"other": "Any other language"}


# When the route cannot be trusted, the one question that would settle it. The keys are what
# the model picks; the agent turns the pick into the sentence it asks (agent.ASK_CALLER).
ASK_NEXT = {
    "account": "Ask for the email address or account number, to find the account",
    "invoice": "Ask for the order or invoice number the caller means",
    "device": "Ask which device and app version the problem is on",
    "which_first": "Ask which of the caller's issues to handle first",
    "nothing": "Nothing: what the caller said is enough to act on",
}

URGENCY = [
    "Can wait a few days: a question, a preference, a minor inconvenience",
    "Should be handled today: something is wrong but the caller can work around it",
    "Needs attention within the hour: the caller is blocked from something they need",
    "Service down, money moving wrongly, or safety at stake: right now",
]


def build_circuit(max_options: int | None = None, v2: bool = False) -> Circuit:
    """`max_options` trims the language list for a backend that caps a question's options
    (SemIf and Laya stop at 16, and refuse the whole request, not just that question).
    `v2` adds the questions only circuit v2 models answer, on a transcript (not a recording):
    every topic the caller raises, and the sentence that shows how urgent it is."""
    c = Circuit()
    languages = LANGUAGES if max_options is None else dict(list(LANGUAGES.items())[: max_options - 1]) | {"other": "Any other language"}

    # ---- what the model is asked, all in one request -------------------------------
    c.choice("dept", "Which team should handle this contact? Pick the single best fit.", DEPARTMENTS)
    c.score("urgency", "How urgent is this for the caller?", URGENCY)
    c.noul("angry", "Is the caller angry, or threatening to leave or escalate?", true="Angry or threatening", false="Calm, even if unhappy")
    c.noul(
        "pii",
        "Does the transcript contain personal data that must not go into a shared log: card or account numbers, government IDs, home addresses, health details?",
        true="Contains such data",
        false="Nothing beyond a name and the issue",
    )
    c.noul(
        "repeat",
        "Does the caller say they have already contacted us about this same issue before?",
        true="A repeat contact",
        false="First contact, or not said",
    )
    c.noul(
        "self_service",
        "Could the caller resolve this themselves by following a standard help article (reset a password, update a card, restart a device)?",
        true="A documented self-service task",
        false="Needs a person or an account change only staff can make",
    )
    c.noul("single_topic", "Is the caller raising one issue, rather than several unrelated ones?", true="One issue", false="Several unrelated issues")
    # Asked as a choice, not "is it English?": the audio model answers that yes/no with yes for
    # every clip it hears, and names the language correctly when it is offered the options.
    c.choice("language", "Which language is the caller speaking?", languages)
    # An ordinary choice every backend answers: what to ask when the route is not trusted.
    c.choice("ask_next", "If we cannot route this call yet, what single question should we ask the caller first?", ASK_NEXT)
    if v2:
        c.multi("topics", "Which of these does the caller raise? Mark every one that applies.", DEPARTMENTS)
        c.locate("urgency_evidence", "Which sentence shows best how urgent this is for the caller?", none="no sentence speaks to urgency")

    # ---- what the code decides ------------------------------------------------------
    # The queue. The pick is trusted only when the model is confident and the checker
    # agrees the contact is about one thing; otherwise a person triages it.
    c.gate("route", verify("dept", check=Q("single_topic"), tau=0.6, min_confidence=0.35), on_uncertain="escalate")
    # P1 to P4 from the expected urgency; near a cutpoint the bucket is marked uncertain.
    c.gate("priority", order("urgency", [0.5, 1.5, 2.5]))
    # A person takes it now if the caller is angry or the situation is a level-3 emergency.
    c.gate("human_now", (Q("angry") | Q("urgency")[3]) >= 0.7, band=0.1, on_uncertain="escalate")
    # When unsure about personal data, redact anyway: the cost of a wrong yes is a hidden digit.
    c.gate("redact", Q("pii") >= 0.5, band=0.15, on_uncertain="default", default=True)
    # Send a help article instead of opening a ticket, but never to someone who is angry.
    c.gate("deflect", (Q("self_service") & ~Q("angry")) >= 0.7, band=0.1, on_uncertain="default", default=False)
    # A repeat contact gets the previous ticket attached and skips the queue's back.
    c.gate("repeat_contact", Q("repeat") >= 0.7, band=0.1, on_uncertain="default", default=False)
    # Not English: a translated queue, whatever the department.
    c.gate("needs_translation", ~Q("language")["English"] >= 0.5, band=0.1, on_uncertain="default", default=False)
    # The question to put to the caller when the route escalates; trusted only when the pick is clear.
    c.gate("ask", argmax("ask_next", min_confidence=0.2))
    if v2:
        # Two or more topics: the ticket lists them all, so triage splits it instead of rereading it.
        c.gate("several_topics", at_least(2, "topics"), band=0.1, on_uncertain="default", default=False)
    # The department argmax on its own, for the shuffle demo: what the model said before verification.
    c.gate("dept_raw", argmax("dept"))
    return c
