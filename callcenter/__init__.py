"""A support-line triage agent built on decision circuits.

A transcript comes in. One request asks the model eight typed questions about it. A
circuit, plain code, turns the probabilities into decisions: which queue, what priority,
whether a person takes it now, whether to redact before logging, whether the route can be
trusted. Then the agent acts and writes down why.
"""

from .agent import Ticket, triage
from .backends import BACKENDS, pick_backend
from .circuit import DEPARTMENTS, build_circuit

__all__ = ["BACKENDS", "DEPARTMENTS", "Ticket", "build_circuit", "pick_backend", "triage"]
