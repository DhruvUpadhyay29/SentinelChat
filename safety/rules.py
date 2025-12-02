import re
from dataclasses import dataclass
from typing import List


PROTECTED_ATTRIBUTES = [
    "race",
    "religion",
    "gender",
    "sexual orientation",
    "disability",
    "nationality",
    "ethnicity",
]

EXTREME_SWEAR_WORDS = [
    "fuck",
    "shit",
    "bitch",
]

CRIME_KEYWORDS = [
    "how to make a bomb",
    "how to hack",
    "bypass password",
    "buy illegal",
]

SELF_HARM_KEYWORDS = [
    "kill myself",
    "suicide",
    "end my life",
]


@dataclass
class RuleTrigger:
    name: str
    severity: float  # 0-1
    span: str
    explanation: str


def _find_any(text: str, patterns: List[str]) -> List[str]:
    matches = []
    lower = text.lower()
    for p in patterns:
        if p in lower:
            matches.append(p)
    return matches


def check_rules(text: str) -> List[RuleTrigger]:
    triggers: List[RuleTrigger] = []
    lower = text.lower()

    for sw in _find_any(lower, EXTREME_SWEAR_WORDS):
        triggers.append(
            RuleTrigger(
                name="extreme_profanity",
                severity=0.6,
                span=sw,
                explanation=f"Contains strong profanity: '{sw}'.",
            )
        )

    for attr in PROTECTED_ATTRIBUTES:
        if re.search(rf"\b(hate|dislike|inferior|superior)\b.*\b{re.escape(attr)}\b", lower):
            triggers.append(
                RuleTrigger(
                    name="targeted_hate_protected_attribute",
                    severity=0.9,
                    span=attr,
                    explanation=f"Potential hate targeting protected attribute '{attr}'.",
                )
            )

    for key in _find_any(lower, CRIME_KEYWORDS):
        triggers.append(
            RuleTrigger(
                name="instructions_for_crime",
                severity=1.0,
                span=key,
                explanation=f"Appears to request or offer illegal guidance: '{key}'.",
            )
        )

    for key in _find_any(lower, SELF_HARM_KEYWORDS):
        triggers.append(
            RuleTrigger(
                name="self_harm",
                severity=1.0,
                span=key,
                explanation=f"Indicates self-harm intent or ideation: '{key}'.",
            )
        )

    return triggers


def aggregate_rule_score(triggers: List[RuleTrigger]) -> float:
    if not triggers:
        return 0.0
    return max(t.severity for t in triggers)
