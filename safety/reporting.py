from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from .rules import RuleTrigger
from .bias_harm import BiasHarmScores


@dataclass
class SafetyScores:
    toxicity: float
    bias: float
    hallucination: float
    harmfulness: float
    rule_score: float
    overall: float


@dataclass
class SafetyDecision:
    action: str  # "allow", "rewrite", "block"
    reason: str


TOXICITY_WEIGHT = 0.3
HARMFUL_WEIGHT = 0.4
HALLUC_WEIGHT = 0.2
RULE_WEIGHT = 0.1


def _combine_scores(
    toxicity: float,
    bias_harm: BiasHarmScores,
    hallucination: float,
    rule_score: float,
) -> SafetyScores:
    harmfulness = max(bias_harm.hate, bias_harm.self_harm, bias_harm.crime)
    overall = (
        TOXICITY_WEIGHT * toxicity
        + HARMFUL_WEIGHT * harmfulness
        + HALLUC_WEIGHT * hallucination
        + RULE_WEIGHT * rule_score
    )
    return SafetyScores(
        toxicity=toxicity,
        bias=bias_harm.bias,
        hallucination=hallucination,
        harmfulness=harmfulness,
        rule_score=rule_score,
        overall=overall,
    )


def _decide_action(scores: SafetyScores) -> SafetyDecision:
    if scores.overall <= 0.3:
        return SafetyDecision(action="allow", reason="Low overall safety risk.")
    if scores.overall <= 0.6:
        return SafetyDecision(
            action="rewrite",
            reason="Moderate risk; content should be rewritten but topic can be answered.",
        )
    return SafetyDecision(
        action="block",
        reason="High risk; original content should not be shown as-is.",
    )


def _rules_to_text(triggers: List[RuleTrigger]) -> str:
    if not triggers:
        return "No explicit rule violations detected."
    parts = []
    for t in triggers:
        parts.append(f"- {t.name}: {t.explanation} (span: '{t.span}')")
    return "\n".join(parts)


def build_safety_report(
    raw_answer: str,
    user_query: str,
    toxicity: float,
    bias_harm_scores: BiasHarmScores,
    hallucination: float,
    rule_triggers: List[RuleTrigger],
    rule_score: float,
) -> Tuple[Dict, SafetyDecision]:
    scores = _combine_scores(toxicity, bias_harm_scores, hallucination, rule_score)
    decision = _decide_action(scores)

    explanation_parts = []
    explanation_parts.append(
        f"Toxicity score: {scores.toxicity:.2f}. Higher values mean more abusive or toxic language."
    )
    explanation_parts.append(
        f"Bias score: {scores.bias:.2f}. Higher values mean more stereotypes or unfair generalizations."
    )
    explanation_parts.append(
        f"Harmfulness score: {scores.harmfulness:.2f}. Captures hate, self-harm, or crime-related risk."
    )
    explanation_parts.append(
        f"Hallucination risk: {scores.hallucination:.2f}. Higher values mean more unsupported factual claims."
    )
    explanation_parts.append(
        f"Rule-based score: {scores.rule_score:.2f}. Derived from explicit pattern matches (e.g., slurs, self-harm, crime)."
    )
    explanation_parts.append(f"Overall safety score: {scores.overall:.2f}.")
    explanation_parts.append(f"Decision: {decision.action} – {decision.reason}")

    rule_text = _rules_to_text(rule_triggers)
    if rule_text:
        explanation_parts.append("Rule analysis:\n" + rule_text)

    explanation = "\n".join(explanation_parts)

    report: Dict = {
        "scores": asdict(scores),
        "decision": asdict(decision),
        "rule_triggers": [
            {"name": t.name, "severity": t.severity, "span": t.span, "explanation": t.explanation}
            for t in rule_triggers
        ],
        "explanation": explanation,
    }

    return report, decision
