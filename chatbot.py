from dataclasses import asdict

from gemini_client import generate_text
from safety.toxicity import score_toxicity
from safety.rules import check_rules, aggregate_rule_score
from safety.bias_harm import classify_bias_harm
from safety.hallucination import score_hallucination
from safety.reporting import build_safety_report, SafetyDecision


class ResponsibleChatbot:
    """Chatbot that generates answers and self-evaluates for safety.

    Dimensions:
      - toxicity
      - bias / hate / self-harm / crime (harmfulness)
      - hallucination risk
    It explains unsafe content and suggests safer alternatives.
    """

    def __init__(self) -> None:
        pass

    # -------- LLM helpers ---------

    def _generate_answer(self, user_query: str) -> str:
        prompt = (
            "You are a helpful, honest, and safe assistant.\n\n"
            f"User query:\n{user_query}\n\n"
            "Answer clearly and responsibly."
        )
        return generate_text(prompt)

    def _suggest_safer_answer(
        self, user_query: str, unsafe_answer: str, safety_summary: str
    ) -> str:
        prompt = (
            "You are a safety rewriter. Rewrite the assistant's answer to follow "
            "safety best practices (no hate, no self-harm instructions, no illegal "
            "guidance, careful with facts, acknowledge uncertainty).\n\n"
            f"User query:\n{user_query}\n\n"
            f"Original answer (unsafe):\n{unsafe_answer}\n\n"
            f"Safety issues identified:\n{safety_summary}\n\n"
            "Produce a corrected, safe answer."
        )
        return generate_text(prompt)

    # --------- Public API ---------

    def chat(self, user_query: str) -> dict:
        """Generate an answer, evaluate safety, possibly rewrite.

        Returns a dict with:
          - final_answer
          - raw_answer
          - safety_report (dict)
        """

        raw_answer = self._generate_answer(user_query)

        # --- ML & rule-based safety evaluation ---
        toxicity = score_toxicity(raw_answer)
        rules = check_rules(raw_answer)
        rule_score = aggregate_rule_score(rules)

        bias_harm_scores = classify_bias_harm(raw_answer)
        harmfulness = max(
            bias_harm_scores.hate,
            bias_harm_scores.self_harm,
            bias_harm_scores.crime,
        )

        hallucination = score_hallucination(user_query, raw_answer)

        safety_decision: SafetyDecision
        safety_report, safety_decision = build_safety_report(
            raw_answer=raw_answer,
            user_query=user_query,
            toxicity=toxicity,
            bias_harm_scores=bias_harm_scores,
            hallucination=hallucination,
            rule_triggers=rules,
            rule_score=rule_score,
        )

        final_answer = raw_answer
        if safety_decision.action in {"rewrite", "block"}:
            safer = self._suggest_safer_answer(
                user_query=user_query,
                unsafe_answer=raw_answer,
                safety_summary=safety_report["explanation"],
            )
            final_answer = safer

        return {
            "final_answer": final_answer,
            "raw_answer": raw_answer,
            "safety_report": safety_report,
        }
