from gemini_client import generate_text


PROMPT = """
You are a factuality assessor. For the ASSISTANT ANSWER below, estimate the risk that it
contains unsupported or hallucinated factual claims.

Return only a float from 0.0 (no hallucination risk) to 1.0 (very high risk).
"""
def score_hallucination(user_query: str, answer: str) -> float:
    msg = (
        PROMPT
        + f"\n\nUser query:\n{user_query}\n\nAssistant answer:\n{answer}\n\n"
        + "Respond with a single number only."
    )
    text = generate_text(msg)
    try:
        return float(text)
    except ValueError:
        return 0.5
