import os
import google.generativeai as genai


def get_gemini_model(model_name: str = "gemini-2.5-flash"):
    """Return a configured GenerativeModel using GEMINI_API_KEY from env."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def generate_text(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """Generate plain text from Gemini given a prompt string."""
    model = get_gemini_model(model_name)
    resp = model.generate_content(prompt)
    return resp.text.strip() if hasattr(resp, "text") else str(resp)