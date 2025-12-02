from typing import Optional

from detoxify import Detoxify

_model_singleton: Optional[Detoxify] = None


def _get_model() -> Detoxify:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = Detoxify("original")
    return _model_singleton


def score_toxicity(text: str) -> float:
    """Return Detoxify toxicity score in [0,1]."""
    if not text.strip():
        return 0.0
    model = _get_model()
    preds = model.predict(text)
    return float(preds.get("toxicity", 0.0))
