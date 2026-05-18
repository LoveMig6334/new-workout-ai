import os
from pathlib import Path
import numpy as np
import pytest

QWEN_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3_5_4b_mxfp4"


@pytest.mark.skipif(not QWEN_DIR.exists(), reason="Qwen model not downloaded")
def test_llm_generates_thai_text():
    from workout_ai.feedback.llm import ThaiCoachLLM
    from workout_ai.analysis.types import RepAnalysis

    llm = ThaiCoachLLM()
    rep = RepAnalysis(
        rep_index=0,
        score=78,
        components={"depth": 30, "valgus": 18, "torso": 20, "symmetry": 12, "tempo": 8},
        violations=[],
        descent_ms=1200,
        ascent_ms=900,
    )
    text = llm.generate(rep, max_tokens=120)
    assert isinstance(text, str)
    assert len(text) > 5
    assert any("฀" <= c <= "๿" for c in text)
