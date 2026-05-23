from pathlib import Path
import pytest

QWEN_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "qwen3_5_4b_mxfp4"


@pytest.mark.skipif(not QWEN_DIR.exists(), reason="Qwen model not downloaded")
def test_llm_generates_thai_text():
    from feedback.llm import ThaiCoachLLM
    from analysis.types import RepAnalysis

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


# Smoke tests for hold/live payloads. Requires the Qwen weights.


@pytest.mark.skipif(not QWEN_DIR.exists(), reason="Qwen weights not downloaded")
def test_generate_accepts_hold_analysis():
    from analysis.types import HoldAnalysis
    from exercises.neck_stretch import NeckStretchLeft
    from feedback.llm import ThaiCoachLLM

    llm = ThaiCoachLLM()
    ex = NeckStretchLeft()
    a = HoldAnalysis(
        exercise_name=ex.name,
        score=88,
        components={"duration": 50, "precision": 25, "stability": 13},
        violations=[],
        in_target_ms=20_000,
        drift_count=1,
    )
    text = llm.generate(a, max_tokens=32, exercise=ex)
    assert text
