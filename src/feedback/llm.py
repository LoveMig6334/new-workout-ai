import re
from pathlib import Path
from typing import Optional
import numpy as np

from analysis.types import HoldAnalysis, LiveSnapshot, RepAnalysis
from feedback.prompt_th import (
    SYSTEM_TH,
    SYSTEM_TH_HOLD,
    build_hold_summary_prompt,
    build_live_prompt,
    build_user_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "qwen3_5_4b_mxfp4"

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


class ThaiCoachLLM:
    """Wraps Qwen3.5-4B (vision-language) via mlx-vlm. v1 sends text-only prompts."""

    def __init__(self, model_dir: Path | str | None = None):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model, self._processor = load(str(model_path))
        self._config = load_config(str(model_path))

    def generate(
        self,
        payload,  # RepAnalysis | HoldAnalysis | LiveSnapshot
        max_tokens: int = 160,
        frame_bgr: Optional[np.ndarray] = None,
        exercise=None,  # required for HoldAnalysis / LiveSnapshot
    ) -> str:
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if isinstance(payload, RepAnalysis):
            system = SYSTEM_TH
            user = build_user_prompt(payload)
        elif isinstance(payload, HoldAnalysis):
            if exercise is None:
                raise ValueError("exercise= required for HoldAnalysis")
            system = SYSTEM_TH_HOLD
            user = build_hold_summary_prompt(payload, exercise)
        elif isinstance(payload, LiveSnapshot):
            if exercise is None:
                raise ValueError("exercise= required for LiveSnapshot")
            system = SYSTEM_TH_HOLD
            user = build_live_prompt(payload, exercise)
        else:
            raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = apply_chat_template(
            self._processor,
            self._config,
            messages,
            num_images=0,
            enable_thinking=False,
        )
        result = mlx_generate(
            self._model,
            self._processor,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        text = getattr(result, "text", str(result))
        return _THINK_BLOCK.sub("", text).strip()

    def warmup(self):
        """First call is slow due to compilation. Run once at app start."""
        dummy = RepAnalysis(
            rep_index=-1,
            score=50,
            components={
                "depth": 10,
                "valgus": 10,
                "torso": 10,
                "symmetry": 10,
                "tempo": 10,
            },
            violations=[],
            descent_ms=0,
            ascent_ms=0,
        )
        _ = self.generate(dummy, max_tokens=16)
