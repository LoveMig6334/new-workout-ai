from pathlib import Path
from typing import Optional
import numpy as np

from workout_ai.feedback.prompt_th import SYSTEM_TH, build_user_prompt
from workout_ai.analysis.types import RepAnalysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "qwen3_5_4b_mxfp4"


class ThaiCoachLLM:
    """Wraps Qwen3.5-4B (vision-language) via mlx-vlm. v1 sends text-only prompts."""

    def __init__(self, model_dir: Path | str | None = None):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model_path = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model, self._processor = load(str(model_path))
        self._config = load_config(str(model_path))

    def generate(self, rep: RepAnalysis, max_tokens: int = 160, frame_bgr: Optional[np.ndarray] = None) -> str:
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        user = build_user_prompt(rep)
        messages = [
            {"role": "system", "content": SYSTEM_TH},
            {"role": "user", "content": user},
        ]
        prompt = apply_chat_template(self._processor, self._config, messages, num_images=0)
        result = mlx_generate(
            self._model,
            self._processor,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return getattr(result, "text", str(result)).strip()

    def warmup(self):
        """First call is slow due to compilation. Run once at app start."""
        dummy = RepAnalysis(
            rep_index=-1,
            score=50,
            components={"depth": 10, "valgus": 10, "torso": 10, "symmetry": 10, "tempo": 10},
            violations=[],
            descent_ms=0,
            ascent_ms=0,
        )
        _ = self.generate(dummy, max_tokens=16)
