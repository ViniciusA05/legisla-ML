"""Framework-free inference path: encode -> forward -> softmax -> decision (FEAT-003)."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.legislabert import head_tail_encode


@dataclass(frozen=True)
class Prediction:
    label: int
    score_positive: float
    scores: tuple[float, float]
    threshold: float
    truncated: bool
    n_tokens: int


class Predictor:
    """Run the fine-tuned classifier on raw statute text using the shared Head-Tail path."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        threshold: float = 0.5,
        max_batch: int = 16,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_batch = max_batch

    def predict_one(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for start in range(0, len(texts), self.max_batch):
            predictions.extend(self._predict_chunk(texts[start : start + self.max_batch]))
        return predictions

    def warmup(self) -> None:
        """Run one forward pass so the first real request does not pay initialisation cost."""
        self.predict_one("warmup")

    def _predict_chunk(self, texts: list[str]) -> list[Prediction]:
        encoded = [head_tail_encode(self.tokenizer, text) for text in texts]
        input_ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
        attention_mask = torch.tensor([e[1] for e in encoded], dtype=torch.long)

        with torch.inference_mode():
            logits, _ = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits.float(), dim=-1).tolist()

        results = []
        for (_, _, n_tokens, truncated), (p0, p1) in zip(encoded, probs, strict=True):
            results.append(
                Prediction(
                    label=int(p1 >= self.threshold),
                    score_positive=p1,
                    scores=(p0, p1),
                    threshold=self.threshold,
                    truncated=truncated,
                    n_tokens=n_tokens,
                )
            )
        return results
