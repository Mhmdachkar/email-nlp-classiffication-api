#!/usr/bin/env python3
"""
Wrapper around a BART zero-shot classification model for email categorization.

This component provides an easy-to-use interface to HuggingFace's transformers
pipeline for zero-shot classification with `facebook/bart-large-mnli` (configurable).
"""

from typing import Dict, List, Optional, Tuple

try:
    from transformers import pipeline  # type: ignore
except Exception as import_error:  # pragma: no cover - handled at runtime
    pipeline = None  # defer error until usage to avoid import-time crashes


class BartZeroShotClassifier:
    """Zero-shot classifier using a BART NLI model.

    Parameters
    ----------
    model_name: str
        HuggingFace model identifier (default: "facebook/bart-large-mnli").
    candidate_labels: List[str]
        Labels to score; can be overridden per-call.
    multi_label: bool
        Whether to allow multiple labels per text. For mutually-exclusive
        categories, keep False.
    device: Optional[int]
        Device index for inference; use -1 for CPU (default). If CUDA is
        available, you can pass 0 for GPU.
    hypothesis_template: str
        Template used by the NLI hypothesis (default: "This text is about {}.").
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        candidate_labels: Optional[List[str]] = None,
        multi_label: bool = False,
        device: int = -1,
        hypothesis_template: str = "This text is about {}.",
    ) -> None:
        if pipeline is None:
            raise ImportError(
                "transformers is not installed. Please add it to requirements and install."
            )

        self.model_name = model_name
        self.multi_label = multi_label
        self.hypothesis_template = hypothesis_template
        self.candidate_labels = candidate_labels or []
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device,
        )

    def classify(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
        multi_label: Optional[bool] = None,
        hypothesis_template: Optional[str] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """Classify a single text.

        Returns
        -------
        predicted_label: str
            The top label by score.
        confidence: float
            Score associated with the top label.
        scores_by_label: Dict[str, float]
            Mapping of label to score for provided labels.
        """
        labels = candidate_labels or self.candidate_labels
        if not labels:
            raise ValueError("candidate_labels must be provided at init or call time")

        result = self._pipeline(
            text,
            labels,
            multi_label=self.multi_label if multi_label is None else multi_label,
            hypothesis_template=self.hypothesis_template
            if hypothesis_template is None
            else hypothesis_template,
        )

        # transformers returns labels sorted by score descending
        labels_out: List[str] = result["labels"]
        scores_out: List[float] = result["scores"]
        scores_by_label = {label: float(score) for label, score in zip(labels_out, scores_out)}
        predicted_label = labels_out[0]
        confidence = float(scores_out[0])
        return predicted_label, confidence, scores_by_label

    def classify_batch(
        self,
        texts: List[str],
        candidate_labels: Optional[List[str]] = None,
        multi_label: Optional[bool] = None,
        hypothesis_template: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Classify a batch of texts. Returns a list of (label, conf, scores)."""
        return [
            self.classify(
                text,
                candidate_labels=candidate_labels,
                multi_label=multi_label,
                hypothesis_template=hypothesis_template,
            )
            for text in texts
        ]


