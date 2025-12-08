import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR


@dataclass
class OCRLine:
    y_center: float
    texts: List[str]


@dataclass
class OCRResult:
    raw_text: str
    lines: List[OCRLine]
    boxes: np.ndarray
    texts: Tuple[str, ...]
    scores: Tuple[float, ...]


class OCREngine:
    def __init__(self, enable_preprocess: bool = True) -> None:
        self._engine = RapidOCR()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enable_preprocess = enable_preprocess

    def run(self, image_path: Path) -> Optional[OCRResult]:
        """Run OCR and return structured text preserving reading order."""
        target_path = image_path
        temp_file: Optional[tempfile.NamedTemporaryFile] = None
        if self.enable_preprocess:
            target_path, temp_file = self._preprocess(image_path)

        try:
            result, _ = self._engine(str(target_path))
        except Exception:
            self.logger.exception("RapidOCR failed on %s", image_path)
            return None

        boxes: Optional[np.ndarray] = None
        texts: Optional[Tuple[str, ...]] = None
        scores: Optional[Tuple[float, ...]] = None

        if result is None:
            self.logger.warning("OCR returned empty result for %s", image_path)
            return None

        if hasattr(result, "boxes") and hasattr(result, "txts"):
            boxes = result.boxes
            texts = result.txts
            scores = getattr(result, "scores", None)
        elif isinstance(result, list) and result:
            try:
                boxes = np.array([item[0] for item in result], dtype=float)
                texts = tuple(item[1] for item in result)
                scores = tuple(item[2] for item in result)
            except Exception:
                self.logger.exception("Failed to parse list-based RapidOCR output for %s", image_path)
                return None

        if boxes is None or texts is None:
            self.logger.warning("OCR returned unsupported result for %s", image_path)
            return None

        lines = self._cluster_lines(boxes, texts)
        ordered_text = "\n".join([" ".join(line.texts) for line in lines])

        ocr_out = OCRResult(
            raw_text=ordered_text,
            lines=lines,
            boxes=boxes,
            texts=texts,
            scores=scores or tuple(),
        )

        if temp_file:
            temp_path = Path(temp_file.name)
            temp_file.close()
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                self.logger.debug("Temp file cleanup failed for %s", temp_path)

        return ocr_out

    def _cluster_lines(self, boxes: np.ndarray, texts: Tuple[str, ...]) -> List[OCRLine]:
        # Compute a representative y for each box and cluster by proximity to restore reading order.
        entries = []
        heights = []
        for box, text in zip(boxes, texts):
            y_values = box[:, 1]
            y_center = float(np.mean(y_values))
            height = float(np.max(y_values) - np.min(y_values))
            entries.append((y_center, text))
            heights.append(height)

        if not entries:
            return []

        median_height = np.median(heights) if heights else 10.0
        # Lines within this delta are treated as the same text line.
        delta = max(10.0, 0.6 * median_height)

        entries.sort(key=lambda item: item[0])
        clustered: List[OCRLine] = []
        current_line: List[str] = []
        current_y: Optional[float] = None

        for y_center, text in entries:
            if current_y is None:
                current_y = y_center
                current_line = [text]
                continue

            if abs(y_center - current_y) <= delta:
                current_line.append(text)
                current_y = (current_y + y_center) / 2.0
            else:
                clustered.append(OCRLine(y_center=current_y, texts=current_line))
                current_y = y_center
                current_line = [text]

        if current_line:
            clustered.append(OCRLine(y_center=current_y or 0.0, texts=current_line))

        return clustered

    def _preprocess(self, image_path: Path) -> Tuple[Path, Optional[tempfile.NamedTemporaryFile]]:
        """Lightweight preprocessing: grayscale + auto-contrast + slight pad."""
        try:
            img = Image.open(image_path).convert("L")
            img = ImageOps.autocontrast(img)
            # Pad 2px to reduce edge clipping
            img = ImageOps.expand(img, border=2, fill=255)
            tmp = tempfile.NamedTemporaryFile(suffix=image_path.suffix or ".png", delete=False)
            img.save(tmp.name)
            return Path(tmp.name), tmp
        except Exception:
            self.logger.exception("Preprocess failed; fallback to raw image for %s", image_path)
            return image_path, None
