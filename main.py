import logging
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

from knowledge_base import KnowledgeBase
from llm_client import LLMClient
from ocr_engine import OCREngine
from pipeline import DocumentPipeline, PipelineResult


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_pipeline(image_dir: Path, output_csv: Path, suggestion_md: Path) -> None:
    load_dotenv()
    setup_logging()
    logger = logging.getLogger("main")

    kb = KnowledgeBase(dict_path=Path("final_law_dict.txt"))
    ocr = OCREngine()
    llm = LLMClient()
    pipeline = DocumentPipeline(llm=llm, kb=kb, ocr=ocr)

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    if not images:
        logger.warning("No images found in %s", image_dir)
        return

    results: List[PipelineResult] = []
    for img_path in images:
        result = pipeline.process_image(img_path)
        if result:
            results.append(result)
            append_suggestion(suggestion_md, result)

    if not results:
        logger.warning("No results produced.")
        return

    df = pd.DataFrame([
        {
            "file_name": r.file_name,
            "file_type": r.file_type,
            "raw_summary": r.raw_summary,
            "corrected_content": r.corrected_content,
            "risks": r.risks,
            "safety_score": r.safety_score,
        }
        for r in results
    ])

    write_mode = "a" if output_csv.exists() else "w"
    df.to_csv(output_csv, mode=write_mode, index=False, header=not output_csv.exists())
    logger.info("Results saved to %s", output_csv)


def append_suggestion(path: Path, result: PipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"## {result.file_name}\n")
        f.write(f"- Type: {result.file_type}\n")
        f.write(f"- Suggestion: {result.suggestion or 'None'}\n\n")


if __name__ == "__main__":
    IMAGE_DIR = Path("img")
    OUTPUT_CSV = Path("result_report.csv")
    SUGGESTION_MD = Path("suggestion.md")
    run_pipeline(IMAGE_DIR, OUTPUT_CSV, SUGGESTION_MD)
