import argparse
import logging
from pathlib import Path
from typing import Iterable, List

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


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legal document OCR + LLM pipeline")
    parser.add_argument("--file", type=Path, help="Single image file to process", dest="file")
    parser.add_argument("--dir", type=Path, help="Directory of images to process", dest="directory")
    parser.add_argument("--result-root", type=Path, default=Path("result"), help="Root directory for outputs")
    return parser.parse_args()


def collect_images(single_file: Path | None, directory: Path | None) -> List[Path]:
    if single_file:
        if single_file.suffix.lower() not in SUPPORTED_SUFFIXES:
            return []
        if not single_file.exists():
            return []
        return [single_file]
    dir_path = directory or Path("img")
    return [p for p in dir_path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES]


def run_pipeline(image_dir: Path, output_csv: Path, suggestion_md: Path, result_root: Path) -> None:
    load_dotenv()
    setup_logging()
    logger = logging.getLogger("main")

    kb = KnowledgeBase(dict_path=Path("final_law_dict.txt"))
    ocr = OCREngine()
    llm = LLMClient()
    pipeline = DocumentPipeline(llm=llm, kb=kb, ocr=ocr)

    images = collect_images(None, image_dir)
    if not images:
        logger.warning("No images found in %s", image_dir)
        return

    aggregated: List[PipelineResult] = []
    for img_path in images:
        result = pipeline.process_image(img_path)
        if not result:
            continue
        aggregated.append(result)
        save_per_image_outputs(result_root, img_path, result)
        append_suggestion(suggestion_md, result)

    if not aggregated:
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
        for r in aggregated
    ])

    write_mode = "a" if output_csv.exists() else "w"
    df.to_csv(output_csv, mode=write_mode, index=False, header=not output_csv.exists())
    logger.info("Aggregated CSV saved to %s", output_csv)


def append_suggestion(path: Path, result: PipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"## {result.file_name}\n")
        f.write(f"- Type: {result.file_type}\n")
        f.write(f"- Suggestion: {result.suggestion or 'None'}\n\n")


def save_per_image_outputs(result_root: Path, image_path: Path, result: PipelineResult) -> None:
    stem = image_path.stem
    out_dir = result_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary: OCR text (raw recognition result)
    ocr_path = out_dir / f"{stem}_result.txt"
    ocr_path.write_text(result.ocr_text, encoding="utf-8")

    # Secondary: report (JSON for structure)
    report_path = out_dir / "report.json"
    report_payload = {
        "file_name": result.file_name,
        "file_type": result.file_type,
        "raw_summary": result.raw_summary,
        "corrected_content": result.corrected_content,
        "risks": result.risks,
        "safety_score": result.safety_score,
        "suggestion": result.suggestion,
    }
    report_path.write_text(pd.Series(report_payload).to_json(force_ascii=False, indent=2), encoding="utf-8")

    # Human-readable report.md
    md_path = out_dir / "report.md"
    md_lines = [
        f"# Report for {result.file_name}",
        f"- Type: {result.file_type}",
        f"- Safety score: {result.safety_score}",
        f"- Risks: {result.risks or 'None'}",
        "",
        "## Corrected Content",
        result.corrected_content or "(empty)",
        "",
        "## Suggestion",
        result.suggestion or "None",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    target_dir = args.directory or Path("img")
    output_csv = Path("result_report.csv")
    suggestion_md = Path("suggestion.md")

    # If a single file is provided, process only that file.
    if args.file:
        images = collect_images(args.file, None)
        if not images:
            raise SystemExit(f"No valid image found: {args.file}")
        # Run pipeline with temp directory as parent of file for iteration.
        run_single_list(images, output_csv, suggestion_md, args.result_root)
    else:
        run_pipeline(target_dir, output_csv, suggestion_md, args.result_root)


def run_single_list(images: Iterable[Path], output_csv: Path, suggestion_md: Path, result_root: Path) -> None:
    load_dotenv()
    setup_logging()
    logger = logging.getLogger("main")

    kb = KnowledgeBase(dict_path=Path("final_law_dict.txt"))
    ocr = OCREngine()
    llm = LLMClient()
    pipeline = DocumentPipeline(llm=llm, kb=kb, ocr=ocr)

    aggregated: List[PipelineResult] = []
    for img_path in images:
        result = pipeline.process_image(img_path)
        if not result:
            continue
        aggregated.append(result)
        save_per_image_outputs(result_root, img_path, result)
        append_suggestion(suggestion_md, result)

    if not aggregated:
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
        for r in aggregated
    ])

    write_mode = "a" if output_csv.exists() else "w"
    df.to_csv(output_csv, mode=write_mode, index=False, header=not output_csv.exists())
    logger.info("Aggregated CSV saved to %s", output_csv)
