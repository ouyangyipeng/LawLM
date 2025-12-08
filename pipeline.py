import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional, Tuple

from knowledge_base import KnowledgeBase
from llm_client import LLMClient
from ocr_engine import OCREngine, OCRResult


@dataclass
class PipelineResult:
    file_name: str
    file_type: str
    raw_summary: str
    ocr_text: str
    corrected_content: str
    risks: str
    safety_score: float
    suggestion: str
    similarity: float = 0.0


class DocumentPipeline:
    def __init__(self, llm: LLMClient, kb: KnowledgeBase, ocr: OCREngine, prompt_dir: Path = Path("prompt")) -> None:
        self.llm = llm
        self.kb = kb
        self.ocr = ocr
        self.prompt_dir = prompt_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_image(self, image_path: Path) -> Optional[PipelineResult]:
        self.logger.info("Processing %s", image_path)

        ocr_result = self.ocr.run(image_path)
        if ocr_result is None:
            self.logger.error("Skipping %s due to OCR failure", image_path)
            return None

        kb_corrections = self.kb.search_and_correct(ocr_result.raw_text)
        kb_hint = self.kb.build_hint(kb_corrections)

        doc_type = self._route_document(ocr_result.raw_text)
        reviewer_out, proofreader_out = self._debate(ocr_result.raw_text, kb_hint)

        final = self._consensus(
            doc_type=doc_type,
            ocr_text=ocr_result.raw_text,
            kb_hint=kb_hint,
            reviewer_out=reviewer_out,
            proofreader_out=proofreader_out,
        )

        similarity = self._similarity(ocr_result.raw_text, final.get("corrected_content", ""))
        raw_summary = (ocr_result.raw_text[:500] + "...") if len(ocr_result.raw_text) > 500 else ocr_result.raw_text

        return PipelineResult(
            file_name=image_path.name,
            file_type=doc_type,
            raw_summary=raw_summary,
            ocr_text=ocr_result.raw_text,
            corrected_content=final.get("corrected_content", ""),
            risks=final.get("risks", ""),
            safety_score=float(final.get("safety_score", 0.0)),
            suggestion=final.get("suggestion", ""),
            similarity=similarity,
        )

    def _route_document(self, text: str) -> str:
        prompt = self._render_prompt("router", content_snippet=text[:1500])
        try:
            response = self.llm.send_chat(prompt, temperature=0.3)
            cleaned = response.lower().strip()
            for option in ["invoice", "contract", "certificate", "regulation", "other"]:
                if option in cleaned:
                    return option
        except Exception:
            self.logger.exception("Routing failed; defaulting to other")
        return "other"

    def _debate(self, text: str, kb_hint: str) -> Tuple[Dict, Dict]:
        reviewer_prompt = self._render_prompt(
            "reviewer",
            kb_hint=kb_hint or "none",
            text_snippet=text[:2000],
        )
        proofreader_prompt = self._render_prompt(
            "proofreader",
            kb_hint=kb_hint or "none",
            text_snippet=text[:2000],
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            reviewer_future = executor.submit(
                self.llm.send_chat,
                reviewer_prompt,
                None,
                1.0,
            )
            proofreader_future = executor.submit(
                self.llm.send_chat,
                proofreader_prompt,
                None,
                0.5,
            )
            reviewer_raw = reviewer_future.result()
            proofreader_raw = proofreader_future.result()

        reviewer_out = self._parse_json_response(reviewer_raw)
        proofreader_out = self._parse_json_response(proofreader_raw)
        reviewer_out["_raw"] = reviewer_raw
        proofreader_out["_raw"] = proofreader_raw
        return reviewer_out, proofreader_out

    def _consensus(
        self,
        doc_type: str,
        ocr_text: str,
        kb_hint: str,
        reviewer_out: Dict,
        proofreader_out: Dict,
    ) -> Dict:
        reviewer_text = reviewer_out.get("corrected_content", "")
        proofreader_text = proofreader_out.get("corrected_content", "")
        similarity = self._similarity(reviewer_text, proofreader_text)

        if similarity >= 0.78:
            merged = self._merge_results(reviewer_out, proofreader_out)
            merged.setdefault("suggestion", "No additional suggestions.")
            return merged

        consensus = {}
        for _ in range(3):
            try:
                arb_user = self._render_prompt(
                    "arbiter",
                    doc_type=doc_type,
                    kb_hint=kb_hint or "none",
                    ocr_snippet=ocr_text[:1500],
                    reviewer_raw=reviewer_out.get("_raw") or str(reviewer_out),
                    proofreader_raw=proofreader_out.get("_raw") or str(proofreader_out),
                )
                arb_response = self.llm.send_chat(user_prompt=arb_user, system_prompt=None, temperature=0.6)
                consensus = self._parse_json_response(arb_response)
                consensus["_raw"] = arb_response
            except Exception:
                self.logger.exception("Arbiter call failed; retrying")
                continue
            if consensus.get("corrected_content"):
                break

        if not consensus.get("corrected_content"):
            consensus = self._merge_results(reviewer_out, proofreader_out)
            consensus.setdefault("suggestion", "Consensus failed; merged best effort.")
        return consensus

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _merge_results(a: Dict, b: Dict) -> Dict:
        # Prefer longer corrected content assuming it is more complete.
        content = a.get("corrected_content", "")
        alt_content = b.get("corrected_content", "")
        corrected_content = content if len(content) >= len(alt_content) else alt_content

        risks = a.get("risks") or b.get("risks") or ""
        score_a = float(a.get("safety_score", 0) or 0)
        score_b = float(b.get("safety_score", 0) or 0)
        safety_score = min(max(score_a, score_b), 100.0)

        suggestion = a.get("suggestion") or b.get("suggestion") or ""
        return {
            "corrected_content": corrected_content,
            "risks": risks,
            "safety_score": safety_score,
            "suggestion": suggestion,
        }

    def _parse_json_response(self, text: str) -> Dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`\n ")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned)
        except Exception:
            # Attempt to locate a JSON object within the text.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except Exception:
                    pass
        # Fallback: return as plain text.
        return {"corrected_content": cleaned, "risks": "", "safety_score": 0}

    def _render_prompt(self, name: str, **kwargs) -> str:
        path = self.prompt_dir / f"{name}.txt"
        try:
            template = path.read_text(encoding="utf-8")
        except Exception:
            self.logger.exception("Missing or unreadable prompt: %s", path)
            return ""
        try:
            return template.format(**kwargs)
        except Exception:
            self.logger.exception("Prompt formatting failed for %s with %s", name, kwargs)
            return template
