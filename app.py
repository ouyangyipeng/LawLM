import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from knowledge_base import KnowledgeBase
from llm_client import LLMClient
from ocr_engine import OCREngine
from pipeline import DocumentPipeline
from main import save_per_image_outputs, _enrich_similarity

try:
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
    convert_from_bytes = None


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pdf"}


def main() -> None:
    st.set_page_config(page_title="LawLM 文书处理", layout="wide")
    load_dotenv()

    st.title("律所辅助文书处理系统")
    st.caption("OCR → 路由 → 知识库预修正 → 双LLM辩论 → 仲裁 → 报告/红线DOCX")

    left, mid, right = st.columns([1, 1, 1.2])

    with left:
        st.header("上传")
        uploaded = st.file_uploader("选择图片或PDF", type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES])
        prompt_profile = st.selectbox("提示词配置", ["default", "finance", "litigation"], index=0)
        preprocess = st.checkbox("开启OCR预处理 (灰度+自适应对比度)", value=True)
        run_btn = st.button("运行流水线", type="primary", use_container_width=True)

    with mid:
        st.header("进度")
        progress = st.progress(0, text="等待开始")
        status_placeholder = st.empty()

    with right:
        st.header("结果")
        result_placeholder = st.empty()
        download_placeholder = st.empty()

    if not run_btn or uploaded is None:
        return

    tmp_input = _save_upload(uploaded)
    if tmp_input is None:
        status_placeholder.error("无法保存上传文件")
        return

    progress.progress(10, text="1/4 OCR 预处理与识别")
    status_placeholder.write("启动 OCR 与流水线…")

    # Initialize components
    kb = KnowledgeBase(dict_path=Path("final_law_dict.txt"))
    ocr = OCREngine(enable_preprocess=preprocess)
    llm = LLMClient()
    prompt_dir = Path("prompt") / prompt_profile if prompt_profile != "default" else Path("prompt")
    pipeline = DocumentPipeline(llm=llm, kb=kb, ocr=ocr, prompt_dir=prompt_dir)

    # If PDF, convert first page to temp image
    input_path = tmp_input
    cleanup_paths = []
    if tmp_input.suffix.lower() == ".pdf":
        if convert_from_bytes is None:
            status_placeholder.error("pdf2image 未安装或不可用，无法处理 PDF。")
            return
        pages = convert_from_bytes(uploaded.read(), dpi=300)
        if not pages:
            status_placeholder.error("PDF 转图片失败")
            return
        pdf_img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pages[0].save(pdf_img_tmp.name)
        input_path = Path(pdf_img_tmp.name)
        cleanup_paths.append(pdf_img_tmp.name)

    progress.progress(40, text="2/4 路由与双LLM辩论")
    result = pipeline.process_image(input_path)
    if not result:
        status_placeholder.error("流水线处理失败")
        _cleanup(tmp_input, cleanup_paths)
        return

    result = _enrich_similarity(result)
    progress.progress(70, text="3/4 生成报告与红线文档")

    # Save outputs to result/<stem>
    out_root = Path("result")
    save_per_image_outputs(out_root, input_path, result)

    progress.progress(100, text="4/4 完成")
    status_placeholder.success("处理完成")

    # Display summary
    with result_placeholder.container():
        st.subheader("识别摘要")
        st.text_area("OCR 原文", result.ocr_text, height=160)
        st.text_area("修正后内容", result.corrected_content, height=220)
        st.markdown(f"**文件类型**: {result.file_type}")
        st.markdown(f"**安全评分**: {result.safety_score}")
        st.markdown(f"**自动相似度**: {result.similarity:.3f}")
        st.markdown("**风险点**:")
        if result.risks:
            if isinstance(result.risks, list):
                for r in result.risks:
                    st.write(f"- {r}")
            else:
                for r in str(result.risks).replace("\n", ";").split(";"):
                    r = r.strip()
                    if r:
                        st.write(f"- {r}")
        else:
            st.write("- 未检测到风险")

    # Offer downloads
    stem = input_path.stem
    docx_path = out_root / stem / "redline.docx"
    report_json = (out_root / stem / "report.json").read_bytes()
    if docx_path.exists():
        with open(docx_path, "rb") as f:
            download_placeholder.download_button(
                label="下载修订版 Word",
                data=f,
                file_name=f"{stem}_redline.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
    st.download_button(
        label="下载结构化报告 JSON",
        data=report_json,
        file_name=f"{stem}_report.json",
        mime="application/json",
        use_container_width=True,
    )

    _cleanup(tmp_input, cleanup_paths)


def _save_upload(uploaded) -> Optional[Path]:
    try:
        suffix = Path(uploaded.name).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        return Path(tmp.name)
    except Exception:
        return None


def _cleanup(tmp_input: Path, extras: list[str]) -> None:
    try:
        Path(tmp_input).unlink(missing_ok=True)
    except Exception:
        pass
    for p in extras:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
