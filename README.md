# 律所辅助文书处理系统 (RapidOCR + DeepSeek)

## 概览
在 WSL2/Ubuntu 22.04 + Conda 环境下，构建面向法律文书的自动化流水线：

```
OCR 识别 → 智能分类(路由) → 知识库预修正 → 双LLM并行校验(辩论) → 一致性仲裁 → 报告产出
```

**技术要点**
- OCR：`rapidocr_onnxruntime`（CPU 推理），兼容 list/结构化输出，按 y 轴聚类恢复阅读顺序。
- 知识库：基于 `final_law_dict.txt` + RapidFuzz 模糊匹配，生成纠错提示注入 LLM。
- LLM：DeepSeek API（OpenAI SDK 兼容），双路角色并行（Reviewer/Proofreader），Arbiter 仲裁，最多 3 轮。
- 提示词：分阶段模板存放 `prompt/`，可审计、可复用。
- 输出：
  - 每图独立目录：`result/<stem>/`，含 `*_result.txt`（OCR 主输出）、`report.json`、`report.md`。
  - 汇总：`result_report.csv` 与 `suggestion.md`。
- CLI：支持单文件或目录批处理，缺省遍历 `./img`。

## 目录结构
```
├── main.py              # CLI 入口，参数解析、批处理、落盘
├── pipeline.py          # 业务流水线：路由→辩论→仲裁→结果
├── ocr_engine.py        # RapidOCR 封装，y 轴聚类恢复阅读顺序
├── knowledge_base.py    # 词库模糊匹配与提示构造
├── llm_client.py        # DeepSeek (OpenAI SDK) 封装，重试+回退
├── prompt/              # 分阶段提示词模板
│   ├── router.txt
│   ├── reviewer.txt
│   ├── proofreader.txt
│   └── arbiter.txt
├── img/                 # 输入图片目录（默认）
├── result/              # 每图独立输出目录
├── result_report.csv    # 汇总表（追加写入）
├── suggestion.md        # 汇总建议
├── final_law_dict.txt   # 法律术语大词库
└── requirements.txt
```

## 流水线细节（论文式说明）
1) **OCR 识别** (`ocr_engine.py`)
   - 兼容 RapidOCR dataclass / list 返回格式，异常与空结果均安全处理。
   - 按 bbox y 中心 + 高度自适应阈值聚类行，重建阅读顺序并输出完整文本。

2) **知识库预修正** (`knowledge_base.py`)
   - RapidFuzz `QRatio` 对分词/字串做模糊匹配，形成候选纠错列表。
   - 将纠错提示嵌入 LLM prompt，约束法律术语与专有名词的准确性。

3) **智能路由** (`pipeline.py` → `prompt/router.txt`)
   - DeepSeek 温度 0.3，输出严格五类之一 `[invoice, contract, certificate, regulation, other]`。

4) **双路辩论** (`pipeline.py` → `prompt/reviewer.txt`, `prompt/proofreader.txt`)
   - Reviewer：关注语义、合规、逻辑；Proofreader：关注 OCR 纠错、形近字与数字精度。
   - 并行推理，强制 JSON 输出 `{corrected_content, risks, safety_score}`。

5) **一致性仲裁** (`pipeline.py` → `prompt/arbiter.txt`)
   - 文本相似度阈值 0.78 以上直接合并；否则 Arbiter 介入（温度 0.6），最多 3 轮。
   - 失败回退：选择两路较优合并，并标记“Consensus failed”提示。

6) **落盘策略** (`main.py`)
   - 每图独立目录：`result/<stem>/` 写入 `*_result.txt`（OCR 主结果）、`report.json`、`report.md`。
   - 汇总：追加写入 `result_report.csv`；建议累积到 `suggestion.md`。

## 环境准备
```bash
# 1) 创建并激活 conda 环境
conda create -n law-pipeline python=3.10 -y
conda activate law-pipeline

# 2) 安装依赖
pip install -r requirements.txt
```

## 运行方式
### 1) 默认遍历 ./img
```bash
python main.py
```

### 2) 单文件
```bash
python main.py --file img/not_complete1.png
```

### 3) 指定目录
```bash
python main.py --dir /path/to/images
```

### 4) 自定义输出根目录
```bash
python main.py --dir img --result-root out_dir
```

## 输入/输出说明
- 输入：`img/` 下的图片，支持 `.png .jpg .jpeg .bmp .tif .tiff`。
- 单图输出：`result/<stem>/`
  - `<stem>_result.txt`：OCR 主文本（首位产出）。
  - `report.json`：结构化结果（类别、摘要、纠正内容、风险、安全分、建议）。
  - `report.md`：可读报告。
- 汇总输出：`result_report.csv`（追加）、`suggestion.md`（建议汇总）。

## 技术特色摘要
- 稳健 OCR 解析：兼容多返回格式，行聚类恢复阅读顺序。
- 知识库增强：大词库模糊纠错提示先验约束 LLM。
- 双路对抗 + 仲裁：Reviewer/Proofreader 并行 + Arbiter 三角校验。
- 可审计提示词：模板集中管理，便于复用、审计与微调。
- 单图隔离产出：溯源友好，汇总文件便于批量分析。

## 参考运行记录
- 已在 `img/1..5.jpg` 与 `img/not_complete1.png` 上验证，输出均落盘到 `result/` 对应子目录，并追加到 `result_report.csv`。

## 后续可拓展方向
- GPU 版 RapidOCR / OCR 前处理（去噪、旋转校正）。
- RAG 增强：向量检索提供上下文支撑。
- 自动评测：CER/WER 与法律要素抽取准确率基准。
- 模板参数化：多语言/多条线合规提示自动切换。
