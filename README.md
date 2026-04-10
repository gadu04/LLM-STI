# 📄 STI PDF Extraction Pipeline

A local-first Python pipeline for extracting structured information from Vietnamese scientific acceptance reports in PDF format.

The project is designed to run with **LM Studio** on `localhost`, so you do not need to manage API keys or external secrets.

## ✨ What It Does

- 📌 Extracts report metadata from the opening pages
- 🧭 Detects the KH&CN product section with a lightweight page scan
- ⚡ Extracts product details in parallel with LLM workers
- 📝 Writes the final result to `data/final_extraction.md`
- 🖥️ Provides a Streamlit UI for upload, progress, and live results

## 🧱 Project Structure

| File | Purpose |
| --- | --- |
| `main.py` | CLI entry point for the extraction pipeline |
| `streamlit_app.py` | Streamlit frontend for uploading PDFs |
| `product_tracker.py` | 3-phase extraction engine |
| `vlm_parser.py` | PDF text extraction helpers |
| `llm_client.py` | LM Studio client factory |
| `config.py` | Hardcoded local configuration for LM Studio |
| `schemas.py` | Pydantic data models |
| `output_writer.py` | Markdown output writer |
| `data/` | Input PDFs and generated outputs |

## ⚙️ Configuration

This project uses **local LM Studio settings inside `config.py`**.

Default values:

- `http://localhost:1234/v1`
- `lm-studio`
- `local-model`
- `90s timeout`
- `0.1 temperature`
- `20000 input characters`

If you want to change the endpoint, model, or timeout, edit [config.py](config.py).

## 🚀 Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

If Streamlit is not already available in your environment:

```bash
pip install streamlit
```

## ▶️ Run the CLI Pipeline

Run the default PDF in `data/`:

```bash
python main.py
```

Run a custom PDF:

```bash
python main.py --pdf path/to/file.pdf
```

The final result is written to:

```text
data/final_extraction.md
```

## 🌐 Run the Streamlit App

Start the frontend:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal.

## 🧠 Extraction Flow

The pipeline runs in 3 phases:

1. **Phase 1** - Extract metadata from the first pages
2. **Phase 2** - Find the KH&CN section and identify products
3. **Phase 3** - Extract detailed product information in parallel

## 📦 Output Fields

The final report includes:

- Tên nhiệm vụ
- Chủ nhiệm
- Tổ chức chủ trì
- Tên sản phẩm
- Loại sản phẩm
- Số lượng
- Mô tả chi tiết
- Kết quả chính
- Ghi chú

## ✅ Notes

- Optimized for Vietnamese PDFs with selectable text
- Phase 2 uses a narrow page window to reduce latency
- Output is validated and saved as Markdown for easy review
- Streamlit shows progress while extraction is running

## 📄 License

No license file is currently provided.
