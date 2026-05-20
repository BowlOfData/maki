# OCR Plugin for Maki Framework

Extracts text from documents (PDF, DOCX, XLSX, images, …) and writes the result as a Markdown file to a configurable output directory.

## Features

- Multiple extraction backends (LLM-preferred, library-based fallbacks)
- Auto-selects backend based on file extension or maki_instance availability
- Output always written as `.md` under a configurable directory
- Path-safe input resolution (no directory traversal)
- Graceful degradation when optional dependencies are missing
- Full agent integration via `TOOL:` directives

## Backends

| Name    | Engine                         | Optional deps                        |
|---------|-------------------------------|--------------------------------------|
| `llm`   | Vision LLM via Ollama (**preferred**) | `pdf2image`, `libreoffice` (for PDF/DOCX/XLSX) |
| `pdf`   | pdfplumber / pytesseract      | `pdfplumber`, `pdf2image`, `pytesseract` |
| `docx`  | python-docx                   | `python-docx`                        |
| `xlsx`  | openpyxl                      | `openpyxl`                           |
| `image` | pytesseract                   | `pytesseract`, `Pillow`              |

The `llm` backend defaults to **glm-ocr** (available via Ollama). Configure via `MAKI_OCR_MODEL`.

## Configuration

| Env var              | Default             | Description                          |
|----------------------|---------------------|--------------------------------------|
| `MAKI_OCR_OUTPUT_DIR` | `~/maki_ocr_output` | Directory where `.md` files are saved |
| `MAKI_OCR_MODEL`      | `glm-ocr`           | Ollama model for the llm backend     |

## Usage

### Standalone

```python
from maki import MakiLLama
from maki.plugins.ocr import OCR

maki = MakiLLama(model="glm-ocr")
ocr = OCR(maki_instance=maki)

# Extract and print Markdown
result = ocr.extract("invoice.pdf")
if result["success"]:
    print(result["markdown"])

# Extract and save to ~/maki_ocr_output/invoice.md
result = ocr.extract_to_file("invoice.pdf")
print(result["output_path"])

# Check available backends
print(ocr.list_backends())
```

### Custom output directory

```python
ocr = OCR(maki_instance=maki, output_dir="~/my_docs/ocr")
```

### Force a specific backend

```python
# Library-based PDF extraction (no LLM needed)
result = ocr.extract("report.pdf", backend="pdf")

# LLM extraction with custom prompt
result = ocr.extract(
    "scan.png",
    backend="llm",
    options={"user_prompt": "Extract only table data as a Markdown table."},
)
```

### In an agent

```python
from maki.agents import Agent

agent = Agent(name="DocParser", maki_instance=maki, role="document processor")
agent.load_plugin("ocr")

response = agent.execute_task(
    "Extract the text from contract.pdf and save it.",
    use_plugins=True,
)
```

The agent can emit:
```
TOOL: {"plugin": "ocr", "method": "extract_to_file", "args": {"file_path": "contract.pdf"}}
```

## Installing optional dependencies

Dependencies are commented out in `requirements.txt`. Install only what you need:

```bash
# LLM backend (PDF/DOCX/XLSX rasterisation)
pip install pdf2image
# Also install poppler: brew install poppler  (macOS) / apt install poppler-utils (Linux)

# Library-based PDF extraction
pip install pdfplumber

# DOCX extraction
pip install python-docx

# XLSX extraction
pip install openpyxl

# Image OCR
pip install pytesseract Pillow
# Also install tesseract: brew install tesseract  (macOS) / apt install tesseract-ocr (Linux)

# DOCX/XLSX → PDF conversion for the LLM backend
# Install LibreOffice: https://www.libreoffice.org/download/download/
```
