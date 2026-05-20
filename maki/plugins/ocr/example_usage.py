"""
Example usage of the OCR plugin with the Maki framework.

Requires a running Ollama instance with glm-ocr pulled:
  ollama pull glm-ocr
"""

from maki import MakiLLama
from maki.agents import Agent
from maki.plugins.ocr import OCR

# Vision-capable Ollama model for OCR.
maki = MakiLLama(model="glm-ocr", base_url="http://localhost:11434")

ocr = OCR(
    maki_instance=maki,
    output_dir="~/my_ocr_output",   # overrides MAKI_OCR_OUTPUT_DIR
)


def example_extract_image(image_path: str) -> None:
    """Extract text from a scanned image and print the Markdown."""
    result = ocr.extract(image_path)
    if result["success"]:
        print(f"Extracted {result['pages']} page(s) via '{result['backend']}':\n")
        print(result["markdown"])
    else:
        print(f"Error: {result['error']}")


def example_extract_pdf_to_file(pdf_path: str) -> None:
    """Extract a PDF and write the result to ~/my_ocr_output/<name>.md."""
    result = ocr.extract_to_file(pdf_path)
    if result["success"]:
        print(f"Saved: {result['output_path']}")
    else:
        print(f"Error: {result['error']}")


def example_custom_backend(docx_path: str) -> None:
    """Force the library-based DOCX backend (no LLM required)."""
    result = ocr.extract(docx_path, backend="docx")
    print(result["markdown"] if result["success"] else result["error"])


def example_agent_usage() -> None:
    """Let an agent decide which document to OCR via a TOOL: directive."""
    agent = Agent(name="DocParser", maki_instance=maki, role="document processor")
    agent.load_plugin("ocr")

    # The agent will emit:
    # TOOL: {"plugin": "ocr", "method": "extract_to_file",
    #        "args": {"file_path": "invoice.pdf"}}
    response = agent.execute_task(
        "Extract the text from invoice.pdf and save it as Markdown.",
        use_plugins=True,
    )
    print(response)


def example_list_backends() -> None:
    """Show which backends are available in the current environment."""
    for name, info in ocr.list_backends().items():
        status = "available" if info["available"] else "unavailable (missing deps)"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    print("OCR plugin — example usage")
    print("=" * 40)

    print("\nAvailable backends:")
    example_list_backends()

    print("\nExtract a scanned PNG:")
    example_extract_image("sample_scan.png")

    print("\nExtract a PDF to file:")
    example_extract_pdf_to_file("sample_doc.pdf")
