import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import maki.maki

from maki.llm_objects.ollama_payload import OLLAMA_PAYLOAD

maki = maki.maki.Maki("http://localhost", "11434", "gemma3:27b")

OLLAMA_PAYLOAD = OLLAMA_PAYLOAD

OLLAMA_PAYLOAD["model"] = maki.get_model()
OLLAMA_PAYLOAD["prompt"] = "this is a test"

result = maki.request(OLLAMA_PAYLOAD)

print(result)

