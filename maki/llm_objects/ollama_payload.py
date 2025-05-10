import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

OLLAMA_PAYLOAD = {
    "model":"",
    "prompt":"",
    "stream":False
    }