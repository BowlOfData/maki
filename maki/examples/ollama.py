import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import maki.maki

maki = maki.maki.Maki("http://localhost", "11434", "gemma3:27b")

data = {
    "model":maki.get_model(),
    "prompt":"tell me your model",
    "stream":False
    }


result = maki.request(data)

print(result)

