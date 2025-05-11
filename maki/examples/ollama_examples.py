
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import maki

ma = maki.Maki("http://localhost", "11434", "gemma3:27b", temperature=0)

result = ma.request("test")

print(ma.version())

print(result)

