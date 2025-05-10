import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from maki.maki import Maki

maki = Maki("http://localhost", "11434", "gemma3:27b", temperature=0)

result = maki.request("test")

print(maki.version())

print(result)

