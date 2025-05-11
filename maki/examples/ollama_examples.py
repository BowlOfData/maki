
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from maki import Maki

ma = Maki("http://localhost", "11434", "gemma3:27b", temperature=0)

result = ma.request("test")

print(ma.version())

print(result)

