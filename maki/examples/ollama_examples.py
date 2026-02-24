
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki import Maki

#ma = Maki("http://localhost", "11434", "gemma3:27b", temperature=0)
ma = Maki("http://localhost", "11434", "qwen3-coder:30b", temperature=0)


result = ma.request("test")

#result = ma.request_with_images("""analyze the following image and report which setting you would change to improve the quality,
#                                if your rating is > 8 don't report any improvement and the only ouptput shall be: FINE!, return the ooutput in json format. 
#                                if your rating is < 8 return in json format the parameters to improve and their numerical values and specify if are positive or negative
#                                """,
#                                "/Users/marcoparrillo/code/maki/img/test2.jpg")

print(ma.version())

print(result)

