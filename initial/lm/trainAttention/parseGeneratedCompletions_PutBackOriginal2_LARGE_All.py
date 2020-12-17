import os
PATH = "/u/scr/mhahn/reinforce-logs-both/full-logs/"
import subprocess
import sys
import os
files = os.listdir(PATH)
import random
random.shuffle(files)
for f in files:
   if "GPT2" in f and "swp" not in f:
      ID = f[f.rfind("_")+1:]
      print(ID)
      processed = os.listdir("/u/scr/mhahn/reinforce-logs-both/full-logs-gpt2-withOrig2/")
      if any(["LARGE" in x and "_"+ID in x for x in processed]):
        continue
      subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "parseGeneratedCompletions_PutBackOriginal2_LARGE.py", ID])
