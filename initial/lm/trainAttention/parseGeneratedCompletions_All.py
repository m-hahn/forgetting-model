import os
PATH = "/u/scr/mhahn/reinforce-logs-both/full-logs/"
processed = os.listdir("/u/scr/mhahn/reinforce-logs-both/full-logs-gpt2/")
import subprocess
import sys
import os
for f in sorted(os.listdir(PATH)):
   if "GPT2" in f:
      ID = f[f.rfind("_")+1:]
      print(ID)
      if any(["_"+ID in x for x in processed]):
        continue
      subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "parseGeneratedCompletions.py", ID])
