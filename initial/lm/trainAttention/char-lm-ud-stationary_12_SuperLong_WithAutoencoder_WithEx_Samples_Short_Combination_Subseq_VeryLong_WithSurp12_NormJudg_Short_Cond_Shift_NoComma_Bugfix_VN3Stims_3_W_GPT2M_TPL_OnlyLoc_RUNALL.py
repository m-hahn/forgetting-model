import random
import subprocess
scripts = []

import sys

script = "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPL_OnlyLoc.py"

import glob
models = glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY_OTHERS/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPS_OnlyLoc.py_*.model")
print(models)
#quit()
random.shuffle(models)
if len(sys.argv) > 1:
   limit = int(sys.argv[1])
else:
   limit = 1000
import os
import time
count = 0
for model in models:
   ID = model[model.rfind("_")+1:model.rfind(".")]
   resultsPath = f"/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem-onlyLoc/{script}_{ID}_Model"
   if len(glob.glob(resultsPath))>0:
     if os.path.getsize(resultsPath) > 0 and time.time() - os.stat(resultsPath).st_mtime < 3600: # written to within the last hour
       print("WORKING ON THIS?", ID)
       continue
     else:
       with open(resultsPath, "r") as inFile:
         nouns = set()
         for line in inFile:
          nouns.add(line[:line.find("\t")])
       if len(nouns) >= 50:
         print("EXISTS", ID, os.path.getsize(resultsPath), len(nouns))
         continue
   with open(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/results/*_{ID}")[0], "r") as inFile:
      args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ") ])
      delta = float(args["deletion_rate"])
      lambda_ = float(args["predictability_weight"])
#      if lambda_ < 1:
 #       print("EXCLUDE", ID)
  #      continue
      if delta < 0.2 or delta > 0.8:
        print("EXCLUDE for now extreme delta", ID)
        continue
#      if delta < 0.2: # and delta*10 != int(delta*10):
 #       print("EXCLUDE", ID)
  #      continue
#      if lambda_ != 1 or delta not in [0.7, 0.75, 0.8]:
 #       continue
   #   if delta > 0.8: # and delta*10 != int(delta*10):
    #    print("EXCLUDE", ID)
     #   continue

   print("DOES NOT EXIST", ID, delta, lambda_)
   #continue
#   continue
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", script, "--load_from_joint="+ID]
   print(command)
   subprocess.call(command)
   count += 1
   if count >= limit:
     break
