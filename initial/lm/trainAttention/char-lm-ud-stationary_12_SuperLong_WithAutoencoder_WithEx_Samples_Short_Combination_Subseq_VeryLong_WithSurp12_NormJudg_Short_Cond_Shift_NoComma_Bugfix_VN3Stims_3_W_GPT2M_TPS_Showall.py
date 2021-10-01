import sys
import random
import subprocess
scripts = []

import sys

script = "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPS.py"

from collections import defaultdict

import glob
for _ in range(int(sys.argv[1])):


   countsByConfig = defaultdict(int)
   configurations = set()
   for i in range(5, 100, 5):
#     if i/100 < 0.2 or i/100 > 0.75:
 #      continue
     for j in [1]: #[0, 0.25, 0.5, 0.75, 1]:
        configurations.add((i/100,j))

   logs = glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/results/{script}_*")

   for log in logs:
      with open(log, "r") as inFile:
          args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ")])
#      print(log)
      try:
         countsByConfig[(float(args["deletion_rate"]), float(args["predictability_weight"]))] += 1
         ID = args["myID"]
         print(float(args["deletion_rate"]), float(args["predictability_weight"]), ID)
      except KeyError:
         print("ERROR", args)
         pass
#      print(configurations)
   break
