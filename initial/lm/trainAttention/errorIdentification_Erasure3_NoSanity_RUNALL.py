import random
import subprocess
scripts = []

import sys
if len(sys.argv) > 1:
  stimulus_file = sys.argv[1]
else:
  stimulus_file = "tabor_2004_expt1_3_tokenized" #random.choice(["BartekEtal", "Staub2006", "Staub_2016"])

if stimulus_file == "tabor_2004_expt1_3_tokenized":
  criticalRegions = "participle_0"
else:
   assert False, stimulus_file

script = "errorIdentification_Erasure3_NoSanity.py"

import glob
models = glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_S.py_*.model")
random.shuffle(models)
limit = 1000
count = 0
for model in models:
   ID = model[model.rfind("_")+1:model.rfind(".")]
   if len(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv-EIC/{script}_{stimulus_file}_{ID}_Model"))>0:
     print("EXISTS", ID)
     continue
   with open(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/results/*_{ID}")[0], "r") as inFile:
      args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ") ])
      delta = float(args["deletion_rate"])
      lambda_ = float(args["predictability_weight"])
#      if lambda_ != 1:
 #       print("FOR NOW DON'T CONSIDER")
#      if delta < 0.35 or delta > 0.6:
 #       print("FOR NOW DON'T CONSIDER", ID, delta)
#        continue
   print("DOES NOT EXIST", ID, delta, lambda_)
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", script, "--stimulus_file="+stimulus_file, "--criticalRegions="+criticalRegions, "--load_from_joint="+ID]
   print(command)
   subprocess.call(command)
   count += 1
   if count >= limit:
     break
