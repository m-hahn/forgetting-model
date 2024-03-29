import random
import subprocess
scripts = []

import sys
stimulus_file = sys.argv[1]

if stimulus_file == "BartekEtal":
   criticalRegions="Critical_0"
else:
   assert False

script = "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPLo.py"

import glob
models = glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPS.py_*.model")

for model in models:
   ID = model[model.rfind("_")+1:model.rfind(".")]
   if len(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/{script}_{stimulus_file}_{ID}_Model"))>0:
     print("EXISTS", ID)
     continue
   print("DOES NOT EXIST", ID)
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", script, "--stimulus_file="+stimulus_file, "--criticalRegions="+criticalRegions, "--load_from_joint="+ID]
   print(command)
   subprocess.call(command)
