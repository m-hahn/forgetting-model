import random
import subprocess
scripts = []

import sys
stimulus_file = sys.argv[1]

if stimulus_file == "BartekEtal":
   criticalRegions="Critical_0"
elif stimulus_file == "Staub2006":
   criticalRegions = "NP1_0,NP1_1,OR,NP2_0,NP2_1"
elif stimulus_file == "cunnings-sturt-2018":
   criticalRegions = "critical"
elif stimulus_file == "Staub_2016":
   criticalRegions = "V0,D1,N1,V1"
elif stimulus_file == "V11_E1_EN":
   criticalRegions = "Critical_0"
else:
   assert False, stimulus_file

script = "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_Lo.py"

import glob
models = glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_S.py_*.model")

for model in models:
   ID = model[model.rfind("_")+1:model.rfind(".")]
   if len(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/{script}_{stimulus_file}_{ID}_Model"))>0:
     print("EXISTS", ID)
     continue
   print("DOES NOT EXIST", ID)
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", script, "--stimulus_file="+stimulus_file, "--criticalRegions="+criticalRegions, "--load_from_joint="+ID]
   print(command)
   subprocess.call(command)
