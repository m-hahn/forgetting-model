import glob
import subprocess

candidates = glob.glob("/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2.py_*")
IDs = [x[x.rfind("_")+1:] for x in candidates]
import os
for ID in IDs:
  if not os.path.isfile("/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2_LARGE.py_"+ID):
      subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py36-mhahn/bin/python", "runGPT2Large.py", ID])


