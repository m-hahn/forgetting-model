import random
import subprocess
scripts = []

scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_GPT2L_OnlyLoc.py")
scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_GPT2M_OnlyLoc.py")

import sys
if len(sys.argv) > 2:
    ITER = int(sys.argv[2])
else:
    ITER=100

for i in range(ITER): # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 
# 0.35, 0.4, 0.45, 0.5, 0.55, 
    # 0.15, 0.2, 0.25, 0.3, 0.35 , 
   deletion_rate = str(random.choice([0.45, 0.5, 0.55, 0.6]))  # , 0.6, 0.7    #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]))
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", random.choice(scripts), "--tuning=1", "--deletion_rate="+deletion_rate, "--predictability_weight="+str(random.choice([ 0.5])), "--learning_rate_memory="+str(random.choice([0.00002, 0.00005, 0.0001, 0.0001, 0.0001])), "--learning_rate_autoencoder="+str(random.choice([0.001, 0.01, 0.1, 0.1, 0.1, 0.1, 0.2])), "--momentum="+str(random.choice([0.5, 0.7, 0.7, 0.7, 0.7, 0.9]))] #, "--predictability_weight=0.0"]
   print(command)
   subprocess.call(command)



