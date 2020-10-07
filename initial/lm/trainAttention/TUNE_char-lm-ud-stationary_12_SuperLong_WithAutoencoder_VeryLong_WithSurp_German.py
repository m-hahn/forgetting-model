import random
import subprocess
scripts = []

#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp4.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp5.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp6.py")

#scripts.append("har-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp7_German_New_Reweighting.py")
scripts.append("har-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp7_German_New_Reweighting_Multiple.py")
scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp7_German_New.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp7_German_New_WithComma.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp7_German_New_WithComma_NoLoss.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp7_German_New_NoLoss.py")

for i in range(100): # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 
# 0.35, 0.4, 0.45, 0.5, 0.55, 
   deletion_rate = str(random.choice([0.55, 0.6, 0.65, 0.7, 0.75])) #, 0.25, 0.3, 0.35])) # , 0.4, 0.45, 0.5, 0.55      #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]))
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", random.choice(scripts), "--tuning=1", "--deletion_rate="+deletion_rate] #, "--predictability_weight=0.0"]
   print(command)
   subprocess.call(command)
