from collections import defaultdict
useNext = True
import sys
ID = sys.argv[1]
with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs-gpt_THAT/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2.py_{ID}", "w") as outFile:
 print("\t".join(["Noun", "Continuation", "isThat", "isNotThat"]), file=outFile)
 with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2.py_{ID}", "r") as inFile:
   withThat = 0
   withoutThat = 0
   noun, cont = None, None
   queue = []
   for line in inFile:
      if line.startswith("Model"):
         line = line.split("\t")
         if len(line) < 3:
           continue
         sentence = line[2].strip().split(" ")
         if sentence[2] == "that":
             withThat+=1
         else:
             withoutThat += 1
         if noun is not None:
           if line[0]!=noun or line[1] != cont:
              print("\t".join([str(x) for x in [noun.split(" ")[1].strip(), cont, withThat, withoutThat]]), file=outFile)
              print(noun, cont, withThat, withoutThat)
              withThat = 0
              withoutThat = 0

         noun, cont = line[0], line[1]
              
