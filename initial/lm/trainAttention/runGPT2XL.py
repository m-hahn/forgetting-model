#ID = "960136486"
import sys
ID = sys.argv[1]
path = "/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2.py_"+ID
pathOut = "/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2_XL.py_"+ID


import sampleNextWordFromGPT2_XL as sampleNextWordFromGPT2

sampleNextWordFromGPT2.prepareModel()

from collections import defaultdict
cache = set()
countByTriple = defaultdict(int)
with open(pathOut, "w") as outFile:
 with open(path, "r") as inFile:
   print(next(inFile).strip(), file=outFile)
   print(next(inFile).strip(), file=outFile)
   for line in inFile:
     if line.startswith("Model"):
        line = line.split("\t")
        if len(line) == 3:
          noun, continuation, sampled_old = line
          sampled_old = " ".join(sampled_old.strip().split(" ")[:8])
          if "OOV" in sampled_old:
             continue
          countByTriple[(noun, continuation, sampled_old)] += 1
          print(len(countByTriple))
 triples = 0
 for noun, continuation, sampled_old in countByTriple:
          triples+=1
          count = countByTriple[(noun, continuation, sampled_old)]
          if count < 10:
             continue
          print((noun, continuation, sampled_old, count, triples/len(countByTriple)))
        
          nextWords = sampleNextWordFromGPT2.sample(sampled_old)
          for sampled in nextWords:
             if "\n" in sampled:
               sampled = sampled[:sampled.index("\n")]
             print(noun,"\t", continuation,"\t", count,  "\t", sampled, file=outFile)


