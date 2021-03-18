import math
import torch
surprisals = {}


with open("/john2/scr1/mhahn/memory/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_CondGPT2.py_749792590_Model.txt_SURP", "r") as inFile:
  for line in inFile:
     sentence, surpPrefix , surpNext = line.strip().split("\t")
     surprisals[sentence] = (float(surpPrefix), float(surpNext))
print(len(surprisals))

soFar = []
last = (None, None, None, None, None, None, None)
counter = 0
with open("/john2/scr1/mhahn/memory/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_CondGPT2.py_749792590_Model.txt_Reweighted", "w") as outFile:
 with open("/john2/scr1/mhahn/memory/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_CondGPT2.py_749792590_Model.txt", "r") as inFile:
   next(inFile)
   for line in inFile:
     counter += 1
     line = line.strip().split("\t")
     condition, compatible, duplicate, noun, item, region, posterior_sample, sentence, likelihoodLogSurp, posteriorLogSurp, nextWord = line
     sentence = " ".join((sentence.strip().split(" ")[1:-1] + [nextWord.strip()]))
  #   print(line)
 #    print(sentence)
     prefixSurpGPT, nextWordSurpGPT = surprisals[sentence]
     unnormalizedLogPosterior = float(likelihoodLogSurp) - prefixSurpGPT
#     print(unnormalizedLogPosterior, posteriorLogSurp)
     identifier = (condition, compatible, duplicate, posterior_sample, noun, item ,region)
  #   print(line)
   #  print(identifier)
     soFar.append((float(likelihoodLogSurp), float(posteriorLogSurp), prefixSurpGPT, nextWordSurpGPT, sentence))
     if identifier != last and last[0] is not None:
 #      print(soFar)
       likelihood = torch.FloatTensor([x[0] for x in soFar]).abs()
       posteriorLSTM = torch.FloatTensor([x[1] for x in soFar]).abs()
       prefixSurpGPT = torch.FloatTensor([x[2] for x in soFar]).abs()
       nextWordGPT = torch.FloatTensor([x[3] for x in soFar]).abs()
       log_importanceWeights = (-likelihood - prefixSurpGPT) - (- posteriorLSTM)
       largestLogImportanceWeight = log_importanceWeights.max()
       log_importanceWeights = (log_importanceWeights - largestLogImportanceWeight)
       for i in range(len(soFar)):
          if float(log_importanceWeights[i]) > -2:
            print(soFar[i][4], "\t", math.exp(float(log_importanceWeights[i])))
       reweighted = (log_importanceWeights.exp() * nextWordGPT).sum() / log_importanceWeights.exp().sum()
       print(float(nextWordGPT.mean()), "reweighted:", float(reweighted))
       forprinting = [str(q) for q  in last + ( float(nextWordGPT.mean()), float(reweighted))]
       #print(forprinting)
       forprinting = "\t".join(forprinting)
       #print(forprinting)
       print(forprinting, file=outFile)
       #quit()
       #print(identifier)
   
       soFar = []
       
      # quit()
     last = identifier
