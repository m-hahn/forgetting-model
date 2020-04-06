import sys
import os

PATH = "/u/scr/mhahn/reinforce-logs-predict/results/"

logs = os.listdir(PATH)
PARAMS = ["RATE_WEIGHT", "batchSize", "entropy_weight", "learning_rate", "momentum", "NUMBER_OF_REPLICATES", "lr_decay", "bilinear_l2"]
with open("lm_noise/rewards.tsv", "w") as outFile:
   print("\t".join(["rate", "version", "filenum", "counter", "reward"]), file=outFile)
   print("###############")
   results = []
   results2 = []
   for filen in logs:
      data = open(PATH+filen, "r").read().strip().split("\n")
      if len(data) == 2:
         continue
         data.append("-1")
      if len(data) == 1:
         continue
      params, perform, memRate, baselineDeviation, predictionLoss = data
      params = params.replace("Namespace(", "")[:-1].split(", ")
      load_from_lm = [x.split("=")[1] for x in params if x.startswith("load_from_lm")][0]
      rate = float([x.split("=")[1] for x in params if x.startswith("RATE_WEIGHT")][0]) #params[0].split("=")[1])

      params = [x for x in params if x.split("=")[0] in PARAMS]
      params_here = dict([x.split("=") for x in params])
      params = [x.replace("learning", "learn").replace("entropy", "ent").replace("momentum", "mom").replace("batchSize", "batch").replace("NUMBER_OF_REPLICATES","NRep").replace("lr_decay", "lrdc") for x in params]

      memRate = memRate.replace("tensor(", "").replace(", device='cuda:0', grad_fn=<MeanBackward0>)", "")
      performance = round(float(perform),4)
      memRate = round(float(memRate),4)
      baselineDeviation = round(float(baselineDeviation),4)
      predictionLoss = float(predictionLoss)
      version, filenum = (lambda x:(filen[:x], filen[x+1:]))(filen.rfind("_"))
      version = version[version.index("LastAndPos")+10:]
      if "10_c_SuperLong" in version or "10_c_Long" in version or "12" in version:
         print(version, filenum)
         with open("/u/scr/mhahn/reinforce-logs-predict/full-logs/"+"char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos"+version+"_"+filenum, "r") as inFile:
           counter = 0
           for line in inFile:
              if line.startswith("PREDICTION_LOSS"):
                _, _, _, _, reward, _ = line.split("\t")
                reward = reward.split(" ")[1]
      #          print(reward)
                counter += 1
                if counter % 100==0:
                   print("\t".join([str(x) for x in [rate, version, filenum, counter, reward]]), file=outFile)
         print(version, filenum, counter)          
