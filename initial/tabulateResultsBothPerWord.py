import sys
import os

PATH = "/u/scr/mhahn/reinforce-logs-both/results/"

logs = os.listdir(PATH)
PARAMS = ["RATE_WEIGHT", "batchSize", "entropy_weight", "learning_rate", "momentum", "NUMBER_OF_REPLICATES", "lr_decay", "bilinear_l2"]
with open("lm_noise/perWordRates_Both.tsv", "w") as outFile:
   print("\t".join(["rate", "version", "filenum", "counter", "word", "position", "score", "performance", "learning_rate", "entropy_weight", "memRate", "lr_decay", "dual_learning_rate", "momentum", "replicates", "predictionLoss", "reconstructionLoss"]), file=outFile)
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
      if len(data) == 5:
          continue
      params, perform, memRate, baselineDeviation, predictionLoss, reconstructionLoss = data
      params = params.replace("Namespace(", "")[:-1].split(", ")
      load_from_lm = [x.split("=")[1] for x in params if x.startswith("load_from_lm")][0]
      rate = float([x.split("=")[1] for x in params if x.startswith("RATE_WEIGHT")][0]) #params[0].split("=")[1])

      lr_decay =float([x.split("=")[1] for x in params if x.startswith("lr_decay")][0])
      dual_learning_rate = float(([x.split("=")[1] for x in params if x.startswith("dual_learning_rate")]+[-1])[0])
      momentum = float([x.split("=")[1] for x in params if x.startswith("momentum")][0])
      replicates = float(([x.split("=")[1] for x in params if x.startswith("NUMBER_OF_REPLICATES")]+[-1])[0])

      params = [x for x in params if x.split("=")[0] in PARAMS]
      params_here = dict([x.split("=") for x in params])
      params = [x.replace("learning", "learn").replace("entropy", "ent").replace("momentum", "mom").replace("batchSize", "batch").replace("NUMBER_OF_REPLICATES","NRep").replace("lr_decay", "lrdc") for x in params]

      memRate = memRate.replace("tensor(", "").replace(", device='cuda:0', grad_fn=<MeanBackward0>)", "")
      performance = round(float(perform),4)
      memRate = round(float(memRate),4)
      baselineDeviation = round(float(baselineDeviation),4)
      predictionLoss = float(predictionLoss)
      reconstructionLoss = float(reconstructionLoss)
      version, filenum = (lambda x:(filen[:x], filen[x+1:]))(filen.rfind("_"))
      script = version
      version = version[version.index("_12")+3:]
      try:
         print(version, filenum)
         with open("/u/scr/mhahn/reinforce-logs-both/full-logs/"+script+"_"+filenum, "r") as inFile:
           counter = 0
           for line in inFile:
              if line.startswith("=========="):
                   counter += 1
              elif line.startswith("SCORES") and counter % 40 == 0:
                   word, scores = line.strip().split("\t")
                   word = word[7:]
                   scores = scores.strip().split(" ")
                   for j in range(21):
                      print("\t".join([str(x) for x in [rate, version, filenum, counter, word, j, scores[j], perform, params_here.get("learning_rate_memory", params_here.get("learning_rate")), params_here["entropy_weight"], memRate, lr_decay, dual_learning_rate, momentum, replicates, predictionLoss, reconstructionLoss]]), file=outFile)
      except FileNotFoundError:
         print("Didn't find file")             
