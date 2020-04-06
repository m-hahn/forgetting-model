import sys
import os

PATH = "/u/scr/mhahn/reinforce-logs-both/results/"

logs = os.listdir(PATH)
PARAMS = ["deletion_rate", "batchSize", "entropy_weight", "learning_rate_memory", "learning_rate_autoencoder", "momentum", "NUMBER_OF_REPLICATES", "lr_decay", "bilinear_l2"]
if True:
   print("###############")
   results = []
   results2 = []
   for filen in logs:
      if "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq" not in filen:
          continue
      data = open(PATH+filen, "r").read().strip().split("\n")
      if len(data) == 2:
         continue
         data.append("-1")
      if len(data) == 1:
         continue
      if len(data) == 5:
          print("Missing field", filen)
          continue
      params, perform, memRate, baselineDeviation, predictionLoss, reconstructionLoss = data
      params = params.replace("Namespace(", "")[:-1].split(", ")
      load_from_lm = [x.split("=")[1] for x in params if x.startswith("load_from_lm")][0]
      rate = float([x.split("=")[1] for x in params if x.startswith("RATE_WEIGHT")][0]) #params[0].split("=")[1])
      deletion_rate = float([x.split("=")[1] for x in params if x.startswith("deletion_rate")][0]) #params[0].split("=")[1])
      predictability_weight = float(([x.split("=")[1] for x in params if x.startswith("predictability_weight")]+[0.5])[0]) #params[0].split("=")[1])

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
      results.append((deletion_rate, predictability_weight, round(predictionLoss,4), round(reconstructionLoss, 4), " ".join(params), version, filenum, load_from_lm))
      results2.append((deletion_rate, predictability_weight, predictionLoss, reconstructionLoss, " ".join(params), version, filenum, load_from_lm) + tuple([params_here.get(x, "NA") for x in PARAMS]))

   results = sorted(results, reverse=True, key=lambda x:(x[0], x[1] * x[2] + (1-x[1]) * x[3]))
   results2 = sorted(results2, reverse=True, key=lambda x:(x[0], x[1] * x[2] + (1-x[1]) * x[3]))

   lastR = None
   with open("lm_noise/tableSearchResults_new.tsv", "w") as outFile:
     print("\t".join(["deletion_rate", "pred_weight", "predictionLoss", "prediction_loss", "params", "version", "filenum", "load_from_lm"] + PARAMS), file=outFile)
     for r in results:
      if lastR is not None and lastR[0] != r[0]:
         print("-----------")
      print("\t".join([str(x) for x in r]))
      lastR = r
     for r in results2:
      print("\t".join([str(x) for x in r]), file=outFile)
     
