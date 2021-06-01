import sys
import os

PATH = "/u/scr/mhahn/reinforce-logs-both-short/results/"

logs = sorted(os.listdir(PATH))

for lossType in ["Erasure"]:
  with open(f"tradeoff/{__file__}_{lossType}.tsv", "w") as outFile:
   print("\t".join(["deletion_rate", "predictability_weight", "reward", "predictability", "reconstructability", "args", "suffix", "autoencoder"]), file=outFile)
   print("###############")
   results = []
   for filen in logs:
#      if "VNStims_3_W_GPT2" not in filen:
      if "NoComma_Bugfix" not in filen:
        continue
      print(filen)
      data = open(PATH+filen, "r").read().strip().split("\n")
      if len(data) == 2:
         continue
         data.append("-1")
      if len(data) == 1:
         continue
#      print(data)
      params, perform, memRate, _, predLoss, recLoss = data
      params = params.replace("Namespace(", "")[:-1].split(", ")
      load_from_autoencoder = [x.split("=")[1] for x in params if x.startswith("load_from_autoencoder")][0]
      params = [x for x in params if x.split("=")[0] in ["deletion_rate", "learning_rate_memory", "learning_rate_autoencoder", "dual_learning_rate", "momentum", "predictability_weight"]]
      params = [x.replace("learning", "learn").replace("entropy", "ent").replace("momentum", "mom").replace("batchSize", "batch") for x in params]
      memRate = float([x for x in params if x.startswith("deletion_rate")][0].split("=")[1]) #memRate.replace("tensor(", "").replace(", device='cuda:0', grad_fn=<MeanBackward0>)", "")
      predWeight = float([x for x in params if x.startswith("predictability_weight")][0].split("=")[1]) #memRate.replace("tensor(", "").replace(", device='cuda:0', grad_fn=<MeanBackward0>)", "")
      params = [x for x in params if x[:x.index("=")] not in ["deletion_rate", "predictability_weight"]]
      performance = round(float(perform),4)
      memRate = round(float(memRate),4)
      results.append(((memRate, predWeight), performance, predLoss, recLoss, " ".join(params), filen[-25:], load_from_autoencoder))
      print("\t".join([str(q) for q in [memRate, predWeight, performance, predLoss, recLoss, ";".join(params), filen[-25:], load_from_autoencoder]]), file=outFile)
   results = sorted(results, reverse=True)
   lastR = None
   for r in results:
      if lastR is not None and lastR[0] != r[0]:
         print("-----------")
      print("\t".join([str(x) for x in r]))
      lastR = r
