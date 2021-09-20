import os
import sys
PATH = "/u/scr/mhahn/reinforce-logs-both-short/results/"
import glob



models = os.listdir(PATH)


with open("models.tsv", "w") as outFile:
  print("ID", "deletion_rate", "predictability_weight", file=outFile)
  for model in models:
    if "EYE2-T" not in model:
      continue
    if not os.path.exists("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/"+model+"_Model"):
       continue
    with open(PATH+model, "r") as inFile:
     args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ")])
     print(args["myID"], args["deletion_rate"], args["predictability_weight"], file=outFile)
