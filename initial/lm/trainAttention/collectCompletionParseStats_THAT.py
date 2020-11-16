import os
import sys
PATH = "/u/scr/mhahn/reinforce-logs-both/full-logs-gpt_THAT/"
PATH2 = "/u/scr/mhahn/reinforce-logs-both/full-logs/"
processed = os.listdir("/u/scr/mhahn/reinforce-logs-both/full-logs-gpt_THAT/")

with open(f"parsed_output/{__file__}.txt", "w") as outFile:
  print("\t".join(["ID", "DeletionRate", "Autoencoder", "NoiseLM", "PlainLM", "PredWeight", "Noun", "Continuation", "That", "NoThat"]), file=outFile)
  for f in sorted(os.listdir(PATH)):
    ID = f[f.rfind("_")+1:]
    print(ID)
    with open(PATH2+f, "r") as inFile:
       next(inFile)
       args = next(inFile).strip()
       args = dict([x.split("=") for x in args[10:-1].split(", ")])
       print(args)
    assert ID == args["myID"], ID
    with open(PATH+f, "r") as inFile:
       next(inFile)
       for line in inFile:
         line = [x.strip() for x in line.strip().split("\t")]
         print("\t".join([ID, args["deletion_rate"], args["load_from_autoencoder"], args["load_from_lm"], args["load_from_plain_lm"], args["predictability_weight"]] + line), file=outFile)
  
