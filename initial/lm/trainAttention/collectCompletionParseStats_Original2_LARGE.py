import os
import sys
PATH = "/u/scr/mhahn/reinforce-logs-both/full-logs-gpt2-withOrig2/" # parsed results
PATH2 = "/u/scr/mhahn/reinforce-logs-both/full-logs/"

filesPATH2 = os.listdir(PATH2)

with open(f"parsed_output/{__file__}.txt", "w") as outFile:
  print("\t".join(["ID", "DeletionRate", "Autoencoder", "NoiseLM", "PlainLM", "PredWeight", "Noun", "Continuation", "complete", "incomplete", "That", "NoThat"]), file=outFile)
  for f in sorted(os.listdir(PATH)):
    if "LARGE" not in f:
      continue
    ID = f[f.rfind("_")+1:]
    print(ID)
    relevant = [x for x in filesPATH2 if x.endswith("_"+ID)]
    assert len(relevant) >= 1, relevant
    with open(PATH2+relevant[0], "r") as inFile:
       next(inFile)
       args = next(inFile).strip()
       args = dict([x.split("=") for x in args[10:-1].split(", ")])
       print(args)
    assert ID == args["myID"], ID
    with open(PATH+f, "r") as inFile:
       try:
         next(inFile)
       except StopIteration:
           continue
       for line in inFile:
         line = [x.strip() for x in line.strip().split("\t")]
         print("\t".join([ID, args["deletion_rate"], args["load_from_autoencoder"], args["load_from_lm"], args["load_from_plain_lm"], args["predictability_weight"]] + line), file=outFile)
  
