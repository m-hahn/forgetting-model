import glob
import sys
results = []
language = sys.argv[1]
script = "autoencoder2_mlp_bidir_Erasure_SelectiveLoss_WithoutComma.py"
files = glob.glob("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+language+"_"+script+"_model_*_*.txt")
for f in files:
  with open(f, "r") as inFile:
     data = inFile.read().strip().split("\n")
     args = dict([x.split("=") for x in data[0][10:-1].split(", ")])
     devLosses = [float(x) for x in data[1].strip().split(" ")]
     num_iter = len(devLosses)
     load_from = args["load_from"]
     last_loss = devLosses[-1]
     hasEnded = len(devLosses) > 1 and devLosses[-2] < devLosses[-1]
     myID = args["myID"]
     results.append((last_loss, num_iter, hasEnded, myID, load_from))
for r in sorted(results, reverse=True):
  print(r)

