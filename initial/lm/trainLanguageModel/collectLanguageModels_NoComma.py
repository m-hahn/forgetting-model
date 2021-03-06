import glob
import sys
results = []
language = sys.argv[1]
script = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_NoComma.py"
files = glob.glob("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+language+"_"+script+"_model_*_*.txt")
for f in files:
  with open(f, "r") as inFile:
     data = inFile.read().strip().split("\n")
     args = dict([x.split("=") for x in data[0][10:-1].split(", ")])
     devLosses = [float(x) for x in data[1].strip().split(" ")]
     num_iter = len(devLosses)
     try:
       load_from = args.get("load_from", "None")
     except KeyError:
       print("ERROR", args)
       continue
     last_loss = devLosses[-1]
     hasEnded = len(devLosses) > 1 and devLosses[-2] < devLosses[-1]
     myID = args["myID"]
     results.append((last_loss, num_iter, hasEnded, myID, load_from))
for r in sorted(results, reverse=True):
  print(r)

