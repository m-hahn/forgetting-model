import os

header = "Noun Region Condition Surprisal ThatFraction".split(" ")
header += ["ID", "predictability_weight", "deletion_rate", "autoencoder", "lm"]

PATH = "/juice/scr/mhahn/reinforce-logs-both/full-logs/"
PATH2 = "/juice/scr/mhahn/reinforce-logs-both/full-logs-tsv/"

with open(f"raw_output/{__file__}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for f in os.listdir(PATH):
   if "12_NormJudg_Shift_NoComma_Bugfix.py" in f:
      accept = False
      with open(PATH+f, "r") as inFile:
         iterations = next(inFile).strip()
         arguments = next(inFile).strip()
         for line in inFile:
             if "THAT" in line:
                if "fixed" in line:
                     accept = True
                     break
      if accept:
          arguments = dict([x.split("=") for x in arguments[10:-1].split(", ")])
          print(arguments)
          print(f)
          predictability_weight = arguments["predictability_weight"]
          deletion_rate = arguments["deletion_rate"]
          with open(PATH2+f+"_Model", "r") as inFile:
             data = [x.split(" ") for x in inFile.read().strip().split("\n")]
             data = data[1:]
             for line in data:
                 print("\t".join(line + [arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], arguments["load_from_autoencoder"], arguments["load_from_plain_lm"]]), file=outFile)
