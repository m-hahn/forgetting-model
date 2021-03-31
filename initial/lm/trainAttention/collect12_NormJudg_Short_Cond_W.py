import os

header = "Noun Region Condition Surprisal SurprisalReweighted ThatFraction ThatFractionReweighted".split(" ")
header += ["Script", "ID", "predictability_weight", "deletion_rate", "autoencoder", "lm"]

PATH = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs/"
PATH2 = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv/"

with open(f"raw_output/{__file__}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for f in os.listdir(PATH):
   shib = "12_NormJudg_Short_Cond_Shift_NoComma_Bugfix"
   if shib in f:
      suffix = "script_"+f[f.index(shib)+len(shib):f.index(".py")]
      if "_W" not in suffix or "GPT2" in suffix:
        continue
      print(suffix)
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
          try:
           with open(PATH2+f+"_Model", "r") as inFile:
             data = [x.split(" ") for x in inFile.read().strip().split("\n")]
             data = data[1:]
             for line in data:
                 print("\t".join(line + [suffix, arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], arguments["load_from_autoencoder"], arguments["load_from_plain_lm"]]), file=outFile)
          except FileNotFoundError:
             pass
