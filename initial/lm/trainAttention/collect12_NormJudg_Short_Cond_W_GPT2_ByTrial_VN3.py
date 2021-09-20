import codecs
import os

header = "Noun Item Region Condition Surprisal SurprisalReweighted ThatFraction ThatFractionReweighted".split(" ")
header += ["S1", "S2", "Word", "Script", "ID", "predictability_weight", "deletion_rate", "autoencoder", "lm"]

PATH = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs/"
PATH2 = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/"

with open(f"{PATH2}/{__file__}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for f in os.listdir(PATH):
   shib = "12_NormJudg_Short_Cond_Shift_NoComma_Bugfix"
   if shib in f:
      suffix = "script_"+f[f.index(shib)+len(shib):f.index(".py")]
      if "_VN3Stims_" not in f or "_OnlyLoc" in f or "ZERO" in f or "_LE" in f:
        continue
      print(f)
      print(suffix)
#      accept = False
      with codecs.open(PATH+f, "r", 'utf-8', "ignore") as inFile:
         try:
           iterations = next(inFile).strip()
           arguments = next(inFile).strip()
         except StopIteration:
           continue
#         for line in inFile:
#             if "THAT" in line:
#                if "fixed" in line:
#                     accept = True
#                     break
 #     print(accept)
      if True or accept:
          try:
            arguments = dict([x.split("=") for x in arguments[10:-1].split(", ")])
          except ValueError:
            print("VALUE ERROR", arguments)
            continue
          print(arguments)
          print(f)
          predictability_weight = arguments["predictability_weight"]
          deletion_rate = arguments["deletion_rate"]
          try:
           with open(PATH2+f+"_Model", "r") as inFile:
             print("Opened", PATH2+f+"_Model")
             data = [x.split("\t") for x in inFile.read().strip().split("\n")]
             data = data[1:]
             for line in data:
                 if len(line) == 10:
                    print("WARNING: COLUMN MISSING!!!", line)
                    _ = float(line[-1]) # Make sure the last entry is a number, as a basic sanity check
                    line.append("NA")
                 assert len(line) == 11, line
                 print("\t".join(line + [suffix, arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], arguments["load_from_autoencoder"], arguments["load_from_plain_lm"]]), file=outFile)
          except FileNotFoundError:
             print("Couldn't open", PATH2+f+"_Model")
             pass
