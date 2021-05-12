import codecs
import os

header = ["Script", "ID", "predictability_weight", "deletion_rate", "Word", "Distance", "AvailableRate"]

PATH = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs/"
PATH2 = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv/"

with open(f"raw_output/{__file__}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for f in os.listdir(PATH):
   shib = "12_NormJudg_Short_Cond_Shift_NoComma_Bugfix"
   if shib in f:
      suffix = "script_"+f[f.index(shib)+len(shib):f.index(".py")]
      if "_W" not in suffix or "GPT2" not in suffix or "_QC_" not in f:
        continue
      print(suffix)
      accept = False
      with codecs.open(PATH+f, "r", 'utf-8', "ignore") as inFile:
         try:
            iterations = next(inFile).strip()
            arguments = next(inFile).strip()
            _ = next(inFile)
            _ = next(inFile)
            scores = next(inFile).strip()
            arguments = dict([x.split("=") for x in arguments[10:-1].split(", ")])
            while scores.startswith("SCORES"):
                 scores = scores.split("\t")
                 word = scores[0].split(" ")[1]
                 attention = scores[1].strip().split(" ")
                 attention = [float(x) for x in attention]
                 print(word, attention)
                 for i in range(len(attention)):
                    print("\t".join([str(w) for w  in [suffix, arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], i, word, attention[i]]]), file=outFile)
   
                 scores = next(inFile).strip()
         except StopIteration:
           continue
#         for line in inFile:
#             print(line)
#             if "THAT" in line:
#                if "fixed" in line:
#                     accept = True
#                     break
#      if True or accept:
#          print(arguments)
#          print(f)
#          predictability_weight = arguments["predictability_weight"]
#          deletion_rate = arguments["deletion_rate"]
#          try:
#           with open(PATH2+f+"_Model", "r") as inFile:
#             print("Opened", PATH2+f+"_Model")
#             data = [x.split(" ") for x in inFile.read().strip().split("\n")]
#             data = data[1:]
#             for line in data:
#                 print("\t".join(line + [suffix, arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], arguments["load_from_autoencoder"], arguments["load_from_plain_lm"]]), file=outFile)
#          except FileNotFoundError:
#             print("Couldn't open", PATH2+f+"_Model")
#             pass
