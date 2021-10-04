import glob
import codecs
import os

header0 = "Sentence Region Word Surprisal SurprisalReweighted Copy".split(" ")
header =  header0 + ["Script", "ID", "predictability_weight", "deletion_rate", "autoencoder", "lm"]

PATH = "/juice/scr/mhahn/reinforce-logs-both-short/full-logs/"
PATH2 = "/juice/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/"

with open(f"{PATH2}/{__file__}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for f in os.listdir(PATH2):
   shib = "12_NormJudg_Short_Cond_Shift_NoComma_Bugfix"
   if shib in f:
      suffix = "script_"+f[f.index(shib)+len(shib):f.index(".py")]
      if "_VN3Stims_" not in f or "_OnlyLoc" in f or "ZERO" in f:
        continue
 #     print(f)
      #print(suffix)
      accept = False
 #     print(f)
#      quit()
      ID = f.replace("_Model", "").split("_")[-1]
      logPath = PATH+"/*_"+ID
      relevantFile = glob.glob(logPath)
      if len(relevantFile) == 0:
        print("NO LOG FOUND", f)
        continue
      with codecs.open(relevantFile[0], "r", 'utf-8', "ignore") as inFile:
         try:
           iterations = next(inFile).strip()
           arguments = next(inFile).strip()
         except StopIteration:
           print("CANNOT FIND ARGUMENTS", f)
           continue
#         for line in inFile:
#             if "THAT" in line:
#                if "fixed" in line:
#                     accept = True
#                     break
      #print(accept)
      if True or accept:
          arguments = dict([x.split("=") for x in arguments[10:-1].split(", ")])
          #print(arguments)
          print("ACCEPTING", f)
          predictability_weight = arguments["predictability_weight"]
          deletion_rate = arguments["deletion_rate"]
          try:
           logPath = PATH2+"/*_"+ID+"_Model"
           relevantFile = glob.glob(logPath)
           if len(relevantFile) == 0:
             print("FAILED TO OPEN", f)
             continue
           with open(relevantFile[0], "r") as inFile:
             data = [x.split("\t") for x in inFile.read().strip().split("\n")]
             data = data[1:]
             for line in data:
                 if len(line) < len(header0):
                     line.append("NA")
                 print("\t".join(line + [suffix, arguments["myID"], arguments["predictability_weight"], arguments["deletion_rate"], arguments["load_from_autoencoder"], arguments["load_from_plain_lm"]]), file=outFile)
          except FileNotFoundError:
             print("FAILED TO OPEN", f)
             pass
