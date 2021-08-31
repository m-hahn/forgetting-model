import glob
import sys

stimulus_file = sys.argv[1]
results = glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/*_{stimulus_file}_*_Model")

header = ["Sentence", "Item", "Condition", "Region", "Word", "Surprisal", "SurprisalReweighted", "Repetition", "Script", "ID", "deletion_rate", "predictability_weight"]
with open(f"/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/{__file__}_{stimulus_file}.tsv", "w") as outFile:
 print("\t".join(header), file=outFile)
 for model in sorted(results):
   script = model.split("/")[-1]
   script = script[:script.rfind("_")]
   script = script[:script.rfind("_")]
   script = script[:script.rfind("_")]
   assert script.endswith(".py"), script

   print(model)
   ID = model.replace("_Model", "")
   ID = ID[ID.rfind("_")+1:]
   with open(glob.glob(f"/u/scr/mhahn/reinforce-logs-both-short/results/*_{ID}")[0], "r") as inFile:
     args = dict([x.split("=") for x in next(inFile).replace("Namespace(", "").rstrip(")").split(", ")])
   assert ID == args["myID"]
   with open(model, "r") as inFile:
      try:
         next(inFile)
      except StopIteration:
         continue
      for line in inFile:
        print("\t".join(line.rstrip("\n").split("\t") + [script[-10:], ID, args["deletion_rate"], args["predictability_weight"]]), file=outFile)
      

