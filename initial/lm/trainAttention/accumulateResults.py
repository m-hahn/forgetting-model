import glob
files = glob.glob("/u/scr/mhahn/reinforce-logs-both/full-logs/char*")
results = []
for f in files:
   id_ = f[f.rfind("_")+1:]
   with open(f, "r") as inFile:
       data = inFile.read().split("\n")
       iterations = data[0]
       if data[1].startswith("Namespace"):
              arguments = data[1]
       else:
              arguments = None
       correlations = [x[28:].split(") ") for x in data if x.startswith("PLAIN LM Correlation ")]
       try:
          correlations = dict((x[1], float(x[0])) for x in correlations)
       except IndexError:
          print(correlations)
          continue
       if "Model 1" not in correlations:
         continue
       print(iterations)
       print(arguments)
       print(correlations)
       if arguments is None:
          continue
       arguments = dict(x.split("=") for x in arguments[10:-1].split(", "))
       print(arguments)
       pred_weight = float(arguments["predictability_weight"])
       del_rate = float(arguments["deletion_rate"])
       learning_rate_memory = float(arguments["learning_rate_memory"])
       momentum = float(arguments["momentum"])
       results.append({"iterations" : iterations, "pred_weight" : pred_weight, "del_rate" : del_rate, "correlations" : correlations, "id" : id_, "lr_mem" : learning_rate_memory, "mom" : momentum})
with open("output/results.tsv", "w") as outFile2:
 print("\t".join(["pred_weight", "del_rate", "model2"]), file=outFile2)
 with open("output/results.txt", "w") as outFile:
  for r in sorted(results, key=lambda x:x["correlations"].get("Model 2", 0.0), reverse=True):
    print(r)
    print(r, file=outFile)
    print("\t".join([str(x) for x in [r["pred_weight"], r["del_rate"], r["correlations"].get("Model 2", 0.0)]]), file=outFile2)


