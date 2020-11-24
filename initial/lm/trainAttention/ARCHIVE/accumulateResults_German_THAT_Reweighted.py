import glob
files = glob.glob("/u/scr/mhahn/reinforce-logs-both/full-logs/har*")
results = []
for f in files:
   if "erman" not in f:
     continue
   if "Reweight" not in f:
      continue
   id_ = f[f.rfind("_")+1:]
   with open(f, "r") as inFile:
       data = inFile.read().split("\n")
       iterations = data[0]
       if data[1].startswith("Namespace"):
              arguments = data[1]
       else:
              arguments = None
       correlations = [x[24:].split(") ") for x in data if x.startswith("THAT_correlation ")]
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
       results.append({"iterations" : iterations, "pred_weight" : pred_weight, "del_rate" : del_rate, "correlations" : correlations, "id" : id_, "lr_mem" : learning_rate_memory, "mom" : momentum, "args" : arguments})
with open("output/results_german_THAT_reweighted.tsv", "w") as outFile2:
 print("\t".join(["pred_weight", "del_rate", "model1", "model2", "sanity1", "sanity2", "denoiser", "noised_lm", "lm"]), file=outFile2)
 with open("output/results_german.txt", "w") as outFile:
  for r in sorted(results, key=lambda x:x["correlations"].get("Model 2", 0.0), reverse=True):
    print(r)
    print(r, file=outFile)
    print("\t".join([str(x) for x in [r["pred_weight"], r["del_rate"], r["correlations"].get("Model 1", "NA"), r["correlations"].get("Model 2", "NA"), r["correlations"].get("Sanity 1", "NA"), r["correlations"].get("Sanity 2", "NA"), r["args"]["load_from_autoencoder"], r["args"]["load_from_lm"], r["args"]["load_from_plain_lm"]]]), file=outFile2)


