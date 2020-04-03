models = [x.split("\t") for x in open("autoencoder-output/searchResults.tsv", "r").read().strip().split("\n") if "autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning" in x and "Saving" in x and "Both" not in x]
print(models)
import subprocess
evScript = "autoencoder2_mlp_bidir_Erasure_constructStimulusSentences_RUN_languagemodel2_OptimizedLoss.py"
with open("autoencoder-output/perModelData3.tsv", "w") as outFile:
 print("\t".join(["Weight", "Reward", "Retention", "Parameters", "Script", "ID"]), file=outFile)
 for line in models:
   ind = line[4].rfind("_")
   script = line[4][:ind]
   id_ = line[4][ind+1:]
   line = [line[0], line[1], line[2], line[3], line[4][:ind], line[4][ind+1:]]
   print("\t".join(line), file=outFile)
   print(id_)
   print(evScript)
   call = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", evScript, f"--load-from-memory={id_}", f"--surpsFile={evScript}_{id_}"]
   print(call)
   subprocess.run(call)


