import os
models = [x.split("\t") for x in open("autoencoder-output/searchResults.tsv", "r").read().strip().split("\n") if "autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving.py" in x]
print(models)
import subprocess
evScript = "autoencoder2_mlp_bidir_constructStimulusSentences_RUN_languagemodel2_BothOptimizedLoss_NoVs.py"
noModelFound = []
with open("autoencoder-output/perModelData8.tsv", "w") as outFile:
 print("\t".join(["Weight", "Reward", "Retention", "Parameters", "Script", "ID", "BasedOnAutoencoder"]), file=outFile)
 for line in models:

   ind = line[4].rfind("_")
   script = line[4][:ind]
   id_ = line[4][ind+1:]
   line = [line[0], line[1], line[2], line[3], line[4][:ind], line[4][ind+1:], line[5]]
   print(line)

   print("\t".join(line), file=outFile)
   print(id_)
   print(evScript)
   print(f"/u/scr/mhahn/CODEBOOKS_memoryPolicy_both/english_autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving.py_code_{id_}.txt")
   if not os.path.exists(f"/u/scr/mhahn/CODEBOOKS_memoryPolicy_both/english_autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving.py_code_{id_}.txt"): # this will happen when the model was aborted because of low retention rate
      noModelFound.append(id_)
   call = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", evScript, f"--load-from-autoencoder={id_}", f"--surpsFile={evScript}_{id_}"]
   print(call)
   subprocess.run(call)

print(noModelFound)
