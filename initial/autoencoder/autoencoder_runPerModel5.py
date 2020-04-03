import subprocess
evScript = "autoencoder2_mlp_bidir_Erasure_constructStimulusSentences_RUN_languagemodel2_NoVs.py"
with open("autoencoder-output/perModelData5.tsv", "w") as outFile:
  print("\t".join(["Autoencoder", "DeletionRate"]), file=outFile)
  for autoencoder in ["878921872", "69900543"]:
   for deletion_rate in [0.05, 0.1, 0.2, 0.3]:
       line = [autoencoder, str(deletion_rate)]
       print("\t".join(line), file=outFile)
       call = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", evScript, f"--load-from-autoencoder={autoencoder}", f"--surpsFile={evScript}_{autoencoder}_{deletion_rate}"]
       print(call)
       subprocess.run(call)
    
    
