import subprocess
evScript = "autoencoder2_mlp_bidir_constructStimulusSentences_RUN_languagemodel2.py"
with open("autoencoder-output/perModelData6.tsv", "w") as outFile:
  print("\t".join(["Autoencoder", "DeletionRate"]), file=outFile)
  for autoencoder in ['264073608', '595155021', '777726352', '984542859', '346951340', '922930056', '449431785', '230092254']:
   for deletion_rate in [0.05, 0.1, 0.2, 0.3]:
       line = [autoencoder, str(deletion_rate)]
       print("\t".join(line), file=outFile)
       call = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", evScript, f"--load-from-autoencoder={autoencoder}", f"--surpsFile={evScript}_{autoencoder}_{deletion_rate}"]
       print(call)
       subprocess.run(call)
    
    
