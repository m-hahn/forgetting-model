import subprocess
import glob

header = set()
data = {}
files = glob.glob("/u/scr/mhahn/CODEBOOKS_memoryPolicy_both/*Punct*")
script = "autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving_Lagrange_BoundarySymbol_NoPunct.py"
with open(f"output/{__file__}.tsv", "w") as outFile:
 print("\t".join(["ID", "deletion_rate", "entropy_weight", "learning_rate_memory", "load_from_autoencoder"]), file=outFile)
 for filename in files:
   ID = filename[filename.rfind("_")+1:-4]

   with open("/u/scr/mhahn/reinforce-logs/results/"+script+"_"+ID, "r") as inFile:
      args = dict([x.split("=") for x in next(inFile).strip().replace("Namespace(", "").rstrip(")").split(", ")])
   print("\t".join([ID, args["deletion_rate"], args["entropy_weight"], args["learning_rate_memory"], args["load_from_autoencoder"]]), file=outFile)
   if len(glob.glob("/u/scr/mhahn/noisy-channel-logs/deletion-gibson/autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving_Lagrange_BoundarySymbol_NoPunct_GetSamples3_W.py_"+ID)) == 0:
        subprocess.call(["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", "autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving_Lagrange_BoundarySymbol_NoPunct_GetSamples3_W.py", "--load-from-joint="+ID])
   with open("/u/scr/mhahn/noisy-channel-logs/deletion-gibson/autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving_Lagrange_BoundarySymbol_NoPunct_GetSamples3_W.py_"+ID, "r") as inFile:
     dataf = [x.split("\t") for x in inFile.read().strip().split("\n")]
     header_here = dataf[0]
     header = header.union(set(header_here))
     data[ID] = (header_here, dataf[1:])
with open(f"output/{__file__}-joint.tsv", "w") as outFile:
  header_total = ["ID"] + sorted(list(header))
  print("\t".join(header_total), file=outFile)
  for ID in data:
    header_here, data_here = data[ID]
    for line in data_here:
      print("\t".join([ID] + [line[header_here.index(c)] if c in header_here else "0" for c in header_total[1:]]), file=outFile)
