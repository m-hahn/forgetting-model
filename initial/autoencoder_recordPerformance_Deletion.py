script = "autoencoder2_mlp_bidir.py"
import os

files = [x for x in os.listdir("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/") if script in x]
print([x.split("_")[-3] for x in files])
quit()

