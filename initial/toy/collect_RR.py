import os
import sys

files = os.listdir("output_rr/")

for f in files:
  if f.startswith("resourceRational5.py_"):
     with open("output_rr/"+f, "r") as inFile:
       header = next(inFile) .strip().split("\t")
       header = dict(list(zip(header, range(len(header)))))
       results = []
       for _ in range(4):
         line = next(inFile).strip().split("\t")
         noun = line[header["Noun"]]
         condition = line[header["Condition"]]
         surprisal = line[header["SurprisalReweighted"]]
         results.append(("High" if "fact" in noun else "Low", "One" if "NoSC" in condition else "Two", float(surprisal)))
       next(inFile)
       next(inFile)
       next(inFile)
       next(inFile)
       next(inFile)
       next(inFile)
       args = next(inFile)
       args = args[args.index("deletion_rate"):]
       args = float(args[args.index("=")+1:args.index(",")])
       for n, c, s in results:
         print(args, n, c, s)
