import sys
import os
from collections import defaultdict
bestValueByBeta = {}
resultsByBeta = {}
for f in os.listdir("output/"):
  with open("output/"+f, "r") as inFile:
     args = dict([x.split(" ") for x in next(inFile).strip().split("\t")])
     mi = next(inFile)
     surp = next(inFile)
     mi = float(mi.strip().split(" ")[-1])
     surp = float(surp.strip().split(" ")[-1])
     beta = float(args["beta"])
     objective = beta * mi + surp
     #print(mi, surp, beta, objective)
     if beta in bestValueByBeta:
       if objective < bestValueByBeta[beta]:
           bestValueByBeta[beta] = objective
       else:
           continue
     resultsByBeta[beta] = []
     surprisals = defaultdict(list)
     for line in inFile:
       line = line.strip()
       past, nextWord, prob = line.split("\t")
       past = past.strip().split(" ")
       if past[0] in ["report", "story", "admission"]:
          condition = ("Low",)
       else:
          condition = ("High",)
       if len(past) == 1:
          condition = condition + ("One",)
       elif len(past) == 5:
          condition = condition + ("Two",)
       else:
          assert False, past
       surprisals[condition].append(float(prob))
     for cond in sorted(list(surprisals)):
        resultsByBeta[beta].append(" ".join([str(q) for q in [args["beta"], " ".join(cond), -sum(surprisals[cond]) / len(surprisals[cond]), f]]))
for beta, ls in resultsByBeta.items():
  for l in ls:
        print(l)
