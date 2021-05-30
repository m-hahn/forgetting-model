import sys
import os
from collections import defaultdict
for f in os.listdir("output/"):
  with open("output/"+f, "r") as inFile:
     args = dict([x.split(" ") for x in next(inFile).strip().split("\t")])
     mi = next(inFile)
     surp = next(inFile)
     surprisals = defaultdict(list)
     for line in inFile:
       line = line.strip()
       past, nextWord, prob = line.split("\t")
       past = past.strip().split(" ")
       if past[0] in ["report", "story", "admission"]:
          condition = ("report",)
       else:
          condition = ("fact",)
       if len(past) == 1:
          condition = condition + ("One",)
       elif len(past) == 5:
          condition = condition + ("Two",)
       else:
          assert False, past
       surprisals[condition].append(float(prob))
     for cond in sorted(list(surprisals)):
        print(args["beta"], " ".join(cond), -sum(surprisals[cond]) / len(surprisals[cond]), f)
