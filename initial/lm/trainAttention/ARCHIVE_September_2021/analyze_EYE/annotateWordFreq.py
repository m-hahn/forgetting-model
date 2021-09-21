import gzip
words = set()

with open("matchData_EYE.py.tsv", "r") as inFile:
   for line in inFile:
      line = line.strip().split("\t")
      if len(line) > 4:
          words.add(line[2].lower())
print(words)
from collections import defaultdict
frequencies = defaultdict(int)
count = 0
# http://kilgarriff.co.uk/bnc-readme.html
with gzip.open("/juice/scr/mhahn/CORPORA/BNC_FREQUENCY/all.num.gz", "rt") as inFile:
   for line in inFile:
       count += 1
       if count % 100000 == 0:
          print(count, len(words)-len(frequencies))
       line = line.split(" ")
       if len(line) >= 4:
          if line[1] in words:
            frequencies[line[1]] += int(line[3])
print([x for x in words if x not in frequencies])
with open("dundee-bnc-frequencies.tsv", "w") as outFile:
 print("\t".join(["LowerCaseToken", "Frequency"]), file=outFile)
 for x, y in frequencies.items(): 
   print("\t".join([x, str(y)]), file=outFile)
