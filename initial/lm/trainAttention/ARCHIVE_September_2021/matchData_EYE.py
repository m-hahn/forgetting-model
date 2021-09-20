import os
import sys
import random
from collections import defaultdict
with open("/u/scr/mhahn/Dundee/DundeeTreebankTokenized.csv", "r") as inFile:
   dundee = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = dundee[0]
   header = dict(zip(header, list(range(len(header)))))
   dundee = dundee[1:]


calibrationSentences = []

for i in range(len(dundee)):
    line = dundee[i]
    Itemno, WNUM, SentenceID, ID, WORD, Token = line
    SentenceID = (SentenceID)
    if i == 0 or SentenceID != dundee[i-1][header["SentenceID"]]:
        calibrationSentences.append([])
        print(SentenceID, dundee[i-1][header["SentenceID"]])

    if i > 0 and SentenceID == dundee[i-1][header["SentenceID"]] and ID == dundee[i-1][header["ID"]]:
        continue
    else:
        calibrationSentences[-1].append((WORD.strip(".").strip(",").strip("?").strip(":").strip(";").replace("’", "'").strip("!").lower(), line))
#    else:
         
if True:
    numberOfSamples = 12
    with open("analyze_EYE/"+__file__+".tsv", "w") as outFile:
#              print("\t".join([str(w) for w in [sentenceID, regions[i], remainingInput[i][0]] + remainingInput[i][1]   ]), file=outFile) #, file=outFile)
#    Itemno, WNUM, SentenceID, ID, WORD, Token = line
      print("\t".join(["Sentence", "Region", "Word", "Itemno", "WNUM", "SentenceID", "ID", "WORD", "Token"]), file=outFile)
      for sentenceID in range(len(calibrationSentences)):
          sentence = calibrationSentences[sentenceID] #.lower().replace(".", "").replace(",", "").replace("n't", " n't").split(" ")
          print(sentence)
          context = sentence[0]
          remainingInput = sentence[1:]
          regions = range(len(sentence))
          print("INPUT", context, remainingInput)
          if len(sentence) < 2:
             continue
          assert len(remainingInput) > 0
          for i in range(len(remainingInput)):
              print("\t".join([str(w) for w in [sentenceID, regions[i], remainingInput[i][0]] + remainingInput[i][1]   ]), file=outFile) #, file=outFile)

