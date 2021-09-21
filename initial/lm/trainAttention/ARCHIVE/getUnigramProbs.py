
import sys
import random

import math

countsTotal = 0
counts = {}

with open("/u/scr/mhahn/FAIR18/english-wiki-word-vocab.txt", "r") as inFile:
   for line in inFile:
      word, count = line.strip().split("\t")
      if len(counts) < 50000:
         counts[word] = int(count)
      countsTotal += int(count)


nounsAndVerbs = []
nounsAndVerbs.append(["the principal",       "the teacher",        "kissed",      "was fired",                     "was quoted in the newspaper", "Was the XXXX quoted in the newspaper?", "Y"])
nounsAndVerbs.append(["the sculptor",        "the painter",        "admired",    "was n't talented",   "was completely untrue", "Was the XXXX untrue?", "Y"])
nounsAndVerbs.append(["the consultant",      "the artist",         "hired",      "was a fraud",       "shocked everyone", "Did the XXXX shock everyone?", "Y"])
nounsAndVerbs.append(["the runner",          "the psychiatrist",   "treated",    "was doping",        "was ridiculous", "Was the XXXX ridiculous?", "Y"])
nounsAndVerbs.append(["the child",           "the medic",          "rescued",    "was unharmed",      "relieved everyone", "Did the XXXX relieve everyone?", "Y"])
nounsAndVerbs.append(["the criminal",        "the officer",        "arrested",   "was guilty",        "was entirely bogus", "Was the XXXX bogus?", "Y"])
nounsAndVerbs.append(["the student",         "the professor",      "hated",      "dropped out",       "made the professor happy", "Did the XXXX make the professor happy?", "Y"])
nounsAndVerbs.append(["the mobster",         "the media",          "portrayed",  "had disappeared",    "turned out to be true", "Did the XXXX turn out to be true?", "Y"])
nounsAndVerbs.append(["the actor",           "the starlet",        "loved",      "was missing",       "made her cry", "Did the XXXX almost make her cry?", "Y"])
nounsAndVerbs.append(["the preacher",        "the parishioners",   "fired",      "stole money",        "proved to be true", "Did the XXXX prove to be true?", "Y"])
nounsAndVerbs.append(["the violinist",       "the sponsors",       "backed",     "abused drugs",                       "is likely true", "Was the XXXX likely true?", "Y"])
nounsAndVerbs.append(["the senator",         "the diplomat",       "opposed",    "was winning",                   "really made him angry", "Did the XXXX make him angry?", "Y"])
nounsAndVerbs.append(["the commander",       "the president",      "appointed",  "was corrupt",         "troubled people", "Did the XXXX trouble people?", "Y"])
nounsAndVerbs.append(["the victims",         "the criminal",       "assaulted",  "were surviving",         "calmed everyone down", "Did the XXXX calm everyone down?", "Y"])
#nounsAndVerbs.append(["the politician",      "the banker",         "bribed",     "laundered money",         "came as a shock to his supporters", "Did the XXXX come as a shock?", "Y"])
# Issue: laundered is OOV
nounsAndVerbs.append(["the surgeon",         "the patient",        "thanked",    "had no degree",         "was not a surprise", "Was the XXXX unsurprising?", "Y"])
nounsAndVerbs.append(["the extremist",       "the agent",          "caught",     "got an award",         "was disturbing", "Was the XXXX disconcerting?", "Y"])
# Issue: disconcerting is OOV, how about disturbing
nounsAndVerbs.append(["the clerk",           "the customer",       "called",     "was a hero",         "seemed absurd", "Did the XXXX seem absurd?", "Y"])
nounsAndVerbs.append(["the trader",          "the businessman",    "consulted",  "had insider information",         "was confirmed", "Was the XXXX confirmed?", "Y"])
nounsAndVerbs.append(["the CEO",             "the employee",       "impressed",  "was retiring",         "was entirely correct", "Was the XXXX correct?", "Y"])

overallUnigramProbs = {"V2" : 0, "V1" : 0}
for line in nounsAndVerbs:
    for key in overallUnigramProbs:
      phrase = line[{"V2" : 3, "V1" : 4}[key]].split(" ")
      logprob = 0
      for word in phrase:
         logprob += math.log(counts[word]) - math.log(countsTotal)
      overallUnigramProbs[key] += logprob/ len(nounsAndVerbs)

print(overallUnigramProbs)

