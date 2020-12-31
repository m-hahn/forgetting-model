from collections import defaultdict
import stanza
queue = []
noun, cont = None, None
useNext = True
import sys


def getParseLabel(sent):
    NOUN = sent.words[1]
    PRINCIPAL = sent.words[4]
    TEACHER = sent.words[7]
    if NOUN.deprel == "root":
      label = "incomplete"
    elif NOUN.head == PRINCIPAL.head:
      label = "incomplete"
    elif TEACHER.head == PRINCIPAL.head:
       label = "incomplete"
    elif not NOUN.deprel.startswith("nsubj"):
        label = "incomplete"
    elif not PRINCIPAL.deprel.startswith("nsubj"):
        label = "incomplete"
    elif len(  [x.text for x in sent.words if x.head == NOUN.head and x.deprel.startswith("nsubj")]) > 1:
        label = "incomplete"
    elif NOUN.deprel.startswith("nsubj") and PRINCIPAL.deprel.startswith("nsubj") and TEACHER.deprel.startswith("nsubj"):
        label = "complete"
    else:
        label = "unknown" # occasional other illformedness, e.g. when TEACHER has a `compound' label
    return label

from collections import defaultdict

byNoun = {}
nouns = ["story", "report", "declaration", "admission", "fact", "assumption", "belief", "assertion"]
for n in nouns:
   byNoun[n] = []

ID = sys.argv[1]
with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2_LARGE.py_{ID}", "r") as inFile:
  iterations = next(inFile).strip()
  config = next(inFile).strip()
  for line in inFile:
      if line.startswith("Model"):
         line = line.split("\t")
         #print(line)
         if len(line) < 4:
           continue
         multiplicity = int(line[2])
         sentence = line[3].strip()
         if ". " in sentence:
            sentence = sentence[:sentence.index(". ")+1]
#            print(sentence[:sentence.index(". ")]+"###"+sentence[sentence.index(". "):]
         #print(sentence)
         noun = line[0]
#         print(noun)
         sentenceR = sentence.strip()
         sentence = (sentence.split(" "))
         sentence = (" ".join((["The", noun.replace("Model", "").strip(), "that", "the", "principal", "who", "the", "teacher"] + sentence[8:])))
         #.append((sentence, multiplicity))
         noun = noun.replace("Model ", "").strip()
         if noun in byNoun:
            byNoun[noun].append((sentenceR, multiplicity, sentenceR))
import numpy as np
output = []
for noun in nouns:
    print(len(byNoun[noun]))
    probabilities = [x[1] for x in byNoun[noun]]
    probabilities = np.array(probabilities)/sum(probabilities)
    chosen = np.random.choice(a=len(byNoun[noun]), p=probabilities, size=2, replace=True)
    print(chosen)
    for i in chosen:
      print(i, noun)
      output.append((noun, byNoun[noun][int(i)][0]))

with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs-gpt2-withOrig2_human/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2_LARGE.py_{ID}", "w") as outFile:
  print("#", iterations, config, file=outFile)
  print("\t".join(["Noun", "Verbs", "Sentence"]), file=outFile)
  for noun, sent in output:
      print("\t".join([noun, "NA", sent]), file=outFile)
 #             print("\t".join([str(x) for x in [noun.split(" ")[1].strip(), cont, countByRelation["complete"], countByRelation["incomplete"], thatCount[True], thatCount[False]]]), file=outFile)

