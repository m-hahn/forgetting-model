#ID = "960136486"
import sys
ID = sys.argv[1]
path = "/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg.py_"+ID
pathOut = "/u/scr/mhahn/reinforce-logs-both/full-logs-semiautomatically/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg.py_"+ID

from collections import defaultdict

posDictionary = defaultdict(dict)
with open("/u/scr/mhahn/FAIR18/english-wiki-word-vocab_POS.txt", "r") as inFile:
   for line in inFile:
      if len(posDictionary) > 100000:
        break
      line = line.strip().split("\t")
      if line[1][0] not in posDictionary[line[0]]:
          posDictionary[line[0]][line[1][0]] = 0
      posDictionary[line[0]][line[1][0]] += int(line[2])
#print(posDictionary["against"])
#print(posDictionary["the"])
#quit()

def isVerb(x):
   counts = posDictionary[x.lower()]
   if not ("v" in counts):
       return False
   total = sum([y for _, y in counts.items()])
   print(x, counts["v"], total, counts)
   return ("v" in counts) and (counts["v"] > 0.7 * total)


def isNoun(x):
   counts = posDictionary[x.lower()]
   total = sum([y for _, y in counts.items()])
   return ("n" in counts) and (counts["n"] > 0.7 * total)


def isPreposition(x):
   counts = posDictionary[x.lower()]
   total = sum([y for _, y in counts.items()])
   return ("i" in counts) and (counts["i"] > 0.7 * total)

def isDeterminer(x):
   counts = posDictionary[x.lower()]
   total = sum([y for _, y in counts.items()])
   return ("d" in counts) and (counts["d"] > 0.7 * total)


def isAdjective(x):
   counts = posDictionary[x.lower()]
   total = sum([y for _, y in counts.items()])
   return ("j" in counts) and (counts["j"] > 0.7 * total)



assert isDeterminer("Any")
assert isDeterminer("No")

cache = set()
countByTriple = defaultdict(int)
resultsByNoun = defaultdict(dict)

with open("semiautomatically-todo.txt", "a") as outFile2:
 with open(pathOut, "w") as outFile:
  with open(path, "r") as inFile:
   print(next(inFile).strip(), file=outFile)
   config = next(inFile).strip()
   print(config, file=outFile)
   for line in inFile:
     if line.startswith("[") and line.strip().endswith("]"):
        if not line.strip().endswith("'Y']"):
           print("???", line)
        else:
          line = line[2:].strip().split("', '")[:5]
          originalPrefix = "The NOUN that "+line[0]+" who "+line[1]
          
     elif line.startswith("Model "):
        line = line.split("\t")
        if len(line) == 3:
          noun, continuation, sampled_old = line
          sampled_old = " ".join(sampled_old.strip().split(" ")[:8])
          if "OOV" in sampled_old:
             continue
          countByTriple[(noun, originalPrefix, continuation, sampled_old)] += 1
#          print(len(countByTriple))
 triples = 0
 totalCount, not_covered = 0, 0
 nouns = set([x[0] for x in countByTriple])
 for noun, originalPrefix, continuation, sampled_old in countByTriple:
          triples+=1
          count = countByTriple[(noun, originalPrefix,continuation, sampled_old)]
#          if count < 10:
 #            continue
          noun = noun.replace("Model", "").strip()
          originalPrefix = originalPrefix.replace("NOUN", noun)
          if noun not in resultsByNoun:
             resultsByNoun[noun] = {"Complete" : 0, "Incomplete" : 0, "Unknown" : 0}

          if originalPrefix == sampled_old:
            resultsByNoun[noun]["Complete"] += count
            pass
          else:
             totalCount += count
             originalPrefix = originalPrefix.split(" ")
             sampled_old = sampled_old.split(" ")
             disagreements = [(i, originalPrefix[i], sampled_old[i]) for i in range(len(sampled_old)) if originalPrefix[i] != sampled_old[i]]
             if (0, "The", "His") in disagreements:
               disagreements.remove((0, "The", "His"))
             if (0, "The", "An") in disagreements:
               disagreements.remove((0, "The", "An"))
             if (0, "The", "A") in disagreements:
               disagreements.remove((0, "The", "A"))
             if (3, "the", "an") in disagreements:
               disagreements.remove((3, "the", "an"))
             if (3, "the", "a") in disagreements:
               disagreements.remove((3, "the", "a"))
             if (6, "the", "is") in disagreements:
               COMPLETE = False
               resultsByNoun[noun]["Incomplete"] += count
               continue
             if (6, "the", "was") in disagreements:
               COMPLETE = False
               resultsByNoun[noun]["Incomplete"] += count
               continue
             if (2, "that", "to") in disagreements:
               COMPLETE = False
               resultsByNoun[noun]["Incomplete"] += count
               continue
             if (2, "that", "of") in disagreements:
               COMPLETE = False
               resultsByNoun[noun]["Incomplete"] += count
               continue
             if (2, "that", "in") in disagreements:
               COMPLETE = False
               resultsByNoun[noun]["Incomplete"] += count
               continue

             OKAY_DISAGREEMENTS = []
             OKAY_DISAGREEMENTS.append((6, 'the', 'a'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'Further'))
             OKAY_DISAGREEMENTS.append((1, 'assertion', 'fact'))
             OKAY_DISAGREEMENTS.append((1, 'assertion', 'perception'))
             OKAY_DISAGREEMENTS.append((1, 'reminder', 'fact'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'Another'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'This'))
             OKAY_DISAGREEMENTS.append((0, 'The', '"'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'the'))
             OKAY_DISAGREEMENTS.append((3, 'the', 'any'))
             OKAY_DISAGREEMENTS.append((3, 'the', 'this'))
             OKAY_DISAGREEMENTS.append((3, 'the', 'one'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'This'))
             OKAY_DISAGREEMENTS.append((0, 'The', '"'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'One'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'Some'))
             OKAY_DISAGREEMENTS.append((0, 'The', '-'))
             OKAY_DISAGREEMENTS.append((3, 'the', '-'))
             if isNoun(sampled_old[4]) and sampled_old[4] != originalPrefix[4]:
                OKAY_DISAGREEMENTS.append((4, originalPrefix[4], sampled_old[4]))
             if isNoun(sampled_old[1]):
                OKAY_DISAGREEMENTS.append((1, originalPrefix[1], sampled_old[1]))
             if isNoun(sampled_old[7]):
                OKAY_DISAGREEMENTS.append((7, originalPrefix[7], sampled_old[7]))
             if isDeterminer(sampled_old[0]) or isAdjective(sampled_old[0]):
                OKAY_DISAGREEMENTS.append((0, originalPrefix[0], sampled_old[0]))
             if isDeterminer(sampled_old[3]):
                OKAY_DISAGREEMENTS.append((3, originalPrefix[3], sampled_old[3]))

             OKAY_DISAGREEMENTS.append((0, 'The', 'Any'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'No'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'One'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'Some'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'That'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'Her'))
             OKAY_DISAGREEMENTS.append((0, 'The', 'Their'))
             if sampled_old[7].startswith(originalPrefix[7]): # this happens because the prefixes are taken from a previous GPT2 run
                OKAY_DISAGREEMENTS.append((7, originalPrefix[7], sampled_old[7]))
               
               
#             OKAY_DISAGREEMENTS.append(
#             OKAY_DISAGREEMENTS.append(
#             OKAY_DISAGREEMENTS.append(
             for noun2 in nouns:
                 OKAY_DISAGREEMENTS.append((1, noun, noun2))
             NOT_OKAY_DISAGREEMENTS = []
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'by'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'They'))
             NOT_OKAY_DISAGREEMENTS.append((6, 'the', 'has'))
             NOT_OKAY_DISAGREEMENTS.append((5, 'who', 'of'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'With'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'In'))
             NOT_OKAY_DISAGREEMENTS.append((5, 'who', 'was'))
             NOT_OKAY_DISAGREEMENTS.append((6, 'the', 'had'))
             NOT_OKAY_DISAGREEMENTS.append((6, 'the', 'are'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'For'))
             NOT_OKAY_DISAGREEMENTS.append((5, 'who', 'tasked'))
             NOT_OKAY_DISAGREEMENTS.append((5, 'who', 'and'))
             NOT_OKAY_DISAGREEMENTS.append((6, 'the', 'also'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'By'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'On'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'for'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'To'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'Despite'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'for'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'was'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'is'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'from'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'After'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'Of'))
             NOT_OKAY_DISAGREEMENTS.append((5, 'who', 'is'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'As'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'Upon'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'He'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'was'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'is'))
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'It'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', 'and'))
             NOT_OKAY_DISAGREEMENTS.append((2, 'that', ',') )
             NOT_OKAY_DISAGREEMENTS.append((5, 'who', ','))
             NOT_OKAY_DISAGREEMENTS.append((6, 'the', '-'))
             NOT_OKAY_DISAGREEMENTS.append((6, 'the', ','))
             NOT_OKAY_DISAGREEMENTS.append((6, 'the', 'were'))
             if isVerb(sampled_old[1]) and sampled_old[1] != originalPrefix[1]:
                NOT_OKAY_DISAGREEMENTS.append((1, noun, sampled_old[1]))
             if isVerb(sampled_old[6]) and sampled_old[6] != originalPrefix[6]:
                NOT_OKAY_DISAGREEMENTS.append((6, "the", sampled_old[6]))
             if isPreposition(sampled_old[0]) and sampled_old[0] != originalPrefix[0]:
                NOT_OKAY_DISAGREEMENTS.append((0, "The", sampled_old[0]))
             if isVerb(sampled_old[2]) and sampled_old[2] != originalPrefix[2]:
                NOT_OKAY_DISAGREEMENTS.append((2, "that", sampled_old[2]))
             if isVerb(sampled_old[3]) and sampled_old[3] != originalPrefix[3]:
                NOT_OKAY_DISAGREEMENTS.append((3, "the", sampled_old[3]))
             if isPreposition(sampled_old[2]) and sampled_old[2] != originalPrefix[2]:
                NOT_OKAY_DISAGREEMENTS.append((2, "that", sampled_old[2]))
             NOT_OKAY_DISAGREEMENTS.append((2, "that", "about"))

#             NOT_OKAY_DISAGREEMENTS.append(
             NOT_OKAY_DISAGREEMENTS.append((0, 'The', 'When'))
#             NOT_OKAY_DISAGREEMENTS.append(
#             NOT_OKAY_DISAGREEMENTS.append(
#             NOT_OKAY_DISAGREEMENTS.append(
#             NOT_OKAY_DISAGREEMENTS.append(
#             NOT_OKAY_DISAGREEMENTS.append(
#             NOT_OKAY_DISAGREEMENTS.append(
             for x in OKAY_DISAGREEMENTS:
                if x in disagreements:
                    disagreements.remove(x)
             FAILED = False
             for x in NOT_OKAY_DISAGREEMENTS:
                 if x in disagreements:
                     FAILED=True
                     break
             if FAILED:
               resultsByNoun[noun]["Incomplete"] += count
               continue
             if len(disagreements) > 0:
               not_covered += count
               print((originalPrefix, sampled_old, disagreements))
               resultsByNoun[noun]["Unknown"] += count
               for x in disagreements:
                 print(x, file=outFile2)
             else:
               resultsByNoun[noun]["Complete"] += count

#             print((noun,originalPrefix, continuation, sampled_old, count, triples/len(countByTriple)))
   #       quit()        

print(totalCount, not_covered)

config = dict([x.split("=") for x in config[10:-1].split(", ")])
with open("semiautomatic/semiautomatically-normjudg-results.tsv", "a") as outFile:
 for noun in resultsByNoun:
   counts = resultsByNoun[noun]
   print("\t".join([str(x) for x in [ID, config["predictability_weight"], config["deletion_rate"], noun, counts['Complete'], counts['Incomplete'], counts['Unknown']]]), file=outFile)

