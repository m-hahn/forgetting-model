from collections import defaultdict
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=True, tokenize_no_ssplit=True)
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

ID = sys.argv[1]
with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs-gpt2-withOrig2/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2_LARGE.py_{ID}", "w") as outFile:
 print("\t".join(["Noun", "Continuation", "complete", "incomplete", "isThat", "isNotThat"]), file=outFile)
 with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2_LARGE.py_{ID}", "r") as inFile:
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
         if noun is not None:
           sentence = (sentence.split(" "))
           sentence = (" ".join((["The", noun.replace("Model", "").strip(), "that", "the", "principal", "who", "the", "teacher"] + sentence[8:])))
         #  print(sentence)
  #         quit()
           queue.append((sentence, multiplicity))
           #print(noun, cont)
           if (line[0]!=noun or line[1] != cont):
              #print(len(queue))
              countByRelation = defaultdict(int)
              thatCount = defaultdict(int)
              queue_sentences = [x[0] for x in queue]
              queue_multiplicities = [x[1] for x in queue]
              doc = nlp("\n\n ".join(queue_sentences))
              queue = []
              collected = list(doc.sentences)
              if len(collected) != len(queue_sentences):
                print("WARNING!!! MISMATCH", len(collected), len(queue_sentences))
              for multiplicity, sent in zip(queue_multiplicities, collected):
                assert multiplicity > 0
                if useNext or True:
                   if len(sent.words) < 4:
                      print("ERROR", " ".join([x.text for x in sent.words]))
                      continue
                   assert sent.words[1].text == noun.split(" ")[1].strip()
                   assert sent.words[4].text == "principal"
                   assert sent.words[7].text == "teacher"
                  

                   label = getParseLabel(sent)
#                   print(label) 
                   countByRelation[label] += multiplicity
                   thatCount[sent.words[2].text == "that"] += multiplicity
                   assert sent.words[1].text == noun.replace("Model", "").strip(), (sent.text, noun.replace("Model", "").strip())
                 
                if "ANOTHER" in (" ".join([x.text for x in sent.words])):
                   assert not useNext, " ".join([x.text for x in sent.words])
                   useNext = True
                else:
                   useNext = False
              print("\t".join([str(x) for x in [noun.split(" ")[1].strip(), cont, countByRelation["complete"], countByRelation["incomplete"], thatCount[True], thatCount[False]]]), file=outFile)
              print( noun, cont, countByRelation)
#              quit()
         noun, cont = line[0], line[1]
              
