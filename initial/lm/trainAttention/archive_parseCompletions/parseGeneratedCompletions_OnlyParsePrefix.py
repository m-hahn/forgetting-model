from collections import defaultdict
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=True)
queue = []
noun, cont = None, None
useNext = True
import sys
ID = sys.argv[1]
with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs-gpt2-onlyPrefix/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2.py_{ID}", "w") as outFile:
 print("\t".join(["Noun", "Continuation", "complete", "incomplete", "isThat", "isNotThat"]), file=outFile)
 with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2.py_{ID}", "r") as inFile:
  for line in inFile:
      if line.startswith("Model"):
         line = line.split("\t")
        # print(line)
         if len(line) < 3:
           continue
         sentence = line[2].strip()
         if ". " in sentence:
            sentence = sentence[:sentence.index(". ")+1]
#            print(sentence[:sentence.index(". ")]+"###"+sentence[sentence.index(". "):]
         if noun is not None:
           sentence = (sentence.split(" "))
           prefix = " ".join(sentence[:8])
           print(prefix)
           if "OOV" in prefix:
              continue
           queue.append(prefix+" stopped yesterday arrived proved true.")

           if line[0]!=noun or line[1] != cont:
              print(len(queue))
              countByRelation = defaultdict(int)
              thatCount = defaultdict(int)
              doc = nlp(". ANOTHER SENTENCE. \n\n ".join(queue))
              queue = []
     #         print(str(doc)[:1000])
              for sent in doc.sentences:
    #            print(sent)
   #             print(" ".join([str(x.head) for x in sent.words]))
  #              print(" ".join([x.deprel for x in sent.words]))
 #               print(" ".join([x.upos for x in sent.words]))
                if useNext:
                   if len(sent.words) < 4:
                      print("ERROR", " ".join([x.text for x in sent.words]))
                      continue
                 
                   print(" ".join([str(x.id)+"_"+x.text+"_"+str(x.head)+"_"+x.deprel for x in sent.words]))
                   continue
                  
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
#                     print(" ".join([x.text for x in sent.words]))
#                     print(NOUN.head, NOUN.deprel, PRINCIPAL.head, PRINCIPAL.deprel, TEACHER.head, TEACHER.deprel)
#                     print(str(sent.words[NOUN.head-1].text), str(sent.words[PRINCIPAL.head-1].text), str(sent.words[TEACHER.head-1].text))
#                     print([x.text for x in sent.words if x.head == NOUN.head and x.deprel.startswith("nsubj")])
#                     print([x.text for x in sent.words if x.head == PRINCIPAL.head and x.deprel.startswith("nsubj")])
#                     print([x.text for x in sent.words if x.head == TEACHER.head and x.deprel.startswith("nsubj")])
#
           
                   countByRelation[label] += 1
                   thatCount[sent.words[2].text == "that"] += 1
         #          if sent.words[1].deprel == "ccomp":
        #               print(" ".join([x.text for x in sent.words]))

 #               else:
#                  print(" ".join([x.text for x in sent.words]))
                 
                if "ANOTHER" in (" ".join([x.text for x in sent.words])):
                   assert not useNext, " ".join([x.text for x in sent.words])
                   useNext = True
                else:
                   useNext = False
      #          quit()
              quit()
              print("\t".join([str(x) for x in [noun.split(" ")[1].strip(), cont, countByRelation["complete"], countByRelation["incomplete"], thatCount[True], thatCount[False]]]), file=outFile)
              print(noun, cont, countByRelation)
#              quit()
         noun, cont = line[0], line[1]
              
