from collections import defaultdict
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', use_gpu=True)
queue = []
noun, cont = None, None
useNext = True
import sys
ID = sys.argv[1]
with open(f"/u/scr/mhahn/reinforce-logs-both/full-logs-gpt2-withOrig/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_Sampling_GPT2.py_{ID}", "w") as outFile:
 print("\t".join(["Noun", "Continuation", "nsubj", "root", "isThat", "isNotThat"]), file=outFile)
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
         print(sentence)
         if noun is not None:
           sentence = (sentence.split(" "))
           sentence = (" ".join((["The", noun.replace("Model", "").strip(), "that", "the", "principal", "who", "the", "teacher"] + sentence[8:])))
           print(sentence)
  #         quit()
           queue.append(sentence)

           if line[0]!=noun or line[1] != cont:
              print(len(queue))
              countByRelation = defaultdict(int)
              thatCount = defaultdict(int)
              doc = nlp(" this was it. And then she walked away. ANOTHER SENTENCE. \n\n ".join(queue))
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
#                   print(noun.split(" ")[1], cont, sent.words[1].deprel, " ".join([x.text for x in sent.words])) # root cs nsubj
                   
                   countByRelation[sent.words[1].deprel] += 1
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
#                print(" ".join([str(x.id)+"_"+x.text+"_"+str(x.head)+"_"+x.deprel for x in sent.words]))
      #          quit()
       #       quit()
              print("\t".join([str(x) for x in [noun.split(" ")[1].strip(), cont, countByRelation["nsubj"]+countByRelation["nsubj:pass"], countByRelation["root"]+countByRelation["ccomp"], thatCount[True], thatCount[False]]]), file=outFile)
              print(noun, cont, countByRelation)
         noun, cont = line[0], line[1]
              
