import scoreWithGPT2MediumByWord as scoreWithGPT2
import gzip

from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenizer = TreebankWordDetokenizer()
def processSentence(sent):
   #print(len(sent))
   #for i in range(0, len(sent)-15):
   i = 0
   sent = [x for x in  detokenizer.detokenize([sent[x][0] for x in range(i, i+15)]).replace(" , ", ", ").replace('"', '').split(" ") if len(x) > 0]
   batch = []
   batch.append(sent)
#   print(sent)
   for j in range(1,len(sent)-5):
 #    print(j, sent[j:j+5])
     batch.append(sent[j:j+5])
   for i in range(len(batch)):
      batch[i] = " ".join(batch[i])
      batch[i] = batch[i][0].upper() + batch[i][1:]
   scores = scoreWithGPT2.scoreSentences(batch)
#   print(scores[0])
#   print(sent) 
 #  print(scores[0])
#   print("OVERALL", batch[0])
   for j in range(1,len(sent)-5):
      if scores[0][j][-1][0][-1] in [",", ":"]: # Could also do this before running GPT2
#         print("SKIPPING", scores[0][j])
         continue
      infix = (scores[0][j+1:j+1+5])
      infixAsPrefix = [x for x in scores[j] if len(x) > 0]
      infix_string = " ".join(["".join([x[0] for x in y]) for y in infix])
      infixAsPrefix_string = " ".join(["".join([x[0] for x in y]) for y in infixAsPrefix])
      if infix_string.strip() != infixAsPrefix_string.strip().lower():
         print("WARNING:", infix_string, "VERSUS", infixAsPrefix_string)
         continue
#      assert infix[0][0][0].strip()[0] == infixAsPrefix[0][0][0].strip()[0].lower(), (infix, "AND", infixAsPrefix)
      infixSurprisal = sum([sum(x[1] for x in y) for y in infix])
      infixAsPrefixSurprisal = sum([sum(x[1] for x in y) for y in infixAsPrefix])
      if infixAsPrefixSurprisal < infixSurprisal:
         print(infixAsPrefixSurprisal - infixSurprisal, j, infix_string, "AT", infixSurprisal, infixAsPrefixSurprisal, "FROM", batch[0], file=outFile)
#      quit()
  
 #  print(scores)
#   quit()
with open(f"/u/scr/mhahn/COHERENCE/{__file__}.txt","w") as outFile:
 for f in ["english-train-parsed_continuation2.txt.gz", "english-train-parsed_continuation.txt.gz", "english-train-parsed.txt.gz"]:
  sentenceBuffer = []
  with gzip.open(f"/u/scr/mhahn/FAIR18_processed/{f}", "rb") as inFile:
   for line in inFile:
     line = line.decode("utf-8").strip().split("\t")
     if line[0] == "<s>":
        processSentence(sentenceBuffer)
        sentenceBuffer = []
     else:
        sentenceBuffer.append(line)

