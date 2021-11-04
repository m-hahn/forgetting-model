# recursive_PRD_Wiki_BPTT_Clip_GRU_Restart_LogExp11.py
# Works without NA
# ./python27 recursive_PRD_Wiki_BPTT_Clip_GRU_Restart_LogExp11.py --beta 4.539993e-05 --lr 0.0001

# recursive_PRD_Wiki_BPTT_Clip_GRU_Restart_LogExp11_CHOSEN_CharAware_BatchSoftmax.py
# Could improve efficiency by batching the Gaussian density computations


import time


#import torchkit.optim
#import torchkit.nn, torchkit.flows, torchkit.utils
import numpy as np

import random
import sys

objectiveName = "LM"

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="english")
parser.add_argument("--dropout_rate", type=float, default=random.choice([0.0]))
parser.add_argument("--emb_dim", type=int, default=200)
parser.add_argument("--rnn_dim", type=int, default=512)
parser.add_argument("--rnn_layers", type=int, default=1)
parser.add_argument("--lr", type=float, default=random.choice([0.0001])) # 0.00001, 
parser.add_argument("--input_dropoutRate", type=float, default=0.0)
parser.add_argument("--batchSize", type=int, default=512)
parser.add_argument("--sequence_length", type=int, default=20)
parser.add_argument("--log_beta", type=float, default=random.choice([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]))
parser.add_argument("--flow_length", type=int, default=0) #random.choice([0,1]))
parser.add_argument("--flowtype", type=str, default=random.choice(["ddsf", "dsf"]))
parser.add_argument("--flow_hid_dim", type=int, default=512)
parser.add_argument("--flow_num_layers", type=int, default=2)
parser.add_argument("--myID", type=int, default=random.randint(0,10000000))
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--norm_clip", type=float, default=2.0)
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)



args=parser.parse_args()
assert args.flow_length == 0
print(str(args))

BETA = math.exp(-args.log_beta)


model = "REAL"



assert args.dropout_rate <= 0.5
assert args.input_dropoutRate <= 0.5

devSurprisalTable = [None] * ((args.sequence_length+1))


print("TESTING, NO LOGGING")


posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]

import math
from math import log, exp
from random import random, shuffle

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

import corpusIteratorWikiWords

originalDistanceWeights = {}





#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax(dim=2)

from paths import CHAR_VOCAB_HOME


char_vocab_path = "vocabularies/"+args.language.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


import os


vocab_size = 50000


word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(itos)+3, embedding_dim=args.emb_dim).cuda()
outVocabSize = 3+len(itos) 


itos_total = ["EOS", "OOV", "SOS"] + itos #+ itos_lemmas[:vocab_size] + itos_morph
assert len(itos_total) == outVocabSize
# could also provide per-word subcategorization frames from the treebank as input???


#baseline = nn.Linear(args.emb_dim, 1).cuda()

dropout = nn.Dropout(args.dropout_rate).cuda()

rnn_both = nn.GRU(args.emb_dim, args.rnn_dim, args.rnn_layers).cuda()
for name, param in rnn_both.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(args.rnn_dim,outVocabSize).cuda()
#pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()

startHidden = nn.Linear(1, args.rnn_dim).cuda()
startHidden.bias.data.fill_(0)


components = [rnn_both, decoder, word_pos_morph_embeddings, startHidden]


 


hiddenToLogSDHidden = nn.Linear(args.rnn_dim, args.rnn_dim).cuda()
cellToMean = nn.Linear(args.rnn_dim, args.rnn_dim).cuda()
sampleToHidden = nn.Linear(args.rnn_dim, args.rnn_dim).cuda()

hiddenToLogSDHidden.bias.data.fill_(0)
cellToMean.bias.data.fill_(0)
sampleToHidden.bias.data.fill_(0)

hiddenToLogSDHidden.weight.data.fill_(0)
cellToMean.weight.data.fill_(0)
sampleToHidden.weight.data.fill_(0)







#import torchkit.nn as nn_













components = components + [hiddenToLogSDHidden, cellToMean, sampleToHidden]








def parameters():
 for c in components:
   for param in c.parameters():
      yield param

initrange = 0.1
word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)




crossEntropy = 10.0

#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)

optimizer = torch.optim.Adam(parameters(), lr=args.lr, betas=(0.9, 0.999) , weight_decay=args.weight_decay)


import torch.cuda
import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=args.input_dropoutRate)


counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 
devMemories = []


lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)


mask1 = torch.FloatTensor([[1 if k > d else 0 for d in range(args.rnn_dim)] for k in range(args.rnn_dim)]).cuda()
mask2 = torch.FloatTensor([[1 if k < d else 0 for d in range(args.rnn_dim)] for k in range(args.rnn_dim)]).cuda()




standardNormal = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(args.rnn_dim)] for _ in range((args.sequence_length+1)*args.batchSize)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(args.rnn_dim)] for _ in range((args.sequence_length+1)*args.batchSize)]).cuda())
standardNormalPerStep = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(args.rnn_dim)] for _ in range(args.batchSize)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(args.rnn_dim)] for _ in range(args.batchSize)]).cuda())






def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      numerified_chars = []
      for chunk in data:
       for char in chunk:
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         if char == ",": # Skip commas
           continue
         numerified.append((stoi[char]+3 if char in stoi else 2))
#         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])

       if len(numerified) > (args.batchSize*((args.sequence_length+1))):
         sequenceLengthHere = (args.sequence_length+1)

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
#         numerifiedCurrent_chars = numerified_chars[:cutoff]

#         for i in range(len(numerifiedCurrent_chars)):
#            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i][:15] + [1]
#            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i] + ([0]*(16-len(numerifiedCurrent_chars[i])))


         numerified = numerified[cutoff:]
#         numerified_chars = numerified_chars[cutoff:]
       
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
#         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(args.batchSize, -1, sequenceLengthHere, 16).transpose(0,1).transpose(1,2).cuda()

         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], None
         hidden = None
       else:
         print("Skipping")



hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = zeroBeginning

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.FloatTensor([0 for _ in range(args.batchSize)]).cuda().view(args.batchSize, 1)

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())




def forward(numeric, surprisalTable=None, doDropout=True, batchSizeHere=1):
       global counter
       global crossEntropy
       global printHere
       global devLosses
 
       global hidden
       global beginning
       global beginning
       global beginning_chars

       if hidden is not None:
           hidden = Variable(hidden.data).detach()
           forRestart = bernoulli.sample()
           #print(forRestart)
           sampled = startHidden(zeroHidden)
           hiddenNew = sampleToHidden(sampled).unsqueeze(0)
#           hidden = forRestart.unsqueeze(0).unsqueeze(2) * hiddenNew + (1-forRestart).unsqueeze(0).unsqueeze(2) * hidden
 #          print(torch.where)
           hidden = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, hiddenNew, hidden)
           beginning = torch.where(forRestart.unsqueeze(0) == 1, zeroBeginning, beginning)
#           beginning = forRestart.unsqueeze(0).unsqueeze(2) * zeroBeginning + (1-forRestart).unsqueeze(0).unsqueeze(2) * beginning
           beginning_chars = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroBeginning_chars, beginning_chars)
       else:
           sampled = startHidden(zeroHidden)
           hiddenNew = sampleToHidden(sampled).unsqueeze(0)
           hidden = hiddenNew
           beginning = zeroBeginning
           beginning_chars = zeroBeginning_chars

       numeric, numeric_chars = numeric
       numeric = torch.cat([beginning, numeric], dim=0)

       beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)
 



       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0

       optimizer.zero_grad()

       for c in components:
          c.zero_grad()
#       for q in parameters_made:
#        for p in q:
#         if p.grad is not None:
#          p.grad.fill_(0)
       totalQuality = 0.0

       if True:
           
           inputTensor = numeric # so it will be sequence_length+1 x args.batchSizeHere
#           print inputTensor
#           quit()

           inputTensorIn = inputTensor[:-1]
           inputTensorOut = inputTensor[1:]



     
#           embedded = word_embeddings(input_tensor)
           inputEmbeddings = word_pos_morph_embeddings(inputTensorIn.view((args.sequence_length+1), batchSizeHere))


           encodedEpsilonForAllSteps = standardNormal.sample().view((args.sequence_length+1), args.batchSize, -1)
           meanHidden = cellToMean(hidden[0])
           klLoss = [None for _ in inputEmbeddings]
           logStandardDeviationHidden = hiddenToLogSDHidden(hidden[0])
   #        print(torch.exp(logStandardDeviationHidden))
           scaleForDist = 1e-8 + torch.log(1+torch.exp(logStandardDeviationHidden))
           scales.append(scaleForDist)
           means.append(meanHidden)
           encodedEpsilon = encodedEpsilonForAllSteps[i] #standardNormalPerStep.sample()
#           encodedEpsilon = torch.clamp(encodedEpsilon, min=-10, max=10)
           sampled = meanHidden + scaleForDist * encodedEpsilon
           sampled_vectors.append(sampled)




           lossesWordTotal = []

           sampled_vectors = []
           logProbConditionals = []

           output_vectors = []


           scales = []
           means = []                      

           meanHidden = torch.stack(means, dim=0)
           scaleForDist = torch.stack(scales, dim=0)
           memoryDistribution = torch.distributions.Normal(loc=meanHidden, scale=scaleForDist)


           sampled = torch.stack(sampled_vectors, dim=0)

           logProbConditional = memoryDistribution.log_prob(sampled).sum(dim=2)  #


           batchSizeInflatedHere = args.batchSize * len(sampled_vectors)


           sampledTotal = sampled.view(batchSizeInflatedHere, -1)

           #print(logProbConditional.size())


#           print(output_vectors[0].size())
           output = torch.cat(output_vectors, dim=0)
    #       print(output.size())
           word_logits = decoder(output)
 #          print(word_logits.size())

#           word_logits = word_logits.view((args.sequence_length+1), batchSizeHere, outVocabSize)
           word_softmax = logsoftmax(word_logits)
#           print(word_softmax)

 #          print(word_softmax.size())
  #         print(torch.exp(word_softmax).sum(dim=2))
#           print(word_softmax.size())
#           print(word_logits.abs().max(), word_softmax.abs().max())
          
#           print(word_softmax.size(), inputTensorOut.size())
           lossesWord = lossModuleTest(word_softmax.view(-1, 50003), inputTensorOut.view(-1))
#           print(inputTensorOut)
 #          print(lossesWord.mean())
 #          lossesWordTotal.append(lossesWord) 

#           lossesWord = torch.stack(lossesWordTotal, dim=0)
           lossWords = lossesWord.sum()
           loss = lossWords

           klLoss = 0

           logProbConditionalsTotal = logProbConditional.view(batchSizeInflatedHere)
#           print(sampledTotal.size(), logProbConditionalsTotal.size())


           #for sampled, logProbConditional in zip(sampled_vectors, logProbConditionals):
           adjustment = []
           epsilon = sampledTotal
   #        n=1
   
           plainPriorLogProb = standardNormal.log_prob(epsilon).sum(dim=1) #- (0.5 * torch.sum(sampled * sampled, dim=1))
           logProbMarginal = plainPriorLogProb 
           #print(loss, logProbConditionalsTotal.mean(), logProbMarginal.mean())   
   
           klLoss = (logProbConditionalsTotal - logProbMarginal)
   #           print(logProbConditional, logProbMarginal)
   #           print(logStandardDeviationHidden)
   #           klLoss = 0.5 * (-1 - 2 * (logStandardDeviationHidden) + torch.pow(meanHidden, 2) + torch.exp(2*logStandardDeviationHidden))
    #          klLoss = klLoss.sum(1)


           klLossSum = klLoss.sum()
           if counter % 10 == 0:
              klLossMean = klLoss.mean()
              print(BETA, args.flow_length, klLossMean, lossesWord.mean(), BETA * klLossMean + lossesWord.mean() )
              if float(klLossMean) != float(klLossMean):
                 print(hidden.abs().max())
                 assert False, "got NA, abort"
           loss = loss + BETA * klLossSum
#           print lossesWord

           if surprisalTable is not None or printHere:           
             lossesCPU = lossesWord.data.cpu().view(((args.sequence_length+1)), batchSizeHere).numpy()
             if printHere:
                for i in range(0,(args.sequence_length+1)): #range(1,maxLength+1): # don't include i==0
                         j = 0
                         print (i, itos_total[numeric[i+1][j]], lossesCPU[i][j])

             if surprisalTable is not None: 
                if printHere:
                   print(surprisalTable)
                for j in range(batchSizeHere):
                  for r in range((args.sequence_length+1)):
                    surprisalTable[r] += lossesCPU[r,j] #.data.cpu().numpy()[0]

           wordNum = ((args.sequence_length+1)-1)*batchSizeHere
                    
       if wordNum == 0:
         print(input_words)
         print(batchOrdered)
         return 0,0,0,0,0
       if printHere:
         print( loss/wordNum)
         print( lossWords/wordNum)
         print( ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)])
         print( ("beta", BETA))
       crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum).data.cpu().numpy()
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
#       probabilities = torch.sigmoid(dhWeights)
#       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

#       policy_related_loss = lr_policy * (entropy_weight * neg_entropy + policyGradientLoss) # lives on CPU
       loss = loss / batchSizeHere
       return loss, None, None, totalQuality, numberOfWords, klLoss.mean() if not doDropout else None


parameterList = list(parameters())

def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns


      # print(cellToMean.weight)
      # print(cellToMean.bias)
      # print(hiddenToLogSDHidden.weight)
      # print(hiddenToLogSDHidden.bias)




       loss.backward()
       if printHere:
         print( "BACKWARD 3 "+__file__+" "+args.language+" "+str(args.myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+str(args))
         print( devLosses)
         print( lastDevLoss)
         print( failedDevRuns)
#       print("MAX NORM", max(p.grad.data.abs().max() for p in parameterList))
       # REMOVED GRADIENT CLIPPING, HOPING FOR SPEED
      # torch.nn.utils.clip_grad_norm(parameterList, args.norm_clip, norm_type='inf')
       optimizer.step()
       for param in parameters():
         if param.grad is None:
           print("WARNING: None gradient")
#           continue
#         param.data.sub_(lr_lm * param.grad.data)



def createStream(corpus):
#    global counter
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       #printHere = (sentCount % 10 == 0)
       ordered = sentence

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered:
          wordStartIndices.append(len(input_indices))
          if line not in stoi:
            input_indices.append(1)
          else:
            input_indices.append(stoi[line]+3)
          if len(wordStartIndices) == (args.sequence_length+1):
             yield input_indices, wordStartIndices
             input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = []






def computeDevLoss():
   global printHere
#   global counter
#   global devSurprisalTable
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("valid")
   corpusDev = corpusIteratorWikiWords.dev(args.language)
   stream = prepareDatasetChunks(corpusDev, train=False)

   surprisalTable = [0 for _ in range((args.sequence_length+1))]
   devCounter = 0
   devMemory = 0
   while True:
     devCounter += 1
     printHere = (devCounter % 50 == 0)
     try:
       with torch.no_grad():
          _, _, _, newLoss, newWords, devMemoryHere = forward(next(stream), surprisalTable = surprisalTable, doDropout=False, batchSizeHere=args.batchSize)
     except StopIteration:
        break

     devMemory += devMemoryHere.data.cpu().numpy()
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print("Dev examples "+str(devCounter))
   devSurprisalTableHere = [surp/(devCounter*args.batchSize) for surp in surprisalTable]
   return devLoss/devWords, devSurprisalTableHere, devMemory/devCounter



wordsProcessed = 0
startTime = time.time()

DEV_PERIOD = 10000
epochCount = 0
while failedDevRuns < 10:
  epochCount += 1
  print("Starting new epoch, permuting corpus")
  corpus = corpusIteratorWikiWords.training(args.language)
#  stream = createStream(corpus)
  stream = prepareDatasetChunks(corpus, train=True)

  while True:
       counter += 1
       printHere = (counter % 50 == 0)


       if counter % DEV_PERIOD == 0:
          hidden = None
          beginning = zeroBeginning

          newDevLoss, devSurprisalTableHere, newDevMemory = computeDevLoss()
          
          hidden = None
          beginning = zeroBeginning
          

#             devLosses.append(
          devLosses.append(newDevLoss)
          devMemories.append(newDevMemory)
          print("New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss))
#          if newDevLoss > 10:
#              print "Abort, training too slow?"
#              devLosses.append(100)

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
         


          print(devSurprisalTable[(args.sequence_length+1)/2])
          print(devMemories)
          assert False
#          with open("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+args.language+"_"+__file__+"_model_"+str(args.myID)+"_"+model+".txt", "w") as outFile:
#              print >> outFile, str(args)
#              print >> outFile, " ".join(map(str,devLosses))
#              print >> outFile, " ".join(map(str,devSurprisalTable))
#              print >> outFile, " ".join(map(str, devMemories))
#              print >> outFile, str(sum([x-y for x, y in zip(devSurprisalTable[:(args.sequence_length+1)/2], devSurprisalTable[(args.sequence_length+1)/2:])]))
#          state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in components]}
#          torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")
#
#
#
#          if lastDevLoss is None or newDevLoss < lastDevLoss:
#             lastDevLoss = newDevLoss
#             #failedDevRuns = 0
#          else:
#             failedDevRuns += 1
#             print "Skip saving, hoping for better model"
#             print devLosses
#             print "Epoch "+str(epochCount)+" "+str(counter)
#             print zip(range(1,(args.sequence_length+1)+1), devSurprisalTable)
#             print "MI(Bottleneck, Future) "+str(sum([x-y for x, y in zip(devSurprisalTable[0:(args.sequence_length+1)/2], devSurprisalTable[(args.sequence_length+1)/2:])]))
#             print "Memories "+str(devMemories)
#             #break

       try:
         loss, baselineLoss, policy_related_loss, _, wordNumInPass, _ = forward(next(stream), batchSizeHere=args.batchSize)
       except StopIteration:
          break

       wordsProcessed += (args.batchSize * (args.sequence_length+1))

       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print("No words, skipped backward")
       if printHere:
          print("Epoch "+str(epochCount)+" "+str(counter))
          print(zip(range(1,(args.sequence_length+1)+1), devSurprisalTable))
          if devSurprisalTable[0] is not None:
             print("MI(Bottleneck, Future) "+str(sum([x-y for x, y in zip(devSurprisalTable[0:(args.sequence_length+1)/2], devSurprisalTable[(args.sequence_length+1)/2:])])))
             print("Memories "+str(devMemories))
          print(str(wordsProcessed/(time.time() - startTime))+" words per second.")



print(devSurprisalTable[int((args.sequence_length+1)/2)])
print(devMemories)

