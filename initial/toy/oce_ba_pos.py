# Computes estimates also from held-out data.

# Was called zNgramIB_5.py.




import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--language", type=str, dest="language", default="English")
parser.add_argument("--horizon", type=int, dest="horizon", default=1)
parser.add_argument("--code_number", type=int, dest="code_number", default=100)
parser.add_argument("--beta", type=float, dest="beta", default=0.1)
parser.add_argument("--dirichlet", type=float, dest="dirichlet", default=0.00001)

args_names = ["language", "horizon", "code_number", "beta", "dirichlet"]
args = parser.parse_args()


args.beta = 1/args.beta

assert args.beta <= 1.0

import random
import sys




header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator import CorpusIterator

ngrams = {}

lastPosUni = ("EOS",)*(2*args.horizon-1)
for sentence in CorpusIterator(args.language,"train", storeMorph=True).iterator():
 for line in sentence:
   nextPosUni = line["posUni"]
   ngram = lastPosUni+(nextPosUni,)
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
 nextPosUni = "EOS"
 ngram = lastPosUni+(nextPosUni,)
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)


#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


ngrams = list(ngrams.iteritems())

#ngrams = [x for x in ngrams if x[1] > 100]
#print(ngrams)
print(["Number of ngrams", len(ngrams)])


keys = [x[0] for x in ngrams]

total = sum([x[1] for x in ngrams])



frequencies = [x[1] for x in ngrams]

pasts = [x[:args.horizon] for x in keys]  #range(horizon:range(horizon, 2*horizon)]
futures = [x[args.horizon:] for x in keys]


itos_pasts = list(set(pasts)) + ["_OOV_"]
itos_futures = list(set(futures)) + ["_OOV_"]
stoi_pasts = dict(zip(itos_pasts, range(len(itos_pasts))))
stoi_futures = dict(zip(itos_futures, range(len(itos_futures))))

import torch

pasts_int = torch.LongTensor([stoi_pasts[x] for x in pasts])
futures_int = torch.LongTensor([stoi_futures[x] for x in futures])


marginal_past = torch.zeros(len(itos_pasts))
for i in range(len(pasts)):
   marginal_past[pasts_int[i]] += frequencies[i]
marginal_past[-1] = args.dirichlet * len(itos_futures)
marginal_past = marginal_past.div(marginal_past.sum())
print(marginal_past)
print(len(marginal_past))

future_given_past = torch.zeros(len(itos_pasts), len(itos_futures))
for i in range(len(pasts)):
  future_given_past[pasts_int[i]][futures_int[i]] = frequencies[i]
future_given_past[-1].fill_(args.dirichlet)
future_given_past[:,-1].fill_(args.dirichlet)

future_given_past += 0.00001

print(future_given_past.sum(1))
#quit()
 
future_given_past = future_given_past.div(future_given_past.sum(1).unsqueeze(1))

print(future_given_past[0].sum())


def logWithoutNA(x):
   y = torch.log(x)
   y[x == 0] = 0
   return y


marginal_future = torch.zeros(len(itos_futures))
for i in range(len(futures)):
   marginal_future[futures_int[i]] += frequencies[i]
marginal_future[-1] = args.dirichlet * len(itos_pasts)
marginal_future = marginal_future.div(marginal_future.sum())

print(marginal_future)
print(len(marginal_future))




encoding = torch.empty(len(itos_pasts), args.code_number).uniform_(0.000001, 1)
encoding = encoding.div(encoding.sum(1).unsqueeze(1))

decoding = torch.empty(args.code_number, len(itos_futures)).uniform_(0.000001, 1)
decoding = decoding.div(decoding.sum(1).unsqueeze(1))
print(decoding[0].sum())
#quit()

marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), encoding).squeeze(0)


import torch.nn.functional

def runOCE():
    global decoding
    global encoding
    global marginal_hidden
    objective = 10000000
    for t in range(500):
       print("Iteration", t)
    
    
       divergence_by_past = (future_given_past * logWithoutNA(future_given_past))
       divergence_by_past = divergence_by_past.sum(1)
    
       log_future_given_past = logWithoutNA(future_given_past)
    
       log_decoding = logWithoutNA(decoding)
   
       ratios = log_future_given_past.unsqueeze(1) - log_decoding.unsqueeze(0)
       divergence2 = (future_given_past.unsqueeze(1) * ratios).sum(2)

       total_distortion = torch.matmul(marginal_past.unsqueeze(0), divergence2 * encoding).sum()
    
       assert total_distortion >= 0, total_distortion
     
       logNewEncoding = logWithoutNA(marginal_hidden.unsqueeze(0)) + (-args.beta * divergence2)

       logNewEncoding = torch.nn.functional.log_softmax( logNewEncoding, dim=1) #                 logNewEncoding - logNorm
       newEncoding = torch.exp(logNewEncoding)
       new_marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), newEncoding).squeeze(0)
       newEncodingInverted = (newEncoding * marginal_past.unsqueeze(1)).div(new_marginal_hidden.unsqueeze(0))
       newEncodingInverted[new_marginal_hidden.unsqueeze(0).expand(len(itos_pasts), -1) == 0] = 0
    
       newDecoding = torch.matmul(future_given_past.t(), newEncodingInverted).t()
       assert abs(newDecoding[0].sum()) < 0.01 or abs(newDecoding[0].sum() - 1.0) < 0.01 , newDecoding[0].sum()
       
       entropy = new_marginal_hidden * logWithoutNA(new_marginal_hidden)
       entropy = -torch.sum(entropy)
        
       print("Entropy", entropy)
       encoding = newEncoding
       decoding = newDecoding
       marginal_hidden = new_marginal_hidden
    
       logDecoding = logWithoutNA(decoding) 
       logFutureMarginal = logWithoutNA(marginal_future)
       miWithFuture = torch.sum((decoding * (logDecoding - logFutureMarginal.unsqueeze(0))).sum(1) * marginal_hidden)
    
       logEncoding = logWithoutNA(encoding)
       log_marginal_hidden = logWithoutNA(marginal_hidden)
    
       miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
       assert miWithFuture <= miWithPast+1e-5, (miWithFuture , miWithPast)
       newObjective = 1/args.beta * miWithPast - miWithFuture
       print(["Mi with future", miWithFuture, "Mi with past", miWithPast])
       print(["objectives","last",objective, "new", newObjective])
       if not (newObjective - 0.1 <= objective):
          print ("WARNING: Objective not improving. ", newObjective, objective)
       if newObjective == objective:
         print("Ending")
         break
       objective = newObjective
    return encoding, decoding, logDecoding, miWithPast, log_marginal_hidden

encoding, decoding, logDecoding, miWithPast_train, log_marginal_hidden = runOCE()

futureSurprisal_train = -((future_given_past * marginal_past.unsqueeze(1)).unsqueeze(1) * encoding.unsqueeze(2) * logDecoding.unsqueeze(0)).sum()


#assert False, "how is the vocabulary for held-out data generated????"
# try on held-out data

ngrams = {}

lastPosUni = ("EOS",)*(2*args.horizon-1)
for sentence in CorpusIterator(args.language,"dev", storeMorph=True).iterator():
 for line in sentence:
   nextPosUni = line["posUni"]
   ngram = lastPosUni+(nextPosUni,)
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
 nextPosUni = "EOS"
 ngram = lastPosUni+(nextPosUni,)
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)


#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


ngrams = list(ngrams.iteritems())

#ngrams = [x for x in ngrams if x[1] > 100]
#print(ngrams)
#print(["Number of ngrams", len(ngrams)])


keys = [x[0] for x in ngrams]

total = sum([x[1] for x in ngrams])



frequencies = [x[1] for x in ngrams]

pasts = [x[:args.horizon] for x in keys]  #range(horizon:range(horizon, 2*horizon)]
futures = [x[args.horizon:] for x in keys]



import torch

pasts_int = torch.LongTensor([stoi_pasts[x] if x in stoi_pasts else stoi_pasts["_OOV_"] for x in pasts])
futures_int = torch.LongTensor([stoi_futures[x]  if x in stoi_futures else stoi_futures["_OOV_"] for x in futures])


marginal_past = torch.zeros(len(itos_pasts))
for i in range(len(pasts)):
   marginal_past[pasts_int[i]] += frequencies[i]
#marginal_past[-1] = len(itos_futures)
marginal_past = marginal_past.div(marginal_past.sum())

future_given_past = torch.zeros(len(itos_pasts), len(itos_futures))
for i in range(len(pasts)):
  future_given_past[pasts_int[i]][futures_int[i]] = frequencies[i]
#future_given_past[-1].fill_(1)
#future_given_past[:,-1].fill_(1)

future_given_past += 0.00001


future_given_past = future_given_past.div(future_given_past.sum(1).unsqueeze(1))

#marginal_future = torch.zeros(len(itos_futures))
#for i in range(len(futures)):
#   marginal_future[futures_int[i]] += frequencies[i]
#marginal_future = marginal_future.div(marginal_future.sum())


marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), encoding).squeeze(0)


logDecoding = logWithoutNA(decoding) 
#logFutureMarginal = logWithoutNA(marginal_future)

futureSurprisal = -((future_given_past * marginal_past.unsqueeze(1)).unsqueeze(1) * encoding.unsqueeze(2) * logDecoding.unsqueeze(0)).sum()



logEncoding = logWithoutNA(encoding)

miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
print(["Mi with past", miWithPast, "Future Surprisal", futureSurprisal/args.horizon, "Horizon", args.horizon]) # "Mi with future", miWithFuture


myID = random.randint(0,10000000)


outpath = "../../results/outputs-oce/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+".txt"
with open(outpath, "w") as outFile:
    print >> outFile, "\t".join(x+" "+str(getattr(args,x)) for x in args_names)
    print >> outFile, float(miWithPast)
    print >> outFile, float(futureSurprisal/args.horizon)
    print >> outFile, float(miWithPast_train)
    print >> outFile, float(futureSurprisal_train/args.horizon)


print(outpath)


