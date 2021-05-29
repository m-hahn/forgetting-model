import numpy as np
# GD version of 5, appears to come up with better solutions than 5
# With arguments

# Was called zNgramIB_9_TOY.py.

import matplotlib
matplotlib.use('Agg')


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--language", type=str, dest="language", default="Recursion")
parser.add_argument("--horizon", type=int, dest="horizon", default=5)
parser.add_argument("--code_number", type=int, dest="code_number", default=100)
parser.add_argument("--beta", type=float, dest="beta", default=0.1)
parser.add_argument("--dirichlet", type=float, dest="dirichlet", default=0.00001)

args_names = ["language", "horizon", "code_number", "beta", "dirichlet"]
args = parser.parse_args()



import random
import sys




header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]


from collections import defaultdict
grammar = defaultdict(list)
grammar["S"].append((("NP1", "VP",), 0.5))
grammar["S"].append((("NP2", "VP",), 0.5))

grammar["NP1"].append((("N1",), 0.1))
#grammar["NP1"].append((("N1", "SC",), 0.7))
#grammar["NP1"].append((("N1", "PP",), 0.2))

grammar["NP2"].append((("N2",), 0.1))
#grammar["NP2"].append((("N2", "SC",), 0.2))
#grammar["NP2"].append((("N2", "PP",), 0.7))

grammar["NP3"].append((("N3",), 0.99))

grammar["PP"].append((("about", "NP3",), 0.99))

grammar["SC"].append((("that", "NP3", "VP",), 0.99))

grammar["VP"].append((("V",), 0.99))
#grammar["VP"].append((("V", "NP3",), 0.99))

grammar["V"].append((("annoyed",), 0.25))
grammar["V"].append((("shocked",), 0.25))
grammar["V"].append((("surprised",), 0.25))
grammar["V"].append((("pleased",), 0.25))

grammar["N1"].append((("fact",), 0.33))
grammar["N1"].append((("belief",), 0.33))
grammar["N1"].append((("reassurance",), 0.33))

grammar["N2"].append((("report",), 0.33))
grammar["N2"].append((("story",), 0.33))
grammar["N2"].append((("admission",), 0.33))

grammar["N3"].append((("doctor",), 0.25))
grammar["N3"].append((("patient",), 0.25))
grammar["N3"].append((("janitor",), 0.25))
grammar["N3"].append((("diplomat",), 0.25))


def sample(cat):
   if cat in grammar:
      productions = grammar[cat]
      probabilities = [x[1] for x in productions]
      probabilities = [x/sum(probabilities) for x in probabilities]
    
      selected = np.random.choice(list(range(len(productions))), p=probabilities)
      r = []
      for x in productions[selected][0]:
         r += sample(x)
      return r
   else:
      return [cat]




ngrams = {}

def process(x):
   while "EOS" in x[:-1] and x.index("EOS") + 1 < len(x):
      x = tuple(x[x.index("EOS")+1:])
   assert len(x) >= 1
   return x

lastPosUni = ("EOS",)*(args.horizon-1)
for _ in range(1000):
 sentence = sample("S")
 for line in sentence:
   nextPosUni = line
   ngram = process(lastPosUni+(nextPosUni,))
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
   print(ngram)
 nextPosUni = "EOS"
 ngram = process(lastPosUni+(nextPosUni,))
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)
#quit()

#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


ngrams = list(ngrams.iteritems())
ngrams = sorted(ngrams, key=lambda x:x[1], reverse=True)
#ngrams = [x for x in ngrams if x[1] > 100]
#print(ngrams)
print(["Number of ngrams", len(ngrams)])

print(ngrams)

keys = [x[0] for x in ngrams]

total = sum([x[1] for x in ngrams])



frequencies = [x[1] for x in ngrams]

pasts = [x[:-1] for x in keys]  #range(horizon:range(horizon, 2*horizon)]
futures = [x[-1:] for x in keys]


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
logFutureMarginal = logWithoutNA(marginal_future)

print(marginal_future)
print(len(marginal_future))

import torch.optim



encoding_logits = torch.empty(len(itos_pasts), args.code_number).uniform_(0.000001, 1)
encoding_logits.requires_grad = True
optimizer = torch.optim.Adam([encoding_logits], lr= 0.01) # also try 0.001, 0.0001
softmax = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()



def runOCE():
    objective = 10000000
    for t in range(100000):
       print("Iteration", t)
       optimizer.zero_grad()
    
       encoding = softmax(encoding_logits)
       marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), encoding).squeeze(0)
       log_marginal_hidden = logWithoutNA(marginal_hidden)
       normalizedEncoding = encoding.div(marginal_hidden.unsqueeze(0))
       decoding = torch.matmul((marginal_past.unsqueeze(1) * future_given_past).t(), normalizedEncoding).t()
       print(decoding[0].sum())
    
       logEncoding = logsoftmax(encoding_logits)
       logDecoding = logWithoutNA(decoding)
       miWithFuture = torch.sum((decoding * (logDecoding - logFutureMarginal.unsqueeze(0))).sum(1) * marginal_hidden) # WRONG
       miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
       assert miWithFuture <= miWithPast + 1e-5, (miWithFuture , miWithPast)
       newObjective = args.beta * miWithPast - miWithFuture
       newObjective.backward()
       optimizer.step()
     
       print(["Mi with future", miWithFuture, "Mi with past", miWithPast])
       print(["objectives","last",objective, "new", newObjective])
       #assert newObjective - 0.1 <= objective, (newObjective, objective)
       if abs(newObjective - objective) < 1e-13:
         print("Ending")
         break
       objective = newObjective
    miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
    assert miWithFuture <= miWithPast + 1e-5, (miWithFuture , miWithPast)

    return encoding, decoding, logDecoding, miWithPast, log_marginal_hidden

encoding, decoding, logDecoding, miWithPast_train, log_marginal_hidden = runOCE()

futureSurprisal_train = -((future_given_past * marginal_past.unsqueeze(1)).unsqueeze(1) * encoding.unsqueeze(2) * logDecoding.unsqueeze(0)).sum()


#assert False, "how is the vocabulary for held-out data generated????"
# try on held-out data

ngrams = {}

lastPosUni = ("EOS",)*(args.horizon-1)
for _ in range(1000):
 sentence = sample("S")
 for line in sentence:
   nextPosUni = line
   ngram = process(lastPosUni+(nextPosUni,))
   ngrams[ngram] = ngrams.get(ngram, 0) + 1
   lastPosUni = lastPosUni[1:]+(nextPosUni,)
 nextPosUni = "EOS"
 ngram = process(lastPosUni+(nextPosUni,))
 ngrams[ngram] = ngrams.get(ngram, 0) + 1
 lastPosUni = lastPosUni[1:]+(nextPosUni,)


#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


ngrams = list(ngrams.iteritems())
ngrams = sorted(ngrams, key=lambda x:x[1])

#ngrams = [x for x in ngrams if x[1] > 100]
#print(ngrams)
#print(["Number of ngrams", len(ngrams)])


keys = [x[0] for x in ngrams]

total = sum([x[1] for x in ngrams])



frequencies = [x[1] for x in ngrams]

pasts = [x[:-1] for x in keys]  #range(horizon:range(horizon, 2*horizon)]
futures = [x[-1:] for x in keys]



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

marginal_future = torch.zeros(len(itos_futures))
for i in range(len(futures)):
   marginal_future[futures_int[i]] += frequencies[i]
marginal_future = marginal_future.div(marginal_future.sum())


marginal_hidden = torch.matmul(marginal_past.unsqueeze(0), encoding).squeeze(0)


logDecoding = logWithoutNA(decoding) 
logFutureMarginal = logWithoutNA(marginal_future)

futureSurprisal = -((future_given_past * marginal_past.unsqueeze(1)).unsqueeze(1) * encoding.unsqueeze(2) * logDecoding.unsqueeze(0)).sum()

for i in range(len(pasts)):
   print(pasts[i], futures[i], (encoding[pasts_int[i]] * logDecoding[:,futures_int[i]]).sum())
print(futureSurprisal.size())
quit()





logEncoding = logWithoutNA(encoding)

miWithPast = torch.sum((encoding * (logEncoding - log_marginal_hidden.unsqueeze(0))).sum(1) * marginal_past)
print(["Mi with past", miWithPast, "Future Surprisal", futureSurprisal/args.horizon, "Horizon", args.horizon, "but the comparison with longer blocks isn't really fair"]) # "Mi with future", miWithFuture


myID = random.randint(0,10000000)


outpath = "../../results/outputs-oce/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+".txt"
#with sys.stdout as outFile: #open(outpath, "w") as outFile:
outFile = sys.stdout
if True:
    print >> outFile, "\t".join(x+" "+str(getattr(args,x)) for x in args_names)
    print >> outFile, "Mi With Past", float(miWithPast)
    print >> outFile, "Future Surprisal", float(futureSurprisal)
    print >> outFile, "MI with Past (train)", float(miWithPast_train)
    print >> outFile, float(futureSurprisal_train)


print(outpath)


