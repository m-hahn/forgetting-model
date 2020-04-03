print("Character aware!")


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from", dest="load_from", type=str, default="264073608") # also 449431785
#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([128]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([512]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([30]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
parser.add_argument("--char_dec_hidden_dim", type=int, default=128)


parser.add_argument("--deletion_rate", type=float, default=0.2)


model = "REAL_REAL"

import math

args=parser.parse_args()

SAMPLES_PER_BATCH = 20


#if "MYID" in args.save_to:
#   args.save_to = args.save_to.replace("MYID", str(args.myID))

#assert "word" in args.save_to, args.save_to

print(args)



import corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

char_vocab_path = "vocabularies/"+args.language.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


itos_total = ["<SOS>", "<EOS>", "OOV"] + itos
stoi_total = dict([(itos_total[i],i) for i in range(len(itos_total))])


with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
     itos_chars = [x for x in inFile.read().strip().split("\n")]
stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])


itos_chars_total = ["<SOS>", "<EOS>", "OOV"] + itos_chars


import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop


rnn_encoder = torch.nn.LSTM(2*args.word_embedding_size, int(args.hidden_dim/2.0), args.layer_num, bidirectional=True).cuda()
rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()




output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)
softmax = torch.nn.Softmax(dim=2)

attention_softmax = torch.nn.Softmax(dim=1)


train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')


attention_proj = torch.nn.Linear(args.hidden_dim, args.hidden_dim, bias=False).cuda()
#attention_layer = torch.nn.Bilinear(args.hidden_dim, args.hidden_dim, 1, bias=False).cuda()
attention_proj.weight.data.fill_(0)


output_mlp = torch.nn.Linear(2*args.hidden_dim, args.hidden_dim).cuda()

modules = [rnn_decoder, rnn_encoder, output, word_embeddings, attention_proj, output_mlp]


#character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total)+3, embedding_dim=args.char_emb_dim).cuda()
#
#char_composition = torch.nn.LSTM(args.char_emb_dim, args.char_enc_hidden_dim, 1, bidirectional=True).cuda()
#char_composition_output = torch.nn.Linear(2*args.char_enc_hidden_dim, args.word_embedding_size).cuda()
#
#char_decoder_rnn = torch.nn.LSTM(args.char_emb_dim + args.hidden_dim, args.char_dec_hidden_dim, 1).cuda()
#char_decoder_output = torch.nn.Linear(args.char_dec_hidden_dim, len(itos_chars_total))
#
#
#modules += [character_embeddings, char_composition, char_composition_output, char_decoder_rnn, char_decoder_output]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

#named_modules = {"rnn" : rnn, "output" : output, "word_embeddings" : word_embeddings, "optim" : optim}

#if args.load_from is not None:
#  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
#  for name, module in named_modules.items():
 #     module.load_state_dict(checkpoint[name])
if args.load_from is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__.replace("_constructStimulusSentences", "")+"_code_"+str(args.load_from)+".txt")
  for i in range(len(checkpoint["components"])):
      modules[i].load_state_dict(checkpoint["components"][i])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

relu = torch.nn.ReLU()






hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize*SAMPLES_PER_BATCH)]).cuda().view(1,args.batchSize*SAMPLES_PER_BATCH)
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize*SAMPLES_PER_BATCH, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize*SAMPLES_PER_BATCH, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize*SAMPLES_PER_BATCH)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * SAMPLES_PER_BATCH * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * SAMPLES_PER_BATCH * 2 * args.hidden_dim)]).cuda())




posDict = {}
with open("/u/scr/mhahn/FAIR18/english-wiki-word-vocab_POS.txt", "r") as dictIn:
    for line in dictIn:
        line = line.strip().split("\t")
        line[2] = int(line[2])
        if line[2] < 100:
           print(len(posDict))
           break
        if line[0] not in posDict:
           posDict[line[0]] = {}
        if line[1] not in posDict[line[0]]:
            posDict[line[0]][line[1]] = 0
        posDict[line[0]][line[1]] += 1

posDictMax = {}
for word, entry in posDict.items():
    maxCount = max([entry[x] for x in entry])
    bestPOS = [x for x in entry if entry[x] == maxCount][0]
    posDictMax[word] = bestPOS
        



def forward(numeric, train=True, printHere=True):
      global beginning
      global beginning_chars
      if True:
          beginning = zeroBeginning
          beginning_chars = zeroBeginning_chars


      numeric_noised = [[x for x in y if random.random() > args.deletion_rate or itos_total[int(x)] == "."] for y in numeric.cpu().t()]
#      numeric_noised = [[x for x in y if itos_total[int(x)] not in ["that"]] for y in numeric.cpu().t()]

      numeric_noised = torch.LongTensor([[0 for _ in range(args.sequence_length-len(y))] + y for y in numeric_noised]).cuda().t().unsqueeze(2).expand(-1, -1, SAMPLES_PER_BATCH).contiguous().view(-1, args.batchSize*SAMPLES_PER_BATCH)

      print(numeric.size())
      numeric = numeric.unsqueeze(2).expand(-1, -1, SAMPLES_PER_BATCH).contiguous().view(-1, args.batchSize*SAMPLES_PER_BATCH)

      numeric = torch.cat([beginning, numeric], dim=0)
      numeric_noised = torch.cat([beginning, numeric_noised], dim=0)

      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = word_embeddings(input_tensor)

      embedded_noised = word_embeddings(input_tensor_noised)

      out_encoder, _ = rnn_encoder(embedded_noised, None)

#      out_decoder, _ = rnn_decoder(embedded, None)

#      attention = torch.bmm(attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
 #     attention = attention_softmax(attention).transpose(0,1)
  #    from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
   #   out_full = torch.cat([out_decoder, from_encoder], dim=2)


#      if train:
 #       mask = bernoulli_output.sample()
  #      mask = mask.view(1, args.batchSize*SAMPLES_PER_BATCH, 2*args.hidden_dim)
   #     out_full = out_full * mask



#      logits = output(relu(output_mlp(out_full) ))
#      log_probs = logsoftmax(logits)
#
#      
#      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))
#
#      if printHere:
#         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize*SAMPLES_PER_BATCH)
#         losses = lossTensor.data.cpu().numpy()
#         numericCPU = numeric.cpu().data.numpy()
#         numeric_noisedCPU = numeric_noised.cpu().data.numpy()

#         print(("NONE", itos_total[numericCPU[0][0]]))
 #        for i in range((args.sequence_length)):
  #          print((losses[i][0], itos_total[numericCPU[i+1][0]], itos_total[numeric_noisedCPU[i+1][0]]))






      hidden = None
      result  = ["" for _ in range(args.batchSize*SAMPLES_PER_BATCH)]
      embeddedLast = embedded[0].unsqueeze(0)
      for i in range(args.sequence_length):
          out_decoder, hidden = rnn_decoder(embeddedLast, hidden)
    
          attention = torch.bmm(attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)

#          print(input_tensor.size())


          logits = output(relu(output_mlp(out_full) )) 
          probs = softmax(logits)

#          print(probs.size(), probs.sum(dim=2))
 #         quit()

          dist = torch.distributions.Categorical(probs=probs)
       
          nextWord = (dist.sample())
#          print(nextWord.size())
          nextWordStrings = [itos_total[x] for x in nextWord.cpu().numpy()[0]]
          for i in range(args.batchSize*SAMPLES_PER_BATCH):
             result[i] += " "+nextWordStrings[i]
          embeddedLast = word_embeddings(nextWord)
          #print(embeddedLast.size())
      for r in result:
         print(r)
      resultNums = {}
      resultNums["nounRecovered"] = (float(len([x for x in result if NOUN in x]))/len(result))
      resultNums["nounThatRecovered"] = (float(len([x for x in result if NOUN+" that" in x]))/len(result))
      nextWords = []
      for res in result:
        try:
          start = res.index(NOUN)
          end = res.index(" ", start+len(NOUN)+2)
          nextWords.append(res[start+len(NOUN)+1:end])
        except ValueError:
          _ = 0
      nextWordsDict = {posDictMax.get(x, x) : 0 for x in nextWords}
      for x in nextWords:
        nextWordsDict[posDictMax.get(x, x)] += 1
      nextWords = sorted(list(nextWordsDict.items()), key=lambda x:-x[1])

      resultNums["NextWords"] = nextWords
      return resultNums, result


nounsAndVerbs = []
nounsAndVerbs.append(["the school principal",       "the teacher",        "had an affair with",                     "had been fired",                     "was quoted in the newspaper"])
nounsAndVerbs.append(["the famous sculptor",        "the painter",        "admired more than anyone",            "wasn't talented",                    "was completely untrue"])
nounsAndVerbs.append(["the marketing whiz",  "the artist",         "had hired",                  "was a fraud",                        "shocked everyone"])
nounsAndVerbs.append(["the marathon runner",         "the psychiatrist",       "treated for his illness",                "was actually doping",            "was ridiculous"])
nounsAndVerbs.append(["the frightened child",           "the medic",          "rescued from the flood",    "was completely unharmed",            "relieved everyone"])
nounsAndVerbs.append(["the alleged criminal",        "the officer",        "arrested after the murder",                  "was not in fact guilty",             "was bogus"])
nounsAndVerbs.append(["the college student",         "the professor",      "accused of cheating",                     "was dropping the class",             "made the professor happy"])
nounsAndVerbs.append(["the suspected mobster",         "the media",          "portrayed in detail",               "was on the run",                     "turned out to be true"])
nounsAndVerbs.append(["the leading man",           "the starlet",        "fell in love with",                    "would miss the show",                "almost made her cry"])
nounsAndVerbs.append(["the old preacher",        "the parishioners",   "fired yesterday",                     "stole money from the church",        "proved to be true"])
nounsAndVerbs.append(["the young violinist",      "the sponsors",       "backed financially",                    "abused drugs",                       "is likely true"])
nounsAndVerbs.append(["the conservative senator",        "the diplomat",       "opposed in the election",                   "won in the run-off",                   "really made him angry"])
#nounsAndVerbs = _.shuffle(nounsAndVerbs)

topNouns = []
topNouns.append("report")
topNouns.append("story")       
#topNouns.append("disclosure")
topNouns.append("proof")
topNouns.append("confirmation")  
topNouns.append("information")
topNouns.append("evidence")
topNouns.append("reminder")
topNouns.append("rumor")
#topNouns.append("thought")
topNouns.append("suggestion")
topNouns.append( "revelation")  
topNouns.append( "belief")
topNouns.append( "fact")
topNouns.append( "realization")
topNouns.append( "suspicion")
topNouns.append( "certainty")
topNouns.append( "idea")
topNouns.append( "admission") 
topNouns.append( "confirmation")
topNouns.append( "complaint"    )
topNouns.append( "certainty"   )
topNouns.append( "prediction"  )
topNouns.append( "declaration")
topNouns.append( "proof"   )
topNouns.append( "suspicion")    
topNouns.append( "allegation"   )
topNouns.append( "revelation"   )
topNouns.append( "realization")
topNouns.append( "news")
topNouns.append( "opinion" )
topNouns.append( "idea")
topNouns.append("myth")

with open("../forgetting/fromCorpus_counts.csv", "r") as inFile:
   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = counts[0]
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[0] : line[1:] for line in counts}

topNouns = sorted(list(set(topNouns)), key=lambda x:float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]]))

results = []
with open("temporary-stimuli/stimuli1.txt", "w") as outFile:
  for NOUN in topNouns:
   for sentenceList in nounsAndVerbs:
     print(sentenceList)
     context = ", the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
     for condition in [0,2]:
        if condition == 0:
           sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} {sentenceList[3]} {sentenceList[4]}"
        elif condition == 2:
           sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} {sentenceList[4]}"
        else:
           assert False
   #     print(sentence)
  #      continue
        numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
        print(len(numerified))
        numerified = numerified[-args.sequence_length:]
        assert len(numerified) == args.sequence_length
        numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
        print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
        print("===========")
        _, result = forward(numerified, train=False)
        results.append((NOUN, sentenceList[0], condition, result))
        for index, imputed in enumerate(result):
           print(f'{NOUN} {sentenceList[0].replace(" ","_")} {int(index / SAMPLES_PER_BATCH)} {condition} {imputed}', file=outFile)
  
