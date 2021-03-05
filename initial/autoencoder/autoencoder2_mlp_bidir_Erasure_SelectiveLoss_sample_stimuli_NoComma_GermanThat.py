print("Character aware!")


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="german")
parser.add_argument("--load-from", dest="load_from", type=str, default=188436350)
#CAN ALSO TAKE myID=878921872,
# OTHER MODEL: 247340595
#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([500]))
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
  try:
     checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__.replace("_sample_stimuli_NoComma_GermanThat", "_WithoutComma")+"_code_"+str(args.load_from)+".txt")
  except FileNotFoundError:
     assert False
  for i in range(len(checkpoint["components"])):
      modules[i].load_state_dict(checkpoint["components"][i])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

relu = torch.nn.ReLU()






hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * 2 * args.hidden_dim)]).cuda())




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


#      numeric_noised = [[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]
      numeric_noised = [[x if itos[int(x)-3] not in ["der", "dass"] else 0 for x in y] for y in numeric.cpu().t()]


      numeric_noised = torch.LongTensor([[0 for _ in range(args.sequence_length-len(y))] + y for y in numeric_noised]).cuda().t()

      numeric = torch.cat([beginning, numeric], dim=0)
      numeric_noised = torch.cat([beginning, numeric_noised], dim=0)

      numeric_onlyNoisedOnes = torch.where(numeric_noised == 0, numeric, 0*numeric) # target is 0 in those places where no noise has happened

      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)


      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = word_embeddings(input_tensor)

      embedded_noised = word_embeddings(input_tensor_noised)

      out_encoder, _ = rnn_encoder(embedded_noised, None)

      out_decoder, _ = rnn_decoder(embedded, None)

      attention = torch.bmm(attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
      attention = attention_softmax(attention).transpose(0,1)
      from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      out_full = torch.cat([out_decoder, from_encoder], dim=2)


      if train:
        mask = bernoulli_output.sample()
        mask = mask.view(1, args.batchSize, 2*args.hidden_dim)
        out_full = out_full * mask



      logits = output(relu(output_mlp(out_full) ))
      log_probs = logsoftmax(logits)

      
      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy()

         print(("NONE", itos_total[numericCPU[0][0]]))
         for i in range((args.sequence_length)):
            print((losses[i][0], itos_total[numericCPU[i+1][0]], itos_total[numeric_noisedCPU[i+1][0]]))






      hidden = None
      result  = ["" for _ in range(args.batchSize)]
      embeddedLast = embedded[0].unsqueeze(0)
      for i in range(args.sequence_length):
          out_decoder, hidden = rnn_decoder(embeddedLast, hidden)
    
          attention = torch.bmm(attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)

          print(input_tensor.size())


          logits = output(relu(output_mlp(out_full) )) 
          probs = softmax(logits)

#          print(probs.size(), probs.sum(dim=2))
 #         quit()

          dist = torch.distributions.Categorical(probs=probs)
       
          nextWord = torch.where(numeric_noised[i] == 0, (dist.sample()), numeric[i:i+1])
          print(nextWord.size())
          nextWordStrings = [itos_total[x] for x in nextWord.cpu().numpy()[0]]
          for j in range(args.batchSize):
             result[j] += " "+nextWordStrings[j]
          embeddedLast = word_embeddings(nextWord)
          print(embeddedLast.size())
      containsIfWhen = {False : 0, True : 0}
      for r in result:
         rb = " dass " in r 
         containsIfWhen[rb] += 1
         print(r+"\t"+str(rb))
      print(containsIfWhen[True] / (containsIfWhen[True] + containsIfWhen[False]))
      return result




results = []
if True:
   PREFIX = "nachher kam die frau dazu . dies trat auf denn sie schlug vor den mann mit einer medizin zu behandeln aber dazu kam es nicht . "

   critical = "die behauptung dass der polizist den der kunde kannte" # 0.316 SCnone
   sentence= PREFIX + critical
   numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")][-args.sequence_length:]
   print(len(numerified))
   assert len(numerified) == args.sequence_length, (len(numerified), args.sequence_length)
   numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
   result = forward(numerified, train=False)
#   print(result)

