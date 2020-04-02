print("Character aware!")

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language1", type=str, default="english")
parser.add_argument("--language2", type=str, default="german")

parser.add_argument("--load_from", type=str, default=None) # 8066636


import random

parser.add_argument("--batchSize", type=int, default=random.choice([128]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
parser.add_argument("--char_dec_hidden_dim", type=int, default=128)

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

################################

char_vocab_path_1 = "vocabularies/"+args.language1.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path_1, "r") as inFile:
     itos_1 = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi_1 = dict([(itos_1[i],i) for i in range(len(itos_1))])

with open("vocabularies/char-vocab-wiki-"+args.language1, "r") as inFile:
     itos_chars_1 = [x for x in inFile.read().strip().split("\n")]
stoi_chars_1 = dict([(itos_chars_1[i],i) for i in range(len(itos_chars_1))])


itos_chars_total_1 = ["SOS", "EOS", "OOV"] + itos_chars_1

################################

char_vocab_path_2 = "vocabularies/"+args.language2.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path_2, "r") as inFile:
     itos_2 = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi_2 = dict([(itos_2[i],i) for i in range(len(itos_2))])

with open("vocabularies/char-vocab-wiki-"+args.language1, "r") as inFile:
     itos_chars_2 = [x for x in inFile.read().strip().split("\n")]
stoi_chars_2 = dict([(itos_chars_2[i],i) for i in range(len(itos_chars_2))])


itos_chars_total_2 = ["SOS", "EOS", "OOV"] + itos_chars_2


################################

vocab_size = 50000
outVocabSize = 2*(3+vocab_size)


itos_total = ["EOS", "OOV", "SOS"] + itos_1 + ["EOS", "OOV", "SOS"] + itos_2 
assert len(itos_total) == outVocabSize



import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop


rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = rnn #WeightDrop(rnn, layer_names=[(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output_1 = torch.nn.Linear(args.hidden_dim,50003).cuda()
output_2 = torch.nn.Linear(args.hidden_dim,50003).cuda()


word_embeddings = torch.nn.Embedding(num_embeddings=3+len(itos_1)+3+len(itos_2), embedding_dim=args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')

modules = [rnn, output_1, output_2, word_embeddings]


character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total_1)+ len(itos_chars_total_2), embedding_dim=args.char_emb_dim).cuda()


char_composition = torch.nn.LSTM(args.char_emb_dim, args.char_enc_hidden_dim, 1, bidirectional=True).cuda()
char_composition_output = torch.nn.Linear(2*args.char_enc_hidden_dim, args.word_embedding_size).cuda()

#char_decoder_rnn = torch.nn.LSTM(args.char_emb_dim + args.hidden_dim, args.char_dec_hidden_dim, 1).cuda()
#char_decoder_output = torch.nn.Linear(args.char_dec_hidden_dim, len(itos_chars_total))


modules += [character_embeddings, char_composition, char_composition_output] #, char_decoder_rnn, char_decoder_output]
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

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout





def prepareDatasetChunksTwo(data1, data2, train=True):
    c1 = prepareDatasetChunks(data1, train=train, batchSizeHere=int(args.batchSize/2), stoi=stoi_1, stoi_chars=stoi_chars_1)
    c2 = prepareDatasetChunks(data2, train=train, batchSizeHere=int(args.batchSize/2), stoi=stoi_2, stoi_chars=stoi_chars_2)


    while True:
       try:
         numerified1, numerified_chars1 = next(c1)
         numerified2, numerified_chars2 = next(c2)
       except StopIteration:
         return
       numerified = torch.cat([numerified1, numerified2], dim=1)
       numerified_chars = torch.cat([numerified_chars1, numerified_chars2], dim=1)
       yield numerified, numerified_chars


def prepareDatasetChunks(data, batchSizeHere=args.batchSize, train=True, stoi=None, stoi_chars=None):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      numerified_chars = []
      for chunk in data:
       for char in chunk:
         count += 1
         numerified.append((stoi[char]+3 if char in stoi else 2))
         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])

       if len(numerified) > (batchSizeHere*args.sequence_length):
         sequenceLengthHere = args.sequence_length

         cutoff = int(len(numerified)/(batchSizeHere*sequenceLengthHere)) * (batchSizeHere*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerifiedCurrent_chars = numerified_chars[:cutoff]

         for i in range(len(numerifiedCurrent_chars)):
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i][:15] + [1]
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i] + ([0]*(16-len(numerifiedCurrent_chars[i])))


         numerified = numerified[cutoff:]
         numerified_chars = numerified_chars[cutoff:]
       
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(batchSizeHere, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(batchSizeHere, -1, sequenceLengthHere, 16).transpose(0,1).transpose(1,2).cuda()

         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], numerifiedCurrent_chars[i]
         hidden = None
       else:
         print("Skipping")



hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())




def forward(numeric, train=True, printHere=False):
      global hidden
      global beginning
      global beginning_chars
      if hidden is not None:
          hidden1 = Variable(hidden[0]).detach()
          hidden2 = Variable(hidden[1]).detach()
          forRestart = bernoulli.sample()
          hidden1 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden1)
          hidden2 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden2)
          hidden = (hidden1, hidden2)
          beginning = torch.where(forRestart.unsqueeze(0) == 1, zeroBeginning, beginning)
          beginning_chars = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroBeginning_chars, beginning_chars)
      elif hidden is None:
          hidden = None
          beginning = zeroBeginning
          beginning_chars = zeroBeginning_chars




      numeric, numeric_chars = numeric
#      print(numeric_chars.size())
      numeric = torch.cat([beginning, numeric], dim=0)

      numeric_chars = torch.cat([beginning_chars, numeric_chars], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)
      beginning_chars = numeric_chars[numeric_chars.size()[0]-1].view(1, args.batchSize, 16)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_chars = Variable(numeric_chars[:-1], requires_grad=False)
      input_tensor_chars[:, int(args.batchSize/2):] = input_tensor_chars[:, int(args.batchSize/2):] + len(itos_chars_total_1)

      # TODO it is weird (BUG) that this is only done for characters, not for words

      embedded_chars = input_tensor_chars.transpose(0,2).transpose(2,1)
      embedded_chars = embedded_chars.contiguous().view(16, -1)
      _, embedded_chars = char_composition(character_embeddings(embedded_chars), None)
      embedded_chars = embedded_chars[0].view(2, args.sequence_length, args.batchSize, args.char_enc_hidden_dim)
      #print(embedded_chars.size())

      embedded_chars = char_composition_output(torch.cat([embedded_chars[0], embedded_chars[1]], dim=2))

      embedded = word_embeddings(input_tensor)
      embedded = torch.cat([embedded, embedded_chars], dim=2)
      if train:
         embedded = char_dropout(embedded)
         mask = bernoulli_input.sample()
         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
         embedded = embedded * mask

      out, hidden = rnn_drop(embedded, hidden)
#      if train:
#          out = dropout(out)


      if train:
        mask = bernoulli_output.sample()
        mask = mask.view(1, args.batchSize, args.hidden_dim)
        out = out * mask

      word_logits1 = output_1(out[:, :int(args.batchSize/2)])
      word_logits2 = output_2(out[:, int(args.batchSize/2):])
      logits = torch.cat([word_logits1, word_logits2], dim=1)



      log_probs = logsoftmax(logits)

      
      loss = train_loss(log_probs.view(-1, 50003), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, 50003), target_tensor.view(-1)).view(-1, args.batchSize)
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
#         print(("NONE", itos[numericCPU[0][0]-3]))
         for i in range((args.sequence_length)):
#            print((losses[i][0], itos[numericCPU[i+1][0]-3]))
            j2 = int(args.batchSize/2)
            print (i, itos_1[numericCPU[i+1][0]-3], losses[i][0], "\t\t\t", itos_2[numericCPU[i+1][j2] - 3], losses[i][j2])

      return loss, target_tensor.view(-1).size()[0]

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(counter, printHere, loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
      optim.step()


lossHasBeenBad = 0

import time

totalStartTime = time.time()


devLosses = []
for epoch in range(10000):
   print(epoch)
   training_data_1 = corpusIteratorWikiWords.training(args.language1)
   training_data_2 = corpusIteratorWikiWords.training(args.language2)
   print("Got data")


   training_chars = prepareDatasetChunksTwo(training_data_1, training_data_2, train=True)




   rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   while True:
      counter += 1
      try:
         numeric = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      if loss.data.cpu().numpy() > 15.0:
          lossHasBeenBad += 1
      else:
          lossHasBeenBad = 0
      if lossHasBeenBad > 100:
          print("Loss exploding, has been bad for a while")
          print(loss)
          quit()
      trainChars += charCounts 
      if printHere:
          print(("Loss here", loss))
          print((epoch,counter))
          print("Dev losses")
          print(devLosses)
          print("Words per sec "+str(trainChars/(time.time()-startTime)))
          print(learning_rate)
          print(__file__)
          print(args)
      if counter % 20000 == 0: # and epoch == 0:
        state = {"arguments" : str(args), "words_1" : itos_1, "words_2" : itos_2, "components" : [c.state_dict() for c in modules]}
        torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language1+"AND"+args.language2+"_"+__file__+"_code_"+str(args.myID)+".txt")

      if (time.time() - totalStartTime)/60 > 4000:
          print("Breaking early to get some result within 72 hours")
          totalStartTime = time.time()
          break

 #     break
   rnn_drop.train(False)


   dev_data_1 = corpusIteratorWikiWords.dev(args.language1)
   dev_data_2 = corpusIteratorWikiWords.dev(args.language2)

   print("Got data")

   dev_chars = prepareDatasetChunksTwo(dev_data_1, dev_data_2, train=False)



     
   dev_loss = 0
   dev_char_count = 0
   counter = 0
   hidden, beginning = None, None
   while True:
       counter += 1
       try:
          numeric = next(dev_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)

   with open("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+args.language1+"AND"+args.language2+"_"+__file__+"_model_"+str(args.myID)+"_"+model+".txt", "w") as outFile:
       print(str(args), file=outFile)
       print(" ".join([str(x) for x in devLosses]), file=outFile)

   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break

   state = {"arguments" : str(args), "words_1" : itos_1, "words_2" : itos_2, "components" : [c.state_dict() for c in modules]}
   torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language1+"AND"+args.language2+"_"+__file__+"_code_"+str(args.myID)+".txt")






   learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
   optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9


