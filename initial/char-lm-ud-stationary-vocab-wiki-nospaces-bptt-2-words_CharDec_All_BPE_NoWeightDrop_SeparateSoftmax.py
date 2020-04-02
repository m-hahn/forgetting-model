print("Character aware!")

# Character-aware version of the `Tabula Rasa' language model

import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
#parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--save-to", dest="save_to", type=str)

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
parser.add_argument("--char_dec_hidden_dim", type=int, default=256)

model = "REAL_REAL"

import math

args=parser.parse_args()

#if "MYID" in args.save_to:
#   args.save_to = args.save_to.replace("MYID", str(args.myID))

#assert "word" in args.save_to, args.save_to

print(args)

CHAR_WORD_LENGTH = 3

import corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


char_vocab_path = f"/u/scr/mhahn/FAIR18/{args.language.lower()}-wiki-word-vocab.txt"
bpe_char_vocab_path = f"/u/scr/mhahn/FAIR18/{args.language.lower()}-wiki-word-vocab_BPE_50000_Parsed.txt"

itos = [None for _ in range(1000000)]
i2BPE = [None for _ in range(1000000)]
with open(char_vocab_path, "r") as inFile:
  with open(bpe_char_vocab_path, "r") as inFileBPE:
     for i in range(1000000):
        if i % 50000 == 0:
           print(i)
        word = next(inFile).strip().split("\t")
        bpe = next(inFileBPE).strip().split("\t")
        itos[i] = word[0]
        i2BPE[i] = bpe[0].split("@@ ")
stoi = dict([(itos[i],i) for i in range(len(itos))])

with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
     itos_chars = [x for x in inFile.read().strip().split("\n")]
with open("/u/scr/mhahn/FAIR18/german-wiki-word-vocab_BPE_50000.txt", "r") as inFile:
     itos_chars += [x.replace(" ",  "") for x in inFile.read().strip().split("\n")]
assert len(itos_chars) > 50000
stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])

itos_chars_total = ["SOS", "EOS", "OOV"] + itos_chars


import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop


rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = rnn #WeightDrop(rnn, layer_names=[(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

#word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='none')

modules = [rnn, output] #, word_embeddings]


character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total)+3, embedding_dim=args.char_emb_dim).cuda()

char_composition = torch.nn.LSTM(args.char_emb_dim, args.char_enc_hidden_dim, 1, bidirectional=True).cuda()
char_composition_output = torch.nn.Linear(2*args.char_enc_hidden_dim, 2*args.word_embedding_size).cuda()

char_decoder_rnn = torch.nn.LSTM(args.char_emb_dim + args.hidden_dim, args.char_dec_hidden_dim, 1).cuda()
char_decoder_output = torch.nn.Linear(args.char_dec_hidden_dim, len(itos_chars_total)).cuda()


modules += [character_embeddings, char_composition, char_composition_output, char_decoder_rnn, char_decoder_output]
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



def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      numerified_chars = []
      word_lengths = []
      bpe_lengths = []

      for chunk in data:
       #print(len(chunk))
       for char in chunk:
#         if char == " ":
 #          continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         numerified.append((stoi[char]+3 if char in stoi else 2))
         bpeRep = ([stoi_chars[x]+3 if x in stoi_chars else 2 for x in i2BPE[stoi[char]]] if char in stoi else [2,1])
         numerified_chars.append([0] + bpeRep)
         bpe_lengths.append(min(CHAR_WORD_LENGTH-1, len(bpeRep)))
         word_lengths.append(len(char))

       if len(numerified) > (args.batchSize*args.sequence_length):
         sequenceLengthHere = args.sequence_length

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerifiedCurrent_chars = numerified_chars[:cutoff]
         word_lengthsCurrent = word_lengths[:cutoff]
         bpe_lengthsCurrent = bpe_lengths[:cutoff]

         for i in range(len(numerifiedCurrent_chars)):
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i][:CHAR_WORD_LENGTH+1]
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i] + ([0]*(CHAR_WORD_LENGTH+1-len(numerifiedCurrent_chars[i])))


         numerified = numerified[cutoff:]
         numerified_chars = numerified_chars[cutoff:]
         word_lengths = word_lengths[cutoff:]
         bpe_lengths = bpe_lengths[cutoff:]

         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(args.batchSize, -1, sequenceLengthHere, CHAR_WORD_LENGTH+1).transpose(0,1).transpose(1,2).cuda()
         word_lengthsCurrent = torch.LongTensor(word_lengthsCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         bpe_lengthsCurrent = torch.LongTensor(bpe_lengthsCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()

#         print(numerifiedCurrent_chars.size())
 #        quit()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], numerifiedCurrent_chars[i], word_lengthsCurrent[i], bpe_lengthsCurrent[i]
         hidden = None
       else:
         print("Skipping")





hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize, CHAR_WORD_LENGTH+1).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())




def forward(numeric, train=True, printHere=False):
      global hidden
      global beginning
      global beginning_chars
      if hidden is None:
          hidden = None
          beginning = zeroBeginning
          beginning_chars = zeroBeginning_chars
      elif hidden is not None:
          hidden1 = Variable(hidden[0]).detach()
          hidden2 = Variable(hidden[1]).detach()
          forRestart = bernoulli.sample()
          hidden1 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden1)
          hidden2 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden2)
          hidden = (hidden1, hidden2)
          beginning = torch.where(forRestart.unsqueeze(0) == 1, zeroBeginning, beginning)
          beginning_chars = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroBeginning_chars, beginning_chars)




      numeric, numeric_chars, wordLengths, bpeLengths = numeric
      #print(bpeLengths.max(), bpeLengths.float().mean())
#      print(numeric_chars.size())
      numeric = torch.cat([beginning, numeric], dim=0)

      numeric_chars = torch.cat([beginning_chars, numeric_chars], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)
      beginning_chars = numeric_chars[numeric_chars.size()[0]-1].view(1, args.batchSize, CHAR_WORD_LENGTH+1)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_chars = Variable(numeric_chars[:-1], requires_grad=False)
      target_tensor_chars = Variable(numeric_chars[1:], requires_grad=False)

      embedded_chars = input_tensor_chars.transpose(0,2).transpose(2,1)
      embedded_chars = embedded_chars.contiguous().view(CHAR_WORD_LENGTH+1, -1)
      _, embedded_chars = char_composition(character_embeddings(embedded_chars), None)
      embedded_chars = embedded_chars[0].view(2, args.sequence_length, args.batchSize, args.char_enc_hidden_dim)
      #print(embedded_chars.size())

      embedded_chars = char_composition_output(torch.cat([embedded_chars[0], embedded_chars[1]], dim=2))
      #print(embedded_chars.size())

    #  print(word_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(word_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #word_embeddings(input_tensor)
      #else:
#      embedded = word_embeddings(input_tensor)
      #print(embedded.size())
#      print("=========")
#      print(numeric[:,5])
#      print(embedded[:,5,:].mean(dim=1)[numeric[:-1,5] == 3])
#      print(embedded_chars[:,5,:].mean(dim=1)[numeric[:-1,5] == 3])
      embedded = embedded_chars #torch.cat([embedded, embedded_chars], dim=2)
      #print(embedded.size())
      if train:
         embedded = char_dropout(embedded)
         mask = bernoulli_input.sample()
         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
         embedded = embedded * mask

      out, hidden = rnn_drop(embedded, hidden)


      if train:
        mask = bernoulli_output.sample()
        mask = mask.view(1, args.batchSize, args.hidden_dim)
        out = out * mask


    #  print(out.size())
     # print(target_tensor_chars.size())
      out_flattened = out.view(-1, args.hidden_dim)
      target_tensor_chars_flattened = target_tensor_chars.view(args.sequence_length * args.batchSize, CHAR_WORD_LENGTH+1)
      #
      out_relevant = out_flattened#[oovs]
      target_tensor_chars_relevant = target_tensor_chars_flattened.t() #[oovs].t()
   #   print(out_relevant.size())
  #    print(character_embeddings(target_tensor_chars_relevant[:-1]).size())


#      print(target_tensor_chars_relevant.size(), out_relevant.size(), character_embeddings(target_tensor_chars_relevant[0])

      inp_to_char_decoder = torch.cat([character_embeddings(target_tensor_chars_relevant[:1]), out_relevant.unsqueeze(0)], dim=2)
      out_chars, hidden_char_decoder = char_decoder_rnn(inp_to_char_decoder)
      out_chars = logsoftmax(char_decoder_output(out_chars))
      loss_chars_FIRST = train_loss_chars(out_chars.view(-1, len(itos_chars_total)), target_tensor_chars_relevant[1].contiguous().view(-1)).view(1, -1, args.batchSize)
#      print(bpeLengths)
      loss_chars_SECOND = 0
      if bpeLengths.max() > 1:
         bpeLengthsFlattened = bpeLengths.view(-1)

         target_tensor_chars_relevant_long = target_tensor_chars_relevant[1:, bpeLengthsFlattened > 1]
       #  print(target_tensor_chars_relevant_long.size())

      #   print(target_tensor_chars_relevant, CHAR_WORD_LENGTH)
         out_relevant_long = out_relevant[bpeLengthsFlattened > 1]
     #    print("OUT_RELEVANT", out_relevant.size(), out_relevant_long.size())
    #     print(target_tensor_chars_relevant_long.size(), out_relevant_long.size())
         inp_to_char_decoder = torch.cat([character_embeddings(target_tensor_chars_relevant_long[:-1]), out_relevant_long.unsqueeze(0).expand(CHAR_WORD_LENGTH-1, -1, -1)], dim=2)

   #      print(hidden_char_decoder[0].size())
#         quit()
         hidden_char_decoder = tuple((x[:, bpeLengthsFlattened > 1] for x in hidden_char_decoder))
         out_chars, hidden_char_decoder = char_decoder_rnn(inp_to_char_decoder, hidden_char_decoder)
  #       print(out_chars.size())
         out_chars = logsoftmax(char_decoder_output(out_chars))
 #        print(target_tensor_chars_relevant_long.size(), out_chars.size())
         loss_chars_SECOND = train_loss_chars(out_chars.view(-1, len(itos_chars_total)), target_tensor_chars_relevant_long[1:].contiguous().view(-1)).view(CHAR_WORD_LENGTH-1, -1)
#         print("loss_chars_SECOND",loss_chars_SECOND.size())
        

      # FIRST run on those 

      
         loss_chars = loss_chars_FIRST.sum() + loss_chars_SECOND.sum()
      else:
          loss_chars = loss_chars_FIRST.sum() 
      
      loss_chars_total = loss_chars.sum() / (args.sequence_length * args.batchSize)
     # quit()
#      print(loss_chars)
#      quit()

#      if train:
#          out = dropout(out)

#      logits = output(out) 
#      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      
      # train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)) + 
      loss = loss_chars_total

      if printHere:
#         oovs = (target_tensor.view(-1) == 2)
         full_loss_chars = torch.zeros(CHAR_WORD_LENGTH, args.sequence_length * args.batchSize).cuda()
 #        print(full_loss_chars.size(), loss_chars_FIRST.size(), loss_chars_SECOND.size())
         full_loss_chars[:1] = loss_chars_FIRST.view(1, args.sequence_length * args.batchSize)
         full_loss_chars[1:, bpeLengthsFlattened > 1] = loss_chars_SECOND
#         print(full_loss_chars.size())
         full_loss_chars_3D = full_loss_chars.view(CHAR_WORD_LENGTH, args.sequence_length, args.batchSize)

         lossTensor = full_loss_chars_3D.sum(0) #print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))
#         lossTensor.masked_scatter_(mask=oovs, source=loss_chars.sum(dim=0)[oovs])



         lossTensor = lossTensor.view(-1, args.batchSize)
         
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
#         boundaries_index = [0 for _ in numeric]
         print(("NONE", itos[numericCPU[0][0]-3]))
         for i in range((args.sequence_length)):
 #           if boundaries_index[0] < len(boundaries[0]) and i+1 == boundaries[0][boundaries_index[0]]:
  #             boundary = True
   #            boundaries_index[0] += 1
    #        else:
     #          boundary = False
            word =          itos[numericCPU[i+1][0]-3]
            print((losses[i][0], word, wordLengths[i][0], full_loss_chars_3D[:, i, 0].tolist()))
      return loss, target_tensor.view(-1).size()[0], wordLengths

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
      optim.step()


lossHasBeenBad = 0

import time

totalStartTime = time.time()


devLosses = []
for epoch in range(10000):
   print(epoch)
   training_data = corpusIteratorWikiWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



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
      loss, charCounts, wordLengths = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      if loss.data.cpu().numpy() > 40.0:
          lossHasBeenBad += 1
      else:
          lossHasBeenBad = 0
      if lossHasBeenBad > 100:
          print("Loss exploding, has been bad for a while")
          print(loss)
          quit()
      trainChars += charCounts 
      if printHere:
          print(("Loss here", loss, "BPC", float((loss * (args.sequence_length * args.batchSize))/wordLengths.sum())/0.6931472))
          print((epoch,counter))
          print("Dev losses")
          print(devLosses)
          print("Words per sec "+str(trainChars/(time.time()-startTime)))
          print(learning_rate)
          print(__file__)
          print(args)
      if counter % 20000 == 0: # and epoch == 0:
     #   if args.save_to is not None:
        state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
        torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")

      if (time.time() - totalStartTime)/60 > 4000:
          print("Breaking early to get some result within 72 hours")
          totalStartTime = time.time()
          break

 #     break
   rnn_drop.train(False)


   dev_data = corpusIteratorWikiWords.dev(args.language)
   print("Got data")
   dev_chars = prepareDatasetChunks(dev_data, train=False)


     
   dev_loss = 0
   dev_char_count = 0
   counter = 0
   hidden, beginning = None, None
   with torch.no_grad():
       while True:
           counter += 1
           try:
              numeric = next(dev_chars)
           except StopIteration:
              break
           printHere = (counter % 50 == 0)
           loss, numberOfCharacters, _ = forward(numeric, printHere=printHere, train=False)
           dev_loss += numberOfCharacters * loss.cpu().data.numpy()
           dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)
#   quit()
   #if args.save_to is not None:
 #     torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")

   with open("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+args.language+"_"+__file__+"_model_"+str(args.myID)+"_"+model+".txt", "w") as outFile:
       print(str(args), file=outFile)
       print(" ".join([str(x) for x in devLosses]), file=outFile)

   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
      break

   state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
   torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")






   learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
   optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9


