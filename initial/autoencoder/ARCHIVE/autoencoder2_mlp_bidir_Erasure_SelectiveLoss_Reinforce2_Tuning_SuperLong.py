print("Character aware!")


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German

import sys
import random
import torch
print(torch.__version__)

import math
from torch.autograd import Variable
import time
import corpusIteratorWikiWords
import sys
import time


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=878921872)


parser.add_argument("--batchSize", type=int, default=random.choice([128])) # , 128, 128, 256
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([512]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([0.01, 0.02, 0.05, 0.1])) #, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])) #, 2.4, 2.6, 2.8]))  # 0.1, 0.2, 0.4, 0.6, 
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([30]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
parser.add_argument("--char_dec_hidden_dim", type=int, default=128)


parser.add_argument("--deletion_rate", type=float, default=0.2)

parser.add_argument("--RATE_WEIGHT", type=float, default=random.choice([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]))
 # 1.5, 2.0, 2.5,  3.0, 3.5, 

#[1.25, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0])) # 0.5, 0.75, 1.0,  ==> this is essentially the point at which showing is better than guessing
parser.add_argument("--momentum", type=float, default=random.choice([0.0, 0.0, 0.0, 0.3, 0.5, 0.7, 0.9]))
parser.add_argument("--entropy_weight", type=float, default=0.0) #random.choice([0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.001])) # 0.0,  0.005, 0.01, 0.1, 0.4]))


model = "REAL_REAL"


args=parser.parse_args()




print(args.myID)
# Redirect output to a file (I did this for experiments via SLURM)
sys.stdout = open("/u/scr/mhahn/reinforce-logs/full-logs/"+__file__+"_"+str(args.myID), "w")

print(args)






def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


#############################################################
# Vocabulary

vocab_path = "vocabularies/"+args.language.lower()+"-wiki-word-vocab-50000.txt"

with open(vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


itos_total = ["<SOS>", "<EOS>", "OOV"] + itos
stoi_total = dict([(itos_total[i],i) for i in range(len(itos_total))])



############################################################
# Model

rnn_encoder = torch.nn.LSTM(2*args.word_embedding_size, int(args.hidden_dim/2.0), args.layer_num, bidirectional=True).cuda()
rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()
output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()
word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)
attention_softmax = torch.nn.Softmax(dim=1)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

attention_proj = torch.nn.Linear(args.hidden_dim, args.hidden_dim, bias=False).cuda()
attention_proj.weight.data.fill_(0)

output_mlp = torch.nn.Linear(2*args.hidden_dim, args.hidden_dim).cuda()

modules_autoencoder = [rnn_decoder, rnn_encoder, output, word_embeddings, attention_proj, output_mlp]

#################################################################
# Noise model
memory_mlp_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
memory_mlp_outer = torch.nn.Linear(500, 1).cuda()

sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()


modules_memory = [memory_mlp_inner, memory_mlp_outer]

###################################################################

def parameters_memory():
   for module in modules_memory:
       for param in module.parameters():
            yield param


def parameters_autoencoder():
   for module in modules_autoencoder:
       for param in module.parameters():
            yield param

parameters_memory_cached = [x for x in parameters_memory()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters_memory(), lr=learning_rate, momentum=args.momentum) # 0.02, 0.9


if args.load_from_autoencoder is not None:
  try:
     checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  except FileNotFoundError:
     checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      modules_autoencoder[i].load_state_dict(checkpoint["components"][i])


##########################################################################
# Encode dataset chunks into tensors

def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      for chunk in data:
       for char in chunk:
         count += 1
         numerified.append((stoi[char]+3 if char in stoi else 2))
       if len(numerified) > (args.batchSize*args.sequence_length):
         sequenceLengthHere = args.sequence_length
         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerified = numerified[cutoff:]
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], None
         hidden = None
       else:
         print("Skipping")





hidden = None
zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None
zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

# Dropout masks
bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())
bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * 2 * args.hidden_dim)]).cuda())


runningAverageReward = 1.0
expectedRetentionRate = 0.5

def forward(numeric, train=True, printHere=False):
      global beginning
      if True:
          beginning = zeroBeginning

      numeric, _ = numeric


      # the true input
      numeric = torch.cat([beginning, numeric], dim=0)
      # word embeddings for the true input (to be fed into noise model)
      embedded_everything = word_embeddings(numeric)

      # computing the noise model
      memory_hidden = sigmoid(memory_mlp_outer(relu(memory_mlp_inner(embedded_everything))))
      # Sampling a mask from the noise model
      memory_filter = torch.bernoulli(input=memory_hidden)
      # logprobabilities of the actions of the noise model
      bernoulli_logprob = torch.where(memory_filter == 1, torch.log(memory_hidden+1e-10), torch.log(1-memory_hidden+1e-10))
      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)

      # If necessary, calculate entropy of decisions (for regularization purposes)
      if args.entropy_weight > 0:
         entropy = -(memory_hidden * torch.log(memory_hidden+1e-10) + (1-memory_hidden) * torch.log(1-memory_hidden+1e-10)).mean()
      else:
         entropy=-1.0


      memory_filter = memory_filter.squeeze(2)
      # Run the noise model
      numeric_noised = torch.where(memory_filter==1, numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]
      # true input in those places where it noised, and 0 elsewhere
      numeric_onlyNoisedOnes = torch.where(memory_filter == 0, numeric, 0*numeric) # target is 0 in those places where no noise has happened

      # Input to the decoding RNN
      input_tensor = Variable(numeric[:-1], requires_grad=False)
      # Input for the encoding RNN
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      # Target for the decoding RNN

      target_tensor = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)

      # Encode input for the decoding RNN
      embedded = word_embeddings(input_tensor)
#      if False and train:
#         embedded = char_dropout(embedded)
#         mask = bernoulli_input.sample()
#         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
#         embedded = embedded * mask
      # Encode input for the encoding RNN
      embedded_noised = word_embeddings(input_tensor_noised)
#      if False and train:
#         embedded_noised = char_dropout(embedded_noised)
#         mask = bernoulli_input.sample()
#         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
#         embedded_noised = embedded_noised * mask

      # Run both encoder and decoder
      out_encoder, _ = rnn_encoder(embedded_noised, None)
      out_decoder, _ = rnn_decoder(embedded, None)

      # Have the decoder attend to the encoder
      attention = torch.bmm(attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
      attention = attention_softmax(attention).transpose(0,1)
      from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      out_full = torch.cat([out_decoder, from_encoder], dim=2)

#      if False and train:
#        mask = bernoulli_output.sample()
#        mask = mask.view(1, args.batchSize, 2*args.hidden_dim)
#        out_full = out_full * mask



      # Obtain logits for reconstruction
      logits = output(relu(output_mlp(out_full) ))
      # Obtain log-probabilities
      log_probs = logsoftmax(logits)

      # Calculate loss. Here, due to the ignore_index=0 argument in the definition of train_loss, loss is incurred only for recovering those words that had been noised.
      lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
      negativeRewardsTerm1 = lossTensor.mean(dim=0)

      # Regularization towards lower retention rates
      negativeRewardsTerm2 = memory_filter.mean(dim=0)

      # Overall Reward
      negativeRewardsTerm = negativeRewardsTerm1 + args.RATE_WEIGHT * negativeRewardsTerm2

      global runningAverageReward
      global expectedRetentionRate

      loss = ((negativeRewardsTerm.detach()-runningAverageReward) * bernoulli_logprob_perBatch).mean()
      if args.entropy_weight > 0:
         loss -= args.entropy_weight  * entropy
      expectedRetentionRate = memory_hidden.mean()

      # Occasionally print
      if printHere:
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy()
         memory_hidden_CPU = memory_hidden[:,0,0].cpu().data.numpy()
         print(("NONE", itos_total[numericCPU[0][0]]))
         for i in range((args.sequence_length)):
            print((losses[i][0], itos_total[numericCPU[i+1][0]], itos_total[numeric_noisedCPU[i+1][0]], memory_hidden_CPU[i+1]))

         print("PREDICTION_LOSS", round(float(negativeRewardsTerm1.mean()),3), "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", float(expectedRetentionRate), "\tDEVIATION FROM BASELINE", float((negativeRewardsTerm.detach()-runningAverageReward).abs().mean()), "\tREWARD", runningAverageReward, "\tENTROPY", float(entropy))
      #runningAveragePredictionLoss = 0.95 * runningAveragePredictionLoss + (1-0.95) * float(negativeRewardsTerm1.mean())
      runningAverageReward = 0.95 * runningAverageReward + (1-0.95) * float(negativeRewardsTerm.mean())

      return loss, target_tensor.view(-1).size()[0]

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_memory_cached, 5.0) #, norm_type="inf")
      optim.step()


lossHasBeenBad = 0


totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
updatesCount = 0
for epoch in range(10000):
   print(epoch)
   training_data = corpusIteratorWikiWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   rnn_encoder.train(True)
   rnn_decoder.train(True)

   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   if updatesCount >= 100000:
     break
   while updatesCount <= 100000:
      counter += 1
      updatesCount += 1
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
          assert False
      trainChars += charCounts 
      if printHere:
          print(("Loss here", loss))
          print((epoch,counter))
          print("Dev losses")
          print(devLosses)
          print("Words per sec "+str(trainChars/(time.time()-startTime)))
          print(learning_rate)
          print(lastSaved)
          print(__file__)
          print(args)
#      if counter % 2000 == 0: # and epoch == 0:
#        state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules_memory]}
#        torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")
#        lastSaved = (epoch, counter)
      if (time.time() - totalStartTime)/60 > 4000:
          print("Breaking early to get some result within 72 hours")
          totalStartTime = time.time()
          break

# #     break
#   rnn_encoder.train(False)
#   rnn_decoder.train(False)
#
#
#   dev_data = corpusIteratorWikiWords.dev(args.language)
#   print("Got data")
#   dev_chars = prepareDatasetChunks(dev_data, train=False)
#
#
#     
#   dev_loss = 0
#   dev_char_count = 0
#   counter = 0
#   hidden, beginning = None, None
#   while True:
#       counter += 1
#       try:
#          numeric = next(dev_chars)
#       except StopIteration:
#          break
#       printHere = (counter % 50 == 0)
#       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
#       dev_loss += numberOfCharacters * loss.cpu().data.numpy()
#       dev_char_count += numberOfCharacters
#   devLosses.append(dev_loss/dev_char_count)
#   print(devLosses)
##   quit()
#
##   with open("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+args.language+"_"+__file__+"_model_"+str(args.myID)+"_"+model+".txt", "w") as outFile:
##       print(str(args), file=outFile)
##       print(" ".join([str(x) for x in devLosses]), file=outFile)
#
#   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
#      break
#
##   state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules_memory]}
##   torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")
##   lastSaved = (epoch, counter)
#
#
#
#
#
#
#   learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
#   optim = torch.optim.SGD(parameters_memory(), lr=learning_rate, momentum=args.momentum) # 0.02, 0.9

with open("/u/scr/mhahn/reinforce-logs/results/"+__file__+"_"+str(args.myID), "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward, file=outFile)
   print(expectedRetentionRate, file=outFile)
