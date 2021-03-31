print("Character aware!")
import random


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=random.choice([673269957,38679926]))
# 777726352, doesn't have the right parameter matrix sizes

#(1.020192962941362, 1, False, '595155021', 'None')
#(1.0125508174521998, 1, False, '346951340', "'860052739'")
#(0.9656905007620464, 1, False, '230092254', "'595155021'")
#(0.9656414629158051, 1, False, '984542859', "'595155021'")
#(0.9622109624303887, 2, False, '264073608', "'595155021'")
#(0.9400924757949716, 2, False, '777726352', "'456889167'")
#(0.9336853147520909, 4, False, '449431785', "'984542859'")




import random

parser.add_argument("--batchSize", type=int, default=random.choice([128]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([512]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.1])) # used to be just 0.1, but I want to explore other options
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.2, 0.4, 0.6, 1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([30]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
parser.add_argument("--char_dec_hidden_dim", type=int, default=128)


parser.add_argument("--deletion_rate", type=float, default=0.2)

parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)

parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.3]))


parser.add_argument("--RATE_WEIGHT", type=float, default=random.choice([5.5, 5.75, 6.0, 6.25, 6.5]))
# 4.75, 5.0, 5.25, 

#6.0, 6.5])) #, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]))
 # 1.5, 2.0, 2.5,  3.0, 3.5, 

#[1.25, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0])) # 0.5, 0.75, 1.0,  ==> this is essentially the point at which showing is better than guessing
parser.add_argument("--momentum", type=float, default=random.choice([0.0, 0.0, 0.0, 0.5]))
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.00002, 0.00005, 


model = "REAL_REAL"

import math

args=parser.parse_args()

print(args.myID)
import sys
print(args, file=sys.stderr)


sys.stdout = open("/u/scr/mhahn/reinforce-logs/full-logs/"+__file__+"_"+str(args.myID), "w")

print(args)



import corpusIteratorWikiWords_BoundarySymbols as corpusIteratorWikiWords



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


#with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
#     itos_chars = [x for x in inFile.read().strip().split("\n")]
#stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])
#
#
#itos_chars_total = ["<SOS>", "<EOS>", "OOV"] + itos_chars


import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop

class Autoencoder:
  def __init__(self):
    self.rnn_encoder = torch.nn.LSTM(2*args.word_embedding_size, int(args.hidden_dim/2.0), args.layer_num, bidirectional=True).cuda()
    self.rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()
    self.output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()
    self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()
    self.logsoftmax = torch.nn.LogSoftmax(dim=2)
    self.attention_softmax = torch.nn.Softmax(dim=1)
    self.train_loss = torch.nn.NLLLoss(ignore_index=0)
    self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
    self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
    self.attention_proj = torch.nn.Linear(args.hidden_dim, args.hidden_dim, bias=False).cuda()
    self.attention_layer = torch.nn.Bilinear(args.hidden_dim, args.hidden_dim, 1, bias=False).cuda()
    self.attention_proj.weight.data.fill_(0)
    self.output_mlp = torch.nn.Linear(2*args.hidden_dim, args.hidden_dim).cuda()
    self.modules_autoencoder = [self.rnn_decoder, self.rnn_encoder, self.output, self.word_embeddings, self.attention_proj, self.output_mlp]
autoencoder = Autoencoder()

class MemoryModel:
  def __init__(self):
     self.memory_mlp_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.memory_mlp_outer = torch.nn.Linear(500, 1).cuda()
     self.sigmoid = torch.nn.Sigmoid()
     self.relu = torch.nn.ReLU()
     self.perword_baseline_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.perword_baseline_outer = torch.nn.Linear(500, 1).cuda()
     # Collect the modules
     self.modules_memory = [self.memory_mlp_inner, self.memory_mlp_outer, self.perword_baseline_inner, self.perword_baseline_outer]
memory = MemoryModel()

def parameters_memory():
   for module in memory.modules_memory:
       for param in module.parameters():
            yield param

dual_weight = torch.cuda.FloatTensor([3.0])
dual_weight.requires_grad=True


parameters_memory_cached = [x for x in parameters_memory()]




def parameters_autoencoder():
   for module in autoencoder.modules_autoencoder:
       for param in module.parameters():
            yield param

parameters_memory_cached = [x for x in parameters_memory()]


#learning_rate = args.learning_rate

optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9

#named_modules_autoencoder = {"rnn" : rnn, "output" : output, "word_embeddings" : word_embeddings, "optim" : optim}

if args.load_from_autoencoder is not None:
 # try:
  print(args.load_from_autoencoder)
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_BoundarySymbol_NoComma.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
#  except FileNotFoundError:
 #    checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidirISTHEREANOTHEROPTION?.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["components"][i])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

relu = torch.nn.ReLU()

def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      numerified_chars = []
      for chunk in data:
       #print(len(chunk))
       for char in chunk:
#         if char == " ":
 #          continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         if char == ",": # Skip commas
           continue
         numerified.append((stoi_total[char] if char in stoi_total else 2))
#         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])

       if len(numerified) > (args.batchSize*args.sequence_length):
         sequenceLengthHere = args.sequence_length

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
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * 2 * args.hidden_dim)]).cuda())


#runningAveragePredictionLoss = 1.0
runningAverageReward = 1.0
runningAverageBaselineDeviation = 2.0
runningAveragePredictionLoss = 5.0
runningAverageReconstructionLoss = 5.0
expectedRetentionRate = 0.5

def product(x):
   r = 1
   for i in x:
     r *= i
   return r

def forward(numeric, train=True, printHere=False, onlyProvideMemoryResult=False):
      global beginning
      global beginning_chars
      if True:
          beginning = zeroBeginning

      numeric, numeric_chars = numeric

      ######################################################
      ######################################################
      # Run Loss Model

      numeric = torch.cat([beginning, numeric], dim=0)

      embedded_everything = autoencoder.word_embeddings(numeric)


      memory_hidden = memory.sigmoid(memory.memory_mlp_outer(relu(memory.memory_mlp_inner(embedded_everything))))

      # Baseline predictions for prediction loss
      baselineValues = 10*memory.sigmoid(memory.perword_baseline_outer(memory.relu(memory.perword_baseline_inner(embedded_everything[-1].detach())))).squeeze(1)
#      assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)



      memory_filter = torch.bernoulli(input=memory_hidden)

      bernoulli_logprob = torch.where(memory_filter == 1, torch.log(memory_hidden+1e-10), torch.log(1-memory_hidden+1e-10))

      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)
      if args.entropy_weight > 0:
         entropy = -(memory_hidden * torch.log(memory_hidden+1e-10) + (1-memory_hidden) * torch.log(1-memory_hidden+1e-10)).mean()
      else:
         entropy=-1.0


      memory_filter = memory_filter.squeeze(2)

      ####################################################################################
      numeric_noised = torch.where(torch.logical_or(memory_filter==1 , numeric == 1), numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]

      
      numeric_noised = [[x for x in y if int(x) != 0] for y in numeric_noised.cpu().t()]
      numeric_noised = torch.LongTensor([[0 for _ in range(args.sequence_length+1-len(y))] + y for y in numeric_noised]).cuda().t()
 #     print(numeric.size(), numeric_noised.size())
#      quit()

      if onlyProvideMemoryResult:
        return numeric, numeric_noised




      input_tensor = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)

      target_tensor = Variable(numeric[1:], requires_grad=False)


      embedded = autoencoder.word_embeddings(input_tensor)
      if False and train:
         embedded = char_dropout(embedded)
         mask = bernoulli_input.sample()
         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
         embedded = embedded * mask

      embedded_noised = autoencoder.word_embeddings(input_tensor_noised)
      if False and train:
         embedded_noised = char_dropout(embedded_noised)
         mask = bernoulli_input.sample()
         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
         embedded_noised = embedded_noised * mask
#      print(input_tensor_noised)
 #     print("NOISED", embedded_noised[1,1,])
  #    print("NOISED", embedded_noised[1,2,])
   #   print("NOISED", embedded_noised[2,2,])

    #  print(embedded_noised.size())


      ##########################################
      ##########################################
      # RUN AUTOENCODER (approximately inverting loss model)

      
      out_encoder, _ = autoencoder.rnn_encoder(embedded_noised, None)
      out_decoder, _ = autoencoder.rnn_decoder(embedded, None)

      attention = torch.bmm(autoencoder.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
      attention = autoencoder.attention_softmax(attention).transpose(0,1)
      from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      out_full = torch.cat([out_decoder, from_encoder], dim=2)


      if False and train:
        mask = bernoulli_output.sample()
        mask = mask.view(1, args.batchSize, 2*args.hidden_dim)
        out_full = out_full * mask



      logits = autoencoder.output(relu(autoencoder.output_mlp(out_full) ))
      log_probs = autoencoder.logsoftmax(logits)

      # Prediction Loss 
      autoencoder_lossTensor = autoencoder.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)

      ##########################################
      ##########################################

      # Reward, term 1
      if True:
        negativeRewardsTerm1 = autoencoder_lossTensor.mean(dim=0)

      # Reward, term 2
      # Regularization towards lower retention rates
      negativeRewardsTerm2 = memory_filter.mean(dim=0)
      retentionTarget = 1-args.deletion_rate
      loss = 0

      # Autoencoder Loss
      loss += autoencoder_lossTensor.mean()

      # Overall Reward
      negativeRewardsTerm = negativeRewardsTerm1 + dual_weight * (negativeRewardsTerm2-retentionTarget)
      if printHere:
         print([float(x) for x in [negativeRewardsTerm.mean(), negativeRewardsTerm1.mean(), dual_weight , negativeRewardsTerm2.mean(), (negativeRewardsTerm2-retentionTarget).mean()]])
      # for the dual weight
      loss += (dual_weight * (negativeRewardsTerm2-retentionTarget).detach()).mean()
      if printHere:
          print(negativeRewardsTerm1.mean(), dual_weight, negativeRewardsTerm2.mean(), retentionTarget)
      #print(loss)

#      global runningAveragePredictionLoss
      global runningAverageReward

      # Reward Minus Baseline
      # Detached surprisal and mean retention
      rewardMinusBaseline = (negativeRewardsTerm.detach() - baselineValues - (dual_weight * (memory_hidden.mean(dim=0).squeeze(dim=1) - retentionTarget)).detach())

      # Important to detach from the baseline!!! 
      loss += (rewardMinusBaseline.detach() * bernoulli_logprob_perBatch.squeeze(1)).mean()
      if args.entropy_weight > 0:
         loss -= args.entropy_weight  * entropy

      # Loss for trained baseline
      loss += args.reward_multiplier_baseline * rewardMinusBaseline.pow(2).mean()

      ############################
      # Construct running averages
      factor = 0.9996 ** args.batchSize

      # Update running averages
      global runningAverageBaselineDeviation
      global runningAveragePredictionLoss
      global runningAverageReconstructionLoss
      global runningAverageReward
      global expectedRetentionRate

      expectedRetentionRate = factor * expectedRetentionRate + (1-factor) * float(memory_hidden.mean())
      runningAverageBaselineDeviation = factor * runningAverageBaselineDeviation + (1-factor) * float((rewardMinusBaseline).abs().mean())

      ############################

      if printHere:
         losses = autoencoder_lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy()
         memory_hidden_CPU = memory_hidden[:,0,0].cpu().data.numpy()
         print(("NONE", itos_total[numericCPU[0][0]]))
         for i in range((args.sequence_length)):
            print((losses[i][0], itos_total[numericCPU[i+1][0]], memory_hidden_CPU[i+1], itos_total[numeric_noisedCPU[i+1][0]]))

         print("PREDICTION_LOSS", round(float(negativeRewardsTerm1.mean()),3), "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", float(expectedRetentionRate), "\tDEVIATION FROM BASELINE", float((negativeRewardsTerm.detach()-runningAverageReward).abs().mean()), "\tREWARD", runningAverageReward, "\tDUAL", float(dual_weight))
         sys.stderr.write(" ".join([str(x) for x in ["\r", "PREDICTION_LOSS", round(float(negativeRewardsTerm1.mean()),3), "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", float(expectedRetentionRate), "\tDEVIATION FROM BASELINE", float((negativeRewardsTerm.detach()-runningAverageReward).abs().mean()), "\tREWARD", runningAverageReward, "\tDUAL", float(dual_weight), counter]]))
         if counter % 5000 == 0:
            print("", file=sys.stderr)
         sys.stderr.flush()

      #runningAveragePredictionLoss = 0.95 * runningAveragePredictionLoss + (1-0.95) * float(negativeRewardsTerm1.mean())
      runningAverageReward = 0.95 * runningAverageReward + (1-0.95) * float(negativeRewardsTerm.mean())

      return loss, target_tensor.view(-1).size()[0]

def backward(loss, printHere):
      optim_autoencoder.zero_grad()
      optim_memory.zero_grad()

      if dual_weight.grad is not None:
         dual_weight.grad.data.fill_(0.0)
      if printHere:
         print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_memory_cached, 5.0) #, norm_type="inf")
      optim_autoencoder.step()
      optim_memory.step()

#      print(dual_weight.grad)
      dual_weight.data.add_(args.dual_learning_rate*dual_weight.grad.data)
 #     print("W", dual_weight)
      dual_weight.data.clamp_(min=0)
  #    print("W", dual_weight)

lossHasBeenBad = 0

import time

totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
updatesCount = 0
epoch = 0
while updatesCount <= 50000:
   epoch += 1
   print(epoch)
   training_data = corpusIteratorWikiWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   autoencoder.rnn_encoder.train(True)
   autoencoder.rnn_decoder.train(True)

   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   if updatesCount >= 50000:
     break
#   if expectedRetentionRate < 0.15:
 #     print(("retention rate has fallen", expectedRetentionRate), file=sys.stderr)
  #    break
   while updatesCount <= 50000:
#      if expectedRetentionRate < 0.15:
 #        print(("retention rate has fallen", expectedRetentionRate), file=sys.stderr)
  #       break
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
          print(args.learning_rate_memory, args.learning_rate_autoencoder)
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

if True:
  modules_memory_and_autoencoder = memory.modules_memory + autoencoder.modules_autoencoder
  state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules_memory_and_autoencoder]}
  torch.save(state, "/u/scr/mhahn/CODEBOOKS_memoryPolicy_both/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")
  lastSaved = (epoch, counter)


with open("/u/scr/mhahn/reinforce-logs/results/"+__file__+"_"+str(args.myID), "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward, file=outFile)
   print(expectedRetentionRate, file=outFile)

