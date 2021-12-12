# ~/python-py37-mhahn errorIdentification_Erasure.py --load_from_joint=42600474
# ~/python-py37-mhahn errorIdentification_Erasure3.py --load_from_joint=631984614  | grep EIS
# ~/python-py37-mhahn errorIdentification_Erasure3.py --load_from_joint=832508929  | grep EIS
# Derived from char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_Lo.py



#(base) mhahn@jagupard12:/juice/scr/mhahn/CODE/forgetting-model/initial/lm/trainAttention$ ~/python-py37-mhahn errorIdentification_Erasure3.py --load_from_joint=245600212 | grep EIS                        
#(base) mhahn@jagupard12:/juice/scr/mhahn/CODE/forgetting-model/initial/lm/trainAttention$ ~/python-py37-mhahn errorIdentification_Erasure3.py --load_from_joint=688174451 | grep EIS
#



#(base) mhahn@jagupard12:/juice/scr/mhahn/CODE/forgetting-model/initial/lm/trainAttention$ ~/python-py37-mhahn errorIdentification_Erasure3_NoSanity.py --load_from_joint=688174451 | grep EIS               
#(base) mhahn@jagupard12:/juice/scr/mhahn/CODE/forgetting-model/initial/lm/trainAttention$ ~/python-py37-mhahn errorIdentification_Erasure3_NoSanity.py --load_from_joint=245600212 | grep EIS               Namespace(NUMBER_OF_REPLICATES=12, batchSize=1, char_dropout_prob=0.01, criticalRegions=None, deletion_rate=0.5, dual_learning_rate=0.3, entropy_weight=0.0, hidden_dim_autoencoder=512, hidden_dim_lm=1024, language='english', layer_num=2, learning_rate_autoencoder=0.1, learning_rate_memory=5e-05, load_from_joint='245600212', lr_decay=1.0, momentum=0.7, myID=712105040, predictability_weight=0.25, reward_multiplier_baseline=0.1, sequence_length=20, stimulus_file=None, tuning=1, verbose=False, weight_dropout_in=0.05, weight_dropout_out=0.05, word_embedding_size=512)

# Based on:
#  char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos12_Long.py (loss model & code for language model)
# And autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Both_Saving.py (autoencoder)
# And (for the plain LM): ../autoencoder/autoencoder2_mlp_bidir_AND_languagemodel_sample.py
print("Character aware!")
import os
# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import glob
import sys
import random
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-joint", dest="load_from_joint", type=str, default=random.choice([119493105, 126895959, 300898142, 514107740]))
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

## Regularization
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.1]))
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

#[1.25, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0])) # 0.5, 0.75, 1.0,  ==> this is essentially the point at which showing is better than guessing
parser.add_argument("--momentum", type=float, default=random.choice([0.0, 0.0, 0.0, 0.5]))
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.00002, 0.00005, 


parser.add_argument("--stimulus_file", type=str)
parser.add_argument("--criticalRegions", type=str)


TRAIN_LM = False
assert not TRAIN_LM



model = "REAL_REAL"

import math

args=parser.parse_args()

print(args.myID)
import sys
print(args, file=sys.stderr)


#sys.stdout = open("/u/scr/mhahn/reinforce-logs/full-logs/"+__file__+"_"+str(args.myID), "w")

print(args)



import corpusIteratorWikiWords_BoundarySymbols as corpusIteratorWikiWords



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


# Load Vocabulary
char_vocab_path = "vocabularies/"+args.language.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


itos_total = ["<SOS>", "<EOS>", "OOV"] + itos
stoi_total = dict([(itos_total[i],i) for i in range(len(itos_total))])


import random
import torch

print(torch.__version__)




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
    self.softmax = torch.nn.Softmax(dim=2)
    self.attention_softmax = torch.nn.Softmax(dim=1)
    self.train_loss = torch.nn.NLLLoss(ignore_index=0)
    self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
    self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
    self.attention_proj = torch.nn.Linear(args.hidden_dim, args.hidden_dim, bias=False).cuda()
    self.attention_layer = torch.nn.Bilinear(args.hidden_dim, args.hidden_dim, 1, bias=False).cuda()
    self.attention_proj.weight.data.fill_(0)
    self.output_mlp = torch.nn.Linear(2*args.hidden_dim, args.hidden_dim).cuda()
    self.modules_autoencoder = [self.rnn_decoder, self.rnn_encoder, self.output, self.word_embeddings, self.attention_proj, self.output_mlp]

 
  def sampleReconstructions(self, numeric, numeric_noised, NOUN, offset, numberOfBatches):
      """ Draws samples from the amortized reconstruction posterior """
      if True:
          beginning = zeroBeginning
      numeric = numeric[:3]
      input_tensor = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      print("Noised Input", " ".join([itos_total[numeric_noised[i,0]] for i in range(numeric_noised.size()[0])]))

      embedded = self.word_embeddings(input_tensor)
      embedded_noised = self.word_embeddings(input_tensor_noised)
      out_encoder, _ = self.rnn_encoder(embedded_noised, None)



      hidden = None
      result  = ["" for _ in range(numberOfBatches)]
      result_numeric = [[] for _ in range(numberOfBatches)]
      embeddedLast = embedded[0].unsqueeze(0)
      amortizedPosterior = torch.zeros(numberOfBatches, device='cuda')
      zeroLogProb = torch.zeros(numberOfBatches, device='cuda')
      hasSampledSOSOnce = torch.zeros(numberOfBatches, device='cuda').bool()
      print(embedded.size(), numberOfBatches)
#      hasSampledSOSTwice = torch.zeros(numberOfBatches, device='cuda').bool()
      for i in range(args.sequence_length+1):
          out_decoder, hidden = self.rnn_decoder(embeddedLast, hidden)
    
          attention = torch.bmm(self.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = self.attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)

 #         print(input_tensor.size())


          logits = self.output(relu(self.output_mlp(out_full) )) 
          probs = self.softmax(logits)

          dist = torch.distributions.Categorical(probs=probs)
       
#          nextWord = (dist.sample())
          nextWord = dist.sample()
          logProbForSampledFromDist = dist.log_prob(nextWord).squeeze(0)

          print(hasSampledSOSOnce.size(), hasSampledSOSOnce.size(), logProbForSampledFromDist.size(), zeroLogProb.size())
          amortizedPosterior += torch.where(hasSampledSOSOnce, logProbForSampledFromDist, zeroLogProb)
#          print(i, itos_total[int(nextWord[0,0])], hasSampledSOSOnce[0], float(amortizedPosterior[0]))
#          hasSampledSOSTwice = torch.logical_and(hasSampledSOSOnce, nextWord.squeeze(0) == stoi_total["<EOS>"])
          hasSampledSOSOnce = torch.logical_xor(hasSampledSOSOnce, nextWord.squeeze(0) == stoi_total["<EOS>"])
  #        print(nextWord.size())
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for j in range(numberOfBatches):
            if hasSampledSOSOnce[j]:
             result[j] += " "+nextWordStrings[j]
            result_numeric[j].append( nextWordDistCPU[j] )
          print("Sampling", i, result[0])
          embeddedLast = self.word_embeddings(nextWord)
#          print(embeddedLast.size())
      for r in result[:2]:
         print("RECONSTRUCTION", r)
# "@@@", " ".join([itos_total[result_numeric[r][i]] for i in range(args.sequence_length+1)]))
      result_numeric = torch.LongTensor(result_numeric).cuda()
      assert result_numeric.size()[0] == numberOfBatches
      return result, result_numeric, None, None, amortizedPosterior

 
autoencoder = Autoencoder()

class MemoryModel:
  """ Noise Model """
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

if args.load_from_joint is not None:
 # try:
  print(args.load_from_joint)
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS_memoryPolicy_both/"+args.language+"_"+"autoencoder2_mlp_bidir_Deletion_Reinforce2_Tuning_Long_Both_Saving_Lagrange_BoundarySymbol_NoPunct.py"+"_code_"+str(args.load_from_joint)+".txt")

#  modules_memory_and_autoencoder = memory.modules_memory + autoencoder.modules_autoencoder
 # state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules_memory_and_autoencoder]}
  #torch.save(state, "/u/scr/mhahn/CODEBOOKS_memoryPolicy_both/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")

  assert itos == checkpoint["words"]
  print("ARGUMENTS FROM TRAINING", checkpoint["arguments"]) 
  modules_memory_and_autoencoder = memory.modules_memory + autoencoder.modules_autoencoder
  for x, y in zip(modules_memory_and_autoencoder, checkpoint["components"]):
     x.load_state_dict(y)

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

PUNCTUATION = torch.LongTensor([stoi_total[x] for x in [".", "OOV", '"', "(", ")", "'", '"', ":", ",", "'s", "[", "]", "<EOS>"]]).cuda()

def forward(numeric, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, onlyProvideKeepProbabilities=False):
      global beginning
      global beginning_chars
      if True:
          beginning = zeroBeginning

      numeric, numeric_chars = numeric

      ######################################################
      ######################################################
      # Run Loss Model
      beginning = torch.LongTensor([0 for _ in range(numeric.size()[1])]).cuda().view(1,numeric.size()[1])

      numeric = torch.cat([beginning, numeric], dim=0)

      embedded_everything = autoencoder.word_embeddings(numeric)


      memory_hidden = memory.sigmoid(memory.memory_mlp_outer(relu(memory.memory_mlp_inner(embedded_everything))))

      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(13, 1, 1)).long().sum(dim=0)).bool())


      if onlyProvideKeepProbabilities:
#         print(punctuation.size(), memory_hidden.size())
         return torch.where(punctuation.unsqueeze(2), 0*memory_hidden+1, memory_hidden)



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
      numeric_noised = torch.where(torch.logical_or(punctuation, memory_filter==1), numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]

      
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

lossHasBeenBad = 0

import time

totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
updatesCount = 0


def encodeContextCrop(inp, context, replicates):
     sentence = context.strip() + " " + inp.strip()
     print("ENCODING", sentence)
     numerified = [stoi_total[char] if char in stoi_total else 2 for char in sentence.split(" ")]
     print(len(numerified))
     numerified = numerified[-args.sequence_length-1:]
     numerified = torch.LongTensor([numerified for _ in range(replicates)]).t().cuda()
     return numerified



                                                                                                                                                                          

def normalize_dict(d):
   t = sum(y for _, y in d.items())
   return {x : y/t for x, y in d.items()}



def getLikelihoodDP(resultNumeric_gpu, numeric_noised, resultNumeric, IMPORTANCE_SAMPLING_K, numberOfSamples):
      print(resultNumeric.size(), numeric_noised.size())
      keep_probabilities = forward((resultNumeric_gpu, None), train=False, printHere=False, provideAttention=False, onlyProvideKeepProbabilities=True)
      keep_probabilities = keep_probabilities.squeeze(2).detach().cpu()
      sampled_results = []
      likelihood = torch.zeros(numberOfSamples*IMPORTANCE_SAMPLING_K)
      for i in range((numberOfSamples*IMPORTANCE_SAMPLING_K)):
         noised = numeric_noised[:,i].cpu().numpy().tolist()
         sample = resultNumeric[:,i].cpu().numpy().tolist()
         try:
           offset = sample.index(stoi_total["<EOS>"])
           noised = noised[noised.index(stoi_total["<EOS>"])+1:]
           sample = sample[sample.index(stoi_total["<EOS>"])+1:]
           noised = noised[:noised.index(stoi_total["<EOS>"])]
           sample = sample[:sample.index(stoi_total["<EOS>"])]
         except ValueError:
           likelihood[i] = 1e-50
           sampled_results.append([])
           continue
         sampled_results.append(sample)
#         print(noised)
 #        print(sample)
         p_table = torch.zeros(len(sample)+1, len(noised)+1)
         p_table[0,0] = 1
  #       print(result[i])
   #      print(keep_probabilities[:,i])
         #quit()
         for r in range(1, len(sample)+1):
            for s in range(min(r+1, len(noised)+1)):
               if s > 0 and sample[r-1] == noised[s-1]:
                 p_table[r,s] += p_table[r-1, s-1] * float(keep_probabilities[offset + r + 1,i]) 
    #           print(itos_total[sample[r]], float(keep_probabilities[offset + r + 2,i]))
               p_table[r,s] += p_table[r-1, s] * (1-float(keep_probabilities[offset + r + 1,i]))
   #            print(r, s, [itos_total[x] for x in sample[:r]], [itos_total[x] for x in noised[:s]], p_table[r,s], itos_total[sample[r-1]], float(keep_probabilities[offset + r + 1,i]), keep_probabilities.size(), offset, r)
     #    print(p_table)
         likelihood[i] = float(p_table[-1, -1])
  #       print(keep_probabilities[offset+2:,i])
 #        print(likelihood[i], " ".join([itos_total[x] for x in sample]), "@@@@", " ".join([itos_total[x] for x in noised]))
      return likelihood
#      quit()


def getSurprisalsStimuli(SANITY="Sanity"):
   # with open(f"/u/scr/mhahn/STIMULI/{args.stimulus_file}.tsv", "r") as inFile:
    #   data = [x.split("\t") for x in inFile.read().strip().split("\n")]
    #   header = data[0]
    #   assert header == ["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]
    #   header = dict(list(zip(header, range(len(header)))))
    #   data = data[1:]
#       from collections import defaultdict
 #      sentences = defaultdict(list)
  #     for line in data:
  #         sentences[line[header["Sentence"]]].append(line)
  #     sentences = sorted(sentences.items(), key=lambda x:x[0])
  #  assert SANITY in ["ModelTmp", "Model", "Sanity", "ZeroLoss"]
    numberOfSamples = 6
    IMPORTANCE_SAMPLING_K = 12
    import scoreWithGPT2Medium as scoreWithGPT2
    sentences = [(1, [{"Item" : 1, "Condition" : 1, "Region" : i, "Word" : x} for i, x in  enumerate(["the", "coach", "looked", "at", "the", "tall", "player", "tossed", "the", "ball"])])]
    sentences += [(2, [{"Item" : 1, "Condition" : 1, "Region" : i, "Word" : x} for i, x in  enumerate(["the", "coach", "looked", "at", "the", "tall", "player", "thrown", "the", "ball"])])]
    with torch.no_grad():
#      outFile = sys.stdout
     with open("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv-EIS/"+__file__+"_"+args.stimulus_file.replace("/", "-")+"_"+str(args.load_from_joint if SANITY != "ZeroLoss" else "ZERO")+"_"+SANITY, "w") as outFile:
      print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "EISReweighted", "SurprisalReweighted", "Repetition"]), file=outFile)
      TRIALS_COUNT = 0
      for sentenceID, sentence in sentences:
          print(sentenceID)
          ITEM = sentence[0]["Item"]
          CONDITION = sentence[0]["Condition"]
          regions = [x["Region"] for x in sentence]
          sentence = [x["Word"].lower() for x in sentence]
          context = sentence[0]

          remainingInput = sentence[1:]
          regions = regions[1:]
          print("INPUT", context, remainingInput)
          assert len(remainingInput) > 0
          for i in range(len(remainingInput)):
            if regions[i] < 2: 
#            if not (args.criticalRegions is None) and regions[i] not in args.criticalRegions:
              continue
            for repetition in range(2):
              sentence_proc = " ".join(remainingInput[:i+1])
              context = "later the nurse suggested they treat the patient with an antibiotic but in the end this did not happen after this something else happened instead and she went away but nobody noticed anything about it <EOS> "+sentence_proc #+ " <EOS> after this something else happened instead and she went away but nobody noticed anything about it"
              numberOfSamples = 12
              numerified = encodeContextCrop(context, "", replicates=numberOfSamples)
              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
        

              # Run the noise model
              if True:
                 assert SANITY in ["Model", "ModelTmp"]
                 numeric, numeric_noised = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 print(numeric.size(), numeric_noised.size(), numerified.size())

              numeric = numeric.unsqueeze(2).expand(-1, -1, IMPORTANCE_SAMPLING_K).contiguous().view(-1, numberOfSamples*IMPORTANCE_SAMPLING_K)
              numeric_noised_original = numeric_noised
              numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, IMPORTANCE_SAMPLING_K).contiguous().view(-1, numberOfSamples*IMPORTANCE_SAMPLING_K)
              print("numeric_noised_original", numeric_noised_original.size(), numeric_noised.size())
              # Get samples from the reconstruction posterior


              # First, get reconstructions when the new word is UNKNOWN
              numeric_noised[args.sequence_length] = 0 # A simple hack for dealing with the issue that the last word 
              # Now get samples from the amortized reconstruction posterior
              print("NOISED: ", " ".join([itos_total[int(x)] for x in numeric_noised[:,0].cpu()]))
              print(704, numeric_noised.size())
              result, resultNumeric, fractions, thatProbs, amortizedPosterior = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*IMPORTANCE_SAMPLING_K)
              resultNumeric = resultNumeric.transpose(0,1).contiguous()
#              for z in range(21):
 #                print(z, itos_total[resultNumeric[z,0]], itos_total[numeric_noised[z,0]])
              for z in range(resultNumeric.size()[1]):
                 if z % 2 == 0:
                   print("LCSTerm", z, " ".join([itos_total[resultNumeric[r,z]] for r in range(args.sequence_length+1)])) #, itos_total[numeric_noised_second[z,0]])

              likelihood = getLikelihoodDP(resultNumeric, numeric_noised, resultNumeric.cpu(), IMPORTANCE_SAMPLING_K, numberOfSamples)


#              likelihood = memory.compute_likelihood(resultNumeric, numeric_noised, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=1, computeProbabilityStartingFrom=pointWhereToStart, expandReplicates=False)
              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*IMPORTANCE_SAMPLING_K)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()

#      resultNumeric_gpu = resultNumeric
#      resultNumeric = resultNumeric.cpu()
#      likelihood = getLikelihoodDP(resultNumeric_gpu, numeric_noised, resultNumeric)
#      resultNumeric_cpu = resultNumeric.detach().cpu()                                                                                                                                              



              # Second, get reconstructions when the new word is KNOWN
#              print(numeric_noised.size(), nextWord.size(), resultNumeric.size())
              numeric_noised_second = torch.cat([numeric_noised[1:-1], nextWord, nextWord], dim=0).contiguous() # Make the new word available
              numeric_noised_second[args.sequence_length] = 0 # A simple hack for dealing with the issue that the last word 
              numeric_second = torch.cat([numeric_noised[1:-1], nextWord, nextWord], dim=0).contiguous()
              result2, resultNumeric2, fractions2, thatProbs2, amortizedPosterior2 = autoencoder.sampleReconstructions(numeric_second, numeric_noised_second, None, 2, numberOfBatches=numberOfSamples*IMPORTANCE_SAMPLING_K)
#              print(result2[0])
 #             print(result2[1])
  #            print(result2[2])
   #           print(result2[3])
              resultNumeric2 = resultNumeric2.transpose(0,1).contiguous()
              resultNumeric2 = torch.cat([numeric[:1], resultNumeric2[:-1]], dim=0)
              print(" ".join([itos_total[int(resultNumeric[i,0])] for i in range(args.sequence_length+1)]))
              print(" ".join([itos_total[int(resultNumeric2[i,0])] for i in range(args.sequence_length+1)]))
   #           assert itos_total[int(resultNumeric2[20,0])] in ["OOV", remainingInput[i]], (itos_total[int(resultNumeric2[20,0])] , remainingInput[i])
             
              numeric_noised_second = torch.cat([numeric[:1], numeric_noised_second[:-2], numeric_noised_second[-1:]], dim=0)
              print(" ".join([itos_total[int(numeric_noised_second[i,0])] for i in range(args.sequence_length+1)]))
    #          assert itos_total[int(numeric_noised_second[20,0])] == "<SOS>"

#              print(resultNumeric2.size(), numeric.size(), numeric_noised_second.size())
 #             print(resultNumeric.size(), resultNumeric2.size())
              for z in range(resultNumeric2.size()[1]):
#                 if z % 2 == 0:
                 print("SecondTerm", z, " ".join([itos_total[resultNumeric2[r,z]] for r in range(args.sequence_length+1)])) #, itos_total[numeric_noised_second[z,0]])
#                 assert "<EOS" in " ".join([itos_total[resultNumeric2[r,z]] for r in range(args.sequence_length+1)])
              likelihood2 = getLikelihoodDP(resultNumeric2, numeric_noised_second, resultNumeric2.cpu(), IMPORTANCE_SAMPLING_K, numberOfSamples)
#              likelihood2 = memory.compute_likelihood(resultNumeric2, numeric_noised_second, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=1, computeProbabilityStartingFrom=pointWhereToStart, expandReplicates=False)
              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*IMPORTANCE_SAMPLING_K)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()


              likelihood = likelihood.view(numberOfSamples, IMPORTANCE_SAMPLING_K)
              likelihood2 = likelihood2.view(numberOfSamples, IMPORTANCE_SAMPLING_K)

              amortizedPosterior = amortizedPosterior.view(numberOfSamples, IMPORTANCE_SAMPLING_K)
              amortizedPosterior2 = amortizedPosterior2.view(numberOfSamples, IMPORTANCE_SAMPLING_K)

              resultNumeric = resultNumeric.view(args.sequence_length+1, numberOfSamples, IMPORTANCE_SAMPLING_K)
              resultNumeric2 = resultNumeric2.view(args.sequence_length+1, numberOfSamples, IMPORTANCE_SAMPLING_K)

              numeric_noised = numeric_noised.view(args.sequence_length+1, numberOfSamples, IMPORTANCE_SAMPLING_K)
              numeric_noised_second = numeric_noised_second.view(args.sequence_length+1, numberOfSamples, IMPORTANCE_SAMPLING_K)

              likelihood = torch.cat([likelihood, likelihood2], dim=1).view(numberOfSamples*2*IMPORTANCE_SAMPLING_K)
              amortizedPosterior = torch.cat([amortizedPosterior, amortizedPosterior2], dim=1).view(numberOfSamples*2*IMPORTANCE_SAMPLING_K) - 0.6931472 # we take the mixture of two proposal distributions, one aware of wr and the other one not
              resultNumeric = torch.cat([resultNumeric, resultNumeric2], dim=2).view(args.sequence_length+1, numberOfSamples*2*IMPORTANCE_SAMPLING_K)
              numeric_noised = torch.cat([numeric_noised, numeric_noised_second], dim=2).view(args.sequence_length+1, numberOfSamples*2*IMPORTANCE_SAMPLING_K)
              result_ = []
              for g in range(numberOfSamples):
                 for h in range(IMPORTANCE_SAMPLING_K):
                  result_.append(result[g*IMPORTANCE_SAMPLING_K+h])
                 for h in range(IMPORTANCE_SAMPLING_K):
                  result_.append(result2[g*IMPORTANCE_SAMPLING_K+h])
              result = result_
              # Evaluate the prior on these samples to estimate next-word surprisal

              resultNumeric_cpu = resultNumeric.detach().cpu()
              batch = [" ".join([itos_total[resultNumeric_cpu[r,s]] for r in range(0, resultNumeric.size()[0])]) for s in range(resultNumeric.size()[1])]
#              for h in range(len(batch)):
 #               print("line 795", h)
                #assert "<EOS>" in h
              batch = [x[x.index("<EOS>")+6:] if "<EOS>" in x else " " for x in batch]
              for h in range(len(batch)):
                 batch[h] = batch[h][:1].upper() + batch[h][1:]
                 if len(batch[h]) > 1:
                   assert batch[h][0] != " ", batch[h]
                 if h % 3 == 0:
                    print("FOR GPT2", h, "@"+batch[h]+"@")
#              quit()
#              # Add the preceding context
#              batchPreceding = [" ".join([itos_total[resultNumeric_cpu[r,s]] for r in range(0,pointWhereToStart+1)]) for s in range(resultNumeric.size()[1])]
#              batch = [x.replace(" .", ".")+" "+y for x, y in zip(batchPreceding, batch)]
#              print(batch)
              totalSurprisal = scoreWithGPT2.scoreSentences(batch)
              surprisals_past = torch.FloatTensor([x["past"] for x in totalSurprisal]).cuda().view(numberOfSamples, 2*IMPORTANCE_SAMPLING_K)
              surprisals_nextWord = torch.FloatTensor([x["next"] for x in totalSurprisal]).cuda().view(numberOfSamples, 2*IMPORTANCE_SAMPLING_K)

# The subsequent code snippet should be a plug-in for GPT2
#              totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False, computeSurprisals=True, returnLastSurprisal=False, numberOfBatches=numberOfSamples*IMPORTANCE_SAMPLING_K)
#              assert resultNumeric.size()[0] == args.sequence_length+1
#              assert totalSurprisal.size()[0] == args.sequence_length
#              # For each of the `numberOfSamples' many replicates, evaluate (i) the probability of the next word under the Monte Carlo estimate of the next-word posterior, (ii) the corresponding surprisal, (iii) the average of those surprisals across the 'numberOfSamples' many replicates.
#              totalSurprisal = totalSurprisal.view(args.sequence_length, numberOfSamples, IMPORTANCE_SAMPLING_K)
#              surprisals_past = totalSurprisal[:-1].sum(dim=0)
#              surprisals_nextWord = totalSurprisal[-1]

              # where numberOfSamples is how many samples we take from the noise model, and IMPORTANCE_SAMPLING_K is how many samples are drawn from the amortized posterior for each noised sample
              amortizedPosterior = amortizedPosterior.view(numberOfSamples, 2*IMPORTANCE_SAMPLING_K)
              likelihood = likelihood.view(numberOfSamples, 2*IMPORTANCE_SAMPLING_K)
    #          print(surprisals_past.size(), surprisals_nextWord.size(), amortizedPosterior.size(), likelihood.size())
   #           print(amortizedPosterior.mean(), likelihood.mean(), surprisals_past.mean(), surprisals_nextWord.mean())

              # PART 1: computing the other term: importance reweighting for P(TP|mt,wt)
              unnormalizedLogTruePosterior = likelihood.cuda() - surprisals_past - surprisals_nextWord
              assert float(unnormalizedLogTruePosterior.max()) <= 1e-5
              assert float(amortizedPosterior.max()) <= 1e-5
              log_importance_weights = unnormalizedLogTruePosterior - amortizedPosterior
              log_importance_weights_maxima, _ = log_importance_weights.max(dim=1, keepdim=True)
              print(log_importance_weights[0])
              for j in range(2*IMPORTANCE_SAMPLING_K): # TODO the importance weights seem wacky
                 if j % 2 != 0:
                    continue
                 print("TERM1", j, "@@", batch[j], float(surprisals_past[0, j]), float(surprisals_nextWord[0, j]), float(log_importance_weights[0, j]), float(likelihood[0, j]), float(amortizedPosterior[0, j]))
              print(" ".join([itos_total[int(x)] for x in numeric_noised[:, 0].detach().cpu()]))
              log_importance_weighted_probs_unnormalized = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima - surprisals_nextWord).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              log_importance_weights_sum = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              reweightedSurprisals = -(log_importance_weighted_probs_unnormalized - log_importance_weights_sum)
              reweightedSurprisalsMean = reweightedSurprisals.mean()
              surprisalOfNextWord_OtherTerm = surprisals_nextWord.mean(dim=1).mean()
              reweightedSurprisals_OtherTerm = reweightedSurprisalsMean

              #quit()

              # PART 2: computing LCS: importance reweighting for P(Tp|mt) and doing log-sum-exp
              unnormalizedLogTruePosterior = likelihood.cuda() - surprisals_past
              assert float(unnormalizedLogTruePosterior.max()) <= 1e-5
              assert float(amortizedPosterior.max()) <= 1e-5
              log_importance_weights = unnormalizedLogTruePosterior - amortizedPosterior
              log_importance_weights_maxima, _ = log_importance_weights.max(dim=1, keepdim=True)
              print(log_importance_weights[0])
              for j in range(2*IMPORTANCE_SAMPLING_K): # TODO the importance weights seem wacky
                 if j % 2 != 0:
                    continue
                 print("TERM2", j, "@@", batch[j], float(surprisals_past[0, j]), float(surprisals_nextWord[0, j]), float(log_importance_weights[0, j]), float(likelihood[0, j]), float(amortizedPosterior[0, j]))
              print(" ".join([itos_total[int(x)] for x in numeric_noised[:, 0].detach().cpu()]))
              log_importance_weighted_probs_unnormalized = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima - surprisals_nextWord).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              log_importance_weights_sum = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              reweightedSurprisals = -(log_importance_weighted_probs_unnormalized - log_importance_weights_sum)
              reweightedSurprisalsMean = reweightedSurprisals.mean()
              surprisalOfNextWord_LCS = surprisals_nextWord.exp().mean(dim=1).log().mean()
              reweightedSurprisals_LCS = reweightedSurprisalsMean


#              EIS_NextWord =  -(surprisalOfNextWord_OtherTerm - surprisalOfNextWord_LCS)
              EIS_NextWord =  -(reweightedSurprisals_OtherTerm - reweightedSurprisals_LCS)

#              print(surprisalOfNextWord_OtherTerm) # these are nonnegative, i.e. minus what they are in the formula
 #             print(surprisalOfNextWord_LCS)
              print("EIS", regions[i], EIS_NextWord, remainingInput[i])
#              continue

              # for printing
              nextWordSurprisal_cpu = surprisals_nextWord.view(-1).detach().cpu()
              reweightedSurprisal_cpu = reweightedSurprisals.detach().cpu()
#              print(nextWordSurprisal_cpu.size())


              for q in range(0, min(3*2*IMPORTANCE_SAMPLING_K, resultNumeric.size()[1]),  2*IMPORTANCE_SAMPLING_K):
                  print("DENOISED PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in resultNumeric[:,q]]), float(nextWordSurprisal_cpu[q])) #, float(reweightedSurprisal_cpu[q//IMPORTANCE_SAMPLING_K]))
              print("SURPRISAL", i, regions[i], remainingInput[i],float( surprisalOfNextWord_LCS), float(reweightedSurprisalsMean))
              print("\t".join([str(w) for w in [sentenceID, ITEM, CONDITION, regions[i], remainingInput[i], round(float( EIS_NextWord),3), round(float( reweightedSurprisalsMean),3), repetition]]), file=outFile)


getSurprisalsStimuli(SANITY="Model") #("Model" if args.deletion_rate > 0 else "ZeroLoss"))
#      quit(
