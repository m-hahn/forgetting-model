# Based on:
#  char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos12_Long.py (loss model & code for language model)
# And autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Both_Saving.py (autoencoder)
# And (for the plain LM): ../autoencoder/autoencoder2_mlp_bidir_AND_languagemodel_sample.py
print("Character aware!")

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=964163553) # language model taking noised input
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=878921872)
parser.add_argument("--load-from-plain-lm", dest="load_from_plain_lm", type=str, default=45661490) # plain language model without noise

import random

parser.add_argument("--batchSize", type=int, default=random.choice([1]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_lm", type=int, default=random.choice([1024]))
parser.add_argument("--hidden_dim_autoencoder", type=int, default=random.choice([512]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.000002, 0.00001, 0.00002, 0.00005])) #, 0.0001, 0.0002 # 1e-7, 0.000001, 0.000002, 0.000005, 0.000007, 
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.001, 0.01, 0.1, 0.2])) # 0.0001, 
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([20]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--deletion_rate", type=float, default=0.5)

parser.add_argument("--predictability_weight", type=float, default=random.choice([0.25, 0.5, 0.75]))


parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--NUMBER_OF_REPLICATES", type=int, default=random.choice([12,20]))

parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
TRAIN_LM = False
assert not TRAIN_LM

parser.add_argument("--RATE_WEIGHT", type=float, default=random.choice([-1.0])) #[3.0, 3.5, 4.0, 4.5, 5.0]))
# 1.5, 2.0, 2.5, 

#[1.25, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0])) # 0.5, 0.75, 1.0,  ==> this is essentially the point at which showing is better than guessing
parser.add_argument("--momentum", type=float, default=random.choice([0.0, 0.3, 0.5, 0.7, 0.9]))
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

parser.add_argument("--tuning", type=int, default=0) #random.choice([0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.001])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

model = "REAL_REAL"

import math

args=parser.parse_args()

############################

assert args.predictability_weight >= 0
assert args.predictability_weight <= 1
assert args.deletion_rate > 0.15
assert args.deletion_rate < 0.9

#############################

assert args.tuning in [0,1]
assert args.batchSize == 1
print(args.myID)
import sys
if args.tuning == 1:
   sys.stdout = open("/u/scr/mhahn/reinforce-logs-both/full-logs/"+__file__+"_"+str(args.myID), "w")

print(args)
print(args, file=sys.stderr)



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


class PlainLanguageModel(torch.nn.Module):
  def __init__(self):
      super(PlainLanguageModel, self).__init__()
      self.rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_lm, args.layer_num).cuda()
      self.output = torch.nn.Linear(args.hidden_dim_lm, len(itos)+3).cuda()
      self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()
      self.logsoftmax = torch.nn.LogSoftmax(dim=2)
      self.softmax = torch.nn.Softmax(dim=2)

      self.train_loss = torch.nn.NLLLoss(ignore_index=0)
      self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
      self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
      self.train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
      self.modules = [self.rnn, self.output, self.word_embeddings]
      #self.learning_rate = args.learning_rate
      #self.optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.0) # 0.02, 0.9
      self.zeroBeginning = torch.LongTensor([0 for _ in range(args.NUMBER_OF_REPLICATES*args.batchSize)]).cuda().view(1,args.NUMBER_OF_REPLICATES*args.batchSize)
      self.beginning = None
      self.zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()
      self.zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim_lm)).cuda()
      self.bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())
      self.bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
      self.bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim_lm)]).cuda())

      self.hidden = None

  def parameters(self):
     for module in self.modules:
         for param in module.parameters():
              yield param

  def sample(self, numeric):
     print("FOR SAMPLING", numeric.size())
     embedded = self.word_embeddings(numeric.unsqueeze(0))
     results = ["" for _ in range(args.NUMBER_OF_REPLICATES*args.batchSize)]     
     for _ in range(10): 
        out, self.hidden = self.rnn(embedded, self.hidden)
        logits = self.output(out) 
        probs = self.softmax(logits)
        print(probs.size())
        dist = torch.distributions.Categorical(probs=probs)
         
        nextWord = (dist.sample())
        nextWordStrings = [itos_total[x] for x in nextWord.cpu().numpy()[0]]
        for i in range(args.NUMBER_OF_REPLICATES*args.batchSize):
            results[i] += " "+nextWordStrings[i]
        embedded = self.word_embeddings(nextWord)
     return results


  def forward(self, numeric, train=True, printHere=False):
       if self.hidden is None or True:
           self.hidden = None
           self.beginning = self.zeroBeginning
       elif self.hidden is not None:
           hidden1 = Variable(self.hidden[0]).detach()
           hidden2 = Variable(self.hidden[1]).detach()
           forRestart = bernoulli.sample()
           hidden1 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden1)
           hidden2 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden2)
           self.hidden = (hidden1, hidden2)
           self.beginning = torch.where(forRestart.unsqueeze(0) == 1, zeroBeginning, self.beginning)
       print("BEGINNING", "NUMERIC", self.beginning.size(), numeric.size())
       numeric = torch.cat([self.beginning, numeric.t()], dim=0)
       self.beginning = numeric[numeric.size()[0]-1].view(1, args.NUMBER_OF_REPLICATES*args.batchSize)
       input_tensor = Variable(numeric[:-1], requires_grad=False)
       target_tensor = Variable(numeric[1:], requires_grad=False)
       embedded = self.word_embeddings(input_tensor)
#       if train:
#          embedded = self.char_dropout(embedded)
#          mask = self.bernoulli_input.sample()
#          mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
#          embedded = embedded * mask
  
       out, self.hidden = self.rnn(embedded, self.hidden)
   
#       if train:
#         mask = self.bernoulli_output.sample()
#         mask = mask.view(1, args.batchSize, args.hidden_dim_lm)
#         out = out * mask
   
       logits = self.output(out) 
       log_probs = self.logsoftmax(logits)
        
       loss = self.train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))
  
       lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.NUMBER_OF_REPLICATES*args.batchSize)

       if printHere:
          lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
          losses = lossTensor.data.cpu().numpy()
          numericCPU = numeric.cpu().data.numpy()
          print(("NONE", itos_total[numericCPU[0][0]]))
          for i in range((args.sequence_length)):
             print((losses[i][0], itos_total[numericCPU[i+1][0]]))
       samples = self.sample(numeric[-1])
       return lossTensor, target_tensor.view(-1).size()[0], samples, log_probs
   



class Autoencoder:
  def __init__(self):
    self.rnn_encoder = torch.nn.LSTM(2*args.word_embedding_size, int(args.hidden_dim_autoencoder/2.0), args.layer_num, bidirectional=True).cuda()
    self.rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_autoencoder, args.layer_num).cuda()
    self.output = torch.nn.Linear(args.hidden_dim_autoencoder, len(itos)+3).cuda()
    self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()
    self.logsoftmax = torch.nn.LogSoftmax(dim=2)
    self.softmax = torch.nn.Softmax(dim=2)
    self.attention_softmax = torch.nn.Softmax(dim=1)
    self.train_loss = torch.nn.NLLLoss(ignore_index=0)
    self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
    self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
    self.attention_proj = torch.nn.Linear(args.hidden_dim_autoencoder, args.hidden_dim_autoencoder, bias=False).cuda()
    self.attention_proj.weight.data.fill_(0)
    self.output_mlp = torch.nn.Linear(2*args.hidden_dim_autoencoder, args.hidden_dim_autoencoder).cuda()
    self.relu = torch.nn.ReLU()
    self.modules_autoencoder = [self.rnn_decoder, self.rnn_encoder, self.output, self.word_embeddings, self.attention_proj, self.output_mlp]
     

autoencoder = Autoencoder()

class LanguageModel:
   def __init__(self):
      self.rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_lm, args.layer_num).cuda()
      self.rnn_drop = self.rnn
      self.output = torch.nn.Linear(args.hidden_dim_lm, len(itos)+3).cuda()
      self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()
      self.logsoftmax = torch.nn.LogSoftmax(dim=2)
      self.train_loss = torch.nn.NLLLoss(ignore_index=0)
      self.print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
      self.char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)
      self.train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
      self.modules_lm = [self.rnn, self.output, self.word_embeddings]


lm = LanguageModel()

#character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total)+3, embedding_dim=args.char_emb_dim).cuda()

class MemoryModel():
  def __init__(self):
     self.memory_mlp_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.memory_mlp_inner_bilinear = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.memory_mlp_inner_from_pos = torch.nn.Linear(256, 500).cuda()
     self.memory_mlp_outer = torch.nn.Linear(500, 1).cuda()
     self.sigmoid = torch.nn.Sigmoid()
     self.relu = torch.nn.ReLU()
     self.positional_embeddings = torch.nn.Embedding(num_embeddings=args.sequence_length+2, embedding_dim=256).cuda()
     self.memory_word_pos_inter = torch.nn.Linear(256, 1, bias=False).cuda()
     self.memory_word_pos_inter.weight.data.fill_(0)
     self.perword_baseline_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
     self.perword_baseline_outer = torch.nn.Linear(500, 1).cuda()
     self.memory_bilinear = torch.nn.Linear(256, 500, bias=False).cuda()
     self.memory_bilinear.weight.data.fill_(0)
     self.modules_memory = [self.memory_mlp_inner, self.memory_mlp_outer, self.memory_mlp_inner_from_pos, self.positional_embeddings, self.perword_baseline_inner, self.perword_baseline_outer, self.memory_word_pos_inter, self.memory_bilinear, self.memory_mlp_inner_bilinear]


memory = MemoryModel()

def parameters_memory():
   for module in memory.modules_memory:
       for param in module.parameters():
            yield param

dual_weight = torch.cuda.FloatTensor([1.0])
dual_weight.requires_grad=True


parameters_memory_cached = [x for x in parameters_memory()]




def parameters_autoencoder():
   for module in autoencoder.modules_autoencoder:
       for param in module.parameters():
            yield param



def parameters_lm():
   for module in lm.modules_lm:
       for param in module.parameters():
            yield param

parameters_lm_cached = [x for x in parameters_lm()]


#learning_rate = args.learning_rate

assert not TRAIN_LM
optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9

#named_modules = {"rnn" : rnn, "output" : output, "word_embeddings" : word_embeddings, "optim" : optim}


if args.load_from_autoencoder is not None:
  try:
     checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  except FileNotFoundError:
     checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["components"][i])
  



if args.load_from_lm is not None:
  lm_file = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure.py"
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+lm_file+"_code_"+str(args.load_from_lm)+".txt")
  for i in range(len(checkpoint["components"])):
      lm.modules_lm[i].load_state_dict(checkpoint["components"][i])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout



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
         numerified.append((stoi[char]+3 if char in stoi else 2))
         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])

       if len(numerified) > (args.batchSize*(args.sequence_length+1)):
         sequenceLengthHere = args.sequence_length+1

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerifiedCurrent_chars = numerified_chars[:cutoff]

         for i in range(len(numerifiedCurrent_chars)):
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i][:15] + [1]
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i] + ([0]*(16-len(numerifiedCurrent_chars[i])))


         numerified = numerified[cutoff:]
         numerified_chars = numerified_chars[cutoff:]
       
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(args.batchSize, -1, sequenceLengthHere, 16).transpose(0,1).transpose(1,2).cuda()

         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], numerifiedCurrent_chars[i]
         hidden = None
       else:
         print("Skipping")




hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.NUMBER_OF_REPLICATES*args.batchSize)]).cuda().view(1,args.NUMBER_OF_REPLICATES*args.batchSize)
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


#zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

#bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
#bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())

#runningAveragePredictionLoss = 1.0
runningAverageReward = 5.0
runningAverageBaselineDeviation = 2.0
runningAveragePredictionLoss = 5.0
runningAverageReconstructionLoss = 5.0
expectedRetentionRate = 0.5



def forward(numeric, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False):
      global hidden
      global beginning
      global beginning_chars
      if True:
          hidden = None
          beginning = zeroBeginning

      numeric, numeric_chars = numeric

      ######################################################
      ######################################################
      # Run Loss Model

      numeric = numeric.expand(-1, args.NUMBER_OF_REPLICATES)
      numeric = torch.cat([beginning, numeric], dim=0)
      embedded_everything = lm.word_embeddings(numeric)

      # Positional embeddings
      numeric_positions = torch.LongTensor(range(args.sequence_length+2)).cuda().unsqueeze(1)
      embedded_positions = memory.positional_embeddings(numeric_positions)
      numeric_embedded = memory.memory_word_pos_inter(embedded_positions)

      # Retention probabilities
      memory_byword_inner = memory.memory_mlp_inner(embedded_everything.detach())
      memory_hidden_logit_per_wordtype = memory.memory_mlp_outer(memory.relu(memory_byword_inner))

  #    print(embedded_positions.size(), embedded_everything.size())
 #     print(memory.memory_bilinear(embedded_positions).size())
#      print(memory.relu(memory.memory_mlp_inner_bilinear(embedded_everything.detach())).transpose(1,2).size())
      attention_bilinear_term = torch.bmm(memory.memory_bilinear(embedded_positions), memory.relu(memory.memory_mlp_inner_bilinear(embedded_everything.detach())).transpose(1,2)).transpose(1,2)

      memory_hidden_logit = numeric_embedded + memory_hidden_logit_per_wordtype + attention_bilinear_term
      memory_hidden = memory.sigmoid(memory_hidden_logit)
      if provideAttention:
         return memory_hidden

      # Baseline predictions for prediction loss
      baselineValues = 10*memory.sigmoid(memory.perword_baseline_outer(memory.relu(memory.perword_baseline_inner(embedded_everything[-1].detach())))).squeeze(1)
      assert tuple(baselineValues.size()) == (args.NUMBER_OF_REPLICATES,)


      # NOISE MEMORY ACCORDING TO MODEL
      memory_filter = torch.bernoulli(input=memory_hidden)
      bernoulli_logprob = torch.where(memory_filter == 1, torch.log(memory_hidden+1e-10), torch.log(1-memory_hidden+1e-10))
      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)
      if args.entropy_weight > 0:
         entropy = -(memory_hidden * torch.log(memory_hidden+1e-10) + (1-memory_hidden) * torch.log(1-memory_hidden+1e-10)).mean()
      else:
         entropy=-1.0
      memory_filter = memory_filter.squeeze(2)

      ####################################################################################
      numeric_noised = torch.where(memory_filter==1, numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]
      numeric_onlyNoisedOnes = torch.where(memory_filter == 0, numeric, 0*numeric) # target is 0 in those places where no noise has happened

      if onlyProvideMemoryResult:
        return numeric, numeric_noised

      input_tensor_pure = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      target_tensor_full = Variable(numeric[1:], requires_grad=False)

      target_tensor_onlyNoised = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)
      #####################################################################################


      ##########################################
      ##########################################
      # RUN AUTOENCODER (approximately inverting loss model)
      autoencoder_embedded = autoencoder.word_embeddings(input_tensor_pure[:-1])
      autoencoder_embedded_noised = autoencoder.word_embeddings(input_tensor_noised[:-1])
      autoencoder_out_encoder, _ = autoencoder.rnn_encoder(autoencoder_embedded_noised, None)
      autoencoder_out_decoder, _ = autoencoder.rnn_decoder(autoencoder_embedded, None)

      autoencoder_attention = torch.bmm(autoencoder.attention_proj(autoencoder_out_encoder).transpose(0,1), autoencoder_out_decoder.transpose(0,1).transpose(1,2))
      autoencoder_attention = autoencoder.attention_softmax(autoencoder_attention).transpose(0,1)
      autoencoder_from_encoder = (autoencoder_out_encoder.unsqueeze(2) * autoencoder_attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      autoencoder_out_full = torch.cat([autoencoder_out_decoder, autoencoder_from_encoder], dim=2)


      autoencoder_logits = autoencoder.output(autoencoder.relu(autoencoder.output_mlp(autoencoder_out_full) ))
      autoencoder_log_probs = autoencoder.logsoftmax(autoencoder_logits)

      # Prediction Loss 
      autoencoder_lossTensor = autoencoder.print_loss(autoencoder_log_probs.view(-1, len(itos)+3), target_tensor_onlyNoised[:-1].view(-1)).view(-1, args.NUMBER_OF_REPLICATES*args.batchSize)

      ##########################################
      ##########################################
      # RUN LANGUAGE MODEL (amortized prediction of next word)
      if args.predictability_weight > 0:
       lm_embedded = lm.word_embeddings(input_tensor_noised)
       lm_out, lm_hidden = lm.rnn_drop(lm_embedded, None)
       lm_out = lm_out[-1:]
       lm_logits = lm.output(lm_out) 
       lm_log_probs = lm.logsoftmax(lm_logits)
 
       # Prediction Loss 
       lm_lossTensor = lm.print_loss(lm_log_probs.view(-1, len(itos)+3), target_tensor_full[-1].view(-1)).view(-1, args.NUMBER_OF_REPLICATES) # , args.batchSize is 1
      ##########################################
      ##########################################

      # Reward, term 1
      if args.predictability_weight > 0:
        negativeRewardsTerm1 = 2*args.predictability_weight * lm_lossTensor.mean(dim=0) + 2*(1-args.predictability_weight) * autoencoder_lossTensor.mean(dim=0)
      else:
        negativeRewardsTerm1 = autoencoder_lossTensor.mean(dim=0)

#      print(autoencoder_lossTensor.mean(dim=0))

      # Reward, term 2
      # Regularization towards lower retention rates
      negativeRewardsTerm2 = memory_filter.mean(dim=0)
      retentionTarget = 1-args.deletion_rate
      loss = 0

      # Autoencoder Loss
      loss += autoencoder_lossTensor.mean()

      # Overall Reward
      negativeRewardsTerm = negativeRewardsTerm1 + dual_weight * (negativeRewardsTerm2-retentionTarget)
      # for the dual weight
      loss += (dual_weight * (negativeRewardsTerm2-retentionTarget).detach()).mean()
      if printHere:
          print(negativeRewardsTerm1.mean(), dual_weight, negativeRewardsTerm2.mean(), retentionTarget)
      #print(loss)

      # baselineValues: the baselines for the prediction loss (term 1)
      # memory_hidden: baseline for term 2
      # Important to detach all but the baseline values

      # Reward Minus Baseline
      # Detached surprisal and mean retention
#      rewardMinusBaseline = (negativeRewardsTerm.detach() - baselineValues - args.RATE_WEIGHT * memory_hidden.mean(dim=0).squeeze(dim=1).detach())
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

      if args.predictability_weight > 0:
       runningAveragePredictionLoss = factor * runningAveragePredictionLoss + (1-factor) * round(float(lm_lossTensor.mean()),3)
      runningAverageReconstructionLoss = factor * runningAverageReconstructionLoss + (1-factor) * round(float(autoencoder_lossTensor.mean()),3)
      runningAverageReward = factor * runningAverageReward + (1-factor) * float(negativeRewardsTerm.mean())
      ############################

      if printHere:
         if args.predictability_weight > 0:
          lm_losses = lm_lossTensor.data.cpu().numpy()
         autoencoder_losses = autoencoder_lossTensor.data.cpu().numpy()

         numericCPU = numeric.cpu().data.numpy()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy()
         memory_hidden_CPU = memory_hidden[:,0,0].cpu().data.numpy()
         memory_hidden_logit_per_wordtype_cpu = memory_hidden_logit_per_wordtype.cpu().data
         attention_bilinear_term = attention_bilinear_term.cpu().data
         numeric_embedded_cpu = numeric_embedded.cpu().data
         print(("NONE", itos_total[numericCPU[0][0]]))
         for i in range((args.sequence_length+1)):
            print(autoencoder_losses[i][0] if i < args.sequence_length else "--", "\t", lm_losses[0][0] if args.predictability_weight > 0 and i == args.sequence_length else "---" , "\t", itos_total[numericCPU[i+1][0]],"\t", itos_total[numeric_noisedCPU[i+1][0]],"\t", memory_hidden_CPU[i+1],"\t", float(baselineValues[0]) if i == args.sequence_length else "","\t", float(numeric_embedded_cpu[i+1,0,0]),"\t", float(memory_hidden_logit_per_wordtype_cpu[i+1,0,0]),"\t", float(attention_bilinear_term[i+1,0,0]))
#            print((, itos_total[numericCPU[i+1][0]], itos_total[numeric_noisedCPU[i+1][0]], memory_hidden_CPU[i+1]))


         if args.predictability_weight > 0:
          print(lm_lossTensor.view(-1))
         print(baselineValues.view(-1))
 #        if args.predictability_weight > 0:
#          print("EMPIRICAL DEVIATION FROM BASELINE", (lm_lossTensor-baselineValues).abs().mean())
               
         print("PREDICTION_LOSS", runningAveragePredictionLoss, "RECONSTRUCTION_LOSS", runningAverageReconstructionLoss, "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", expectedRetentionRate, "\tDEVIATION FROM BASELINE", runningAverageBaselineDeviation, "\tREWARD", runningAverageReward, "\tENTROPY", float(entropy))
         print(dual_weight)
      if updatesCount % 5000 == 0:
         print("updatesCount", updatesCount, updatesCount/maxUpdates)
         print("\t".join([str(x) for x in ("PREDICTION_LOSS", runningAveragePredictionLoss, "RECONSTRUCTION_LOSS", runningAverageReconstructionLoss, "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", expectedRetentionRate, "\tDEVIATION FROM BASELINE", runningAverageBaselineDeviation, "\tREWARD", runningAverageReward, "\tENTROPY", float(entropy))]), file=sys.stderr)

      #runningAveragePredictionLoss = 0.95 * runningAveragePredictionLoss + (1-0.95) * float(negativeRewardsTerm1.mean())

      return loss, target_tensor_full.view(-1).size()[0]



def sampleReconstructions(numeric, numeric_noised, NOUN, offset):
      if True:
          beginning = zeroBeginning

      numeric, _ = numeric

      input_tensor = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      #target_tensor = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = autoencoder.word_embeddings(input_tensor)

      embedded_noised = autoencoder.word_embeddings(input_tensor_noised)

      out_encoder, _ = autoencoder.rnn_encoder(embedded_noised, None)



      hidden = None
      result  = ["" for _ in range(args.batchSize*args.NUMBER_OF_REPLICATES)]
      result_numeric = [[] for _ in range(args.batchSize*args.NUMBER_OF_REPLICATES)]
      embeddedLast = embedded[0].unsqueeze(0)
      for i in range(args.sequence_length+1):
          out_decoder, hidden = autoencoder.rnn_decoder(embeddedLast, hidden)
    
          attention = torch.bmm(autoencoder.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = autoencoder.attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)

 #         print(input_tensor.size())


          logits = autoencoder.output(autoencoder.relu(autoencoder.output_mlp(out_full) )) 
          probs = autoencoder.softmax(logits)
          if i == 15-offset:
            assert args.sequence_length == 20
            thatProbs = float(probs[0,:, stoi["that"]+3].mean())
#          print(i, probs[0,:, stoi["that"]+3].mean())
 #         quit()

          dist = torch.distributions.Categorical(probs=probs)
       
#          nextWord = (dist.sample())
          nextWord = torch.where(numeric_noised[i] == 0, (dist.sample()), numeric[i:i+1])
  #        print(nextWord.size())
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for i in range(args.batchSize*args.NUMBER_OF_REPLICATES):
             result[i] += " "+nextWordStrings[i]
             result_numeric[i].append( nextWordDistCPU[i] )
          embeddedLast = autoencoder.word_embeddings(nextWord)
#          print(embeddedLast.size())
      for r in result[:10]:
         print(r)
      nounFraction = (float(len([x for x in result if NOUN in x]))/len(result))

      thatFraction = (float(len([x for x in result if NOUN+" that" in x]))/len(result))

      return result, torch.LongTensor(result_numeric).cuda(), (nounFraction, thatFraction), thatProbs




def backward(loss, printHere):
      optim_autoencoder.zero_grad()
      optim_memory.zero_grad()

      if dual_weight.grad is not None:
         dual_weight.grad.data.fill_(0.0)
      if printHere:
         print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_memory_cached, 5.0) #, norm_type="inf")
      if TRAIN_LM:
         torch.nn.utils.clip_grad_value_(parameters_lm_cached, 5.0) #, norm_type="inf")
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

maxUpdates = 5000000 if args.tuning == 1 else 10000000000

def showAttention(word):
    attention = forward((torch.cuda.LongTensor([stoi[word]+3 for _ in range(args.sequence_length+1)]).view(-1, 1), None), train=True, printHere=True, provideAttention=True)
    attention = attention[:,0,0]
    print(*(["SCORES", word, "\t"]+[round(x,2) for x in list(attention.cpu().data.numpy())]))







nounsAndVerbs = []
#nounsAndVerbs.append(["the school principal",       "the teacher",        "had an affair with",                     "had been fired",                     "was quoted in the newspaper"])
#nounsAndVerbs.append(["the famous sculptor",        "the painter",        "admired more than anyone",            "wasn't talented",                    "was completely untrue"])
#nounsAndVerbs.append(["the marketing whiz",  "the artist",         "had hired",                  "was a fraud",                        "shocked everyone"])
#nounsAndVerbs.append(["the marathon runner",         "the psychiatrist",       "treated for his illness",                "was actually doping",            "was ridiculous"])
#nounsAndVerbs.append(["the frightened child",           "the medic",          "rescued from the flood",    "was completely unharmed",            "relieved everyone"])
#nounsAndVerbs.append(["the alleged criminal",        "the officer",        "arrested after the murder",                  "was not in fact guilty",             "was bogus"])
#nounsAndVerbs.append(["the college student",         "the professor",      "accused of cheating",                     "was dropping the class",             "made the professor happy"])
#nounsAndVerbs.append(["the suspected mobster",         "the media",          "portrayed in detail",               "was on the run",                     "turned out to be true"])
#nounsAndVerbs.append(["the leading man",           "the starlet",        "fell in love with",                    "would miss the show",                "almost made her cry"])
#nounsAndVerbs.append(["the old preacher",        "the parishioners",   "fired yesterday",                     "stole money from the church",        "proved to be true"])
#nounsAndVerbs.append(["the young violinist",      "the sponsors",       "backed financially",                    "abused drugs",                       "is likely true"])
#nounsAndVerbs.append(["the conservative senator",        "the diplomat",       "opposed in the election",                   "won in the run-off",                   "really made him angry"])

nounsAndVerbs.append(["the senator",        "the diplomat",       "opposed"])

#nounsAndVerbs = nounsAndVerbs[:1]

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

topNouns.append("announcement")
topNouns.append("suspicion")
topNouns.append("allegation")
topNouns.append("realization")
topNouns.append("indication")
topNouns.append("remark")
topNouns.append("speculation")
topNouns.append("assurance")
topNouns.append("presumption")
topNouns.append("concern")
topNouns.append("finding")
topNouns.append("assertion")
topNouns.append("feeling")
topNouns.append("perception")
topNouns.append("statement")
topNouns.append("assumption")
topNouns.append("conclusion")


topNouns.append("report")
topNouns.append("story")
#topNouns.append("disclosure")
topNouns.append("confirmation")   
topNouns.append("information")
topNouns.append("evidence")
topNouns.append("reminder")
topNouns.append("rumor")
topNouns.append("thought")
topNouns.append("suggestion")
topNouns.append("revelation")    
topNouns.append("belief")
#topNouns.append("inkling") # this is OOV for the model
topNouns.append("suspicion")
topNouns.append("idea")
topNouns.append("claim")
topNouns.append("news")
topNouns.append("proof")
topNouns.append("admission")
topNouns.append("declaration")



with open("../../../forgetting/fromCorpus_counts.csv", "r") as inFile:
   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = counts[0]
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[0] : line[1:] for line in counts}

topNouns = [x for x in topNouns if x in counts]
topNouns = sorted(list(set(topNouns)), key=lambda x:float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]]))

print(topNouns)
print(len(topNouns))



def getPerNounReconstructionsSanity():
    fractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = []
    
           for condition in [0]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]}"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("###########")
              surprisalsPerRun = []
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, _ = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 0)
                 (nounFraction, thatFraction) = fractions
                 thatFractions.append(math.log(thatProbs))
    
                 
              print(thatFractions)
              print("NOUNS SO FAR", topNouns.index(NOUN))
    
         fractionsPerNoun.append((NOUN, sum(thatFractions)/len(thatFractions)))
    print("FRACTIONS_PER_NOUN, WITHOUT VERB, SANITY NEW")
    print(fractionsPerNoun)
    
def getPerNounReconstructionsSanityVerb():
    fractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = []
    
           for condition in [0]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} knew"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("###########")
              surprisalsPerRun = []
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, _ = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 1)
                 (nounFraction, thatFraction) = fractions
                 thatFractions.append(math.log(thatProbs))
    
                 
              print(thatFractions)
              print("NOUNS SO FAR", topNouns.index(NOUN))
    
         fractionsPerNoun.append((NOUN, sum(thatFractions)/len(thatFractions)))
    print("FRACTIONS_PER_NOUN, WITH VERB, SANITY NEW")
    print(fractionsPerNoun)
    

def getPerNounReconstructionsSanity2Verbs():
    fractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = []
    
           for condition in [0]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} knew was"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("###########")
              surprisalsPerRun = []
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, _ = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 2)
                 (nounFraction, thatFraction) = fractions
                 thatFractions.append(math.log(thatProbs))
    
                 
              print(thatFractions)
              print("NOUNS SO FAR", topNouns.index(NOUN))
    
         fractionsPerNoun.append((NOUN, sum(thatFractions)/len(thatFractions)))
    print("FRACTIONS_PER_NOUN, WITH 2 VERBS, SANITY NEW")
    print(fractionsPerNoun)
    



def getPerNounReconstructions():
    fractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = []
    
           for condition in [0]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]}"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("===========")
              surprisalsPerRun = []
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, numeric_noised = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 0)
                 (nounFraction, thatFraction) = fractions
                 thatFractions.append(math.log(thatProbs))
    
                 
              print(thatFractions)
              print("NOUNS SO FAR", topNouns.index(NOUN))
    
         fractionsPerNoun.append((NOUN, sum(thatFractions)/len(thatFractions)))
    print("FRACTIONS_PER_NOUN, WITHOUT VERB NEW")
    print(fractionsPerNoun)
    
    
 
def getPerNounReconstructionsVerb():
    fractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = []
    
           for condition in [0]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} knew"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("===========")
              surprisalsPerRun = []
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, numeric_noised = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 1)
                 (nounFraction, thatFraction) = fractions
                 thatFractions.append(math.log(thatProbs))
    
                 
              print(thatFractions)
              print("NOUNS SO FAR", topNouns.index(NOUN))
    
         fractionsPerNoun.append((NOUN, sum(thatFractions)/len(thatFractions)))
    print("FRACTIONS_PER_NOUN, WITH VERB NEW")
    print(fractionsPerNoun)
    
    
def getPerNounReconstructions2Verbs():
    fractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = []
    
           for condition in [0]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} knew was"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("===========")
              surprisalsPerRun = []
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, numeric_noised = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 2)
                 (nounFraction, thatFraction) = fractions
                 thatFractions.append(math.log(thatProbs))
    
                 
              print(thatFractions)
              print("NOUNS SO FAR", topNouns.index(NOUN))
    
         fractionsPerNoun.append((NOUN, sum(thatFractions)/len(thatFractions)))
    print("FRACTIONS_PER_NOUN, WITH TWO VERBS NEW")
    print(fractionsPerNoun)
    
    
plain_lm = PlainLanguageModel()
plain_lmFileName = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars.py"

if args.load_from_plain_lm is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+plain_lmFileName+"_code_"+str(args.load_from_plain_lm)+".txt")
  for i in range(len(checkpoint["components"])):
      plain_lm.modules[i].load_state_dict(checkpoint["components"][i])



def getPerNounReconstructionsSanity2VerbsUsingPlainLM3(): # Surprisal for EOS after 2 or 3 verbs
    print(plain_lm) 
    surprisalsPerNoun = []
    thatFractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = { 0 : [], 1 : []}
           surprisals = { 0 : [], 1 : []}

    
           for condition in [0,1]:
              if condition == 0 or True:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} won"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("###########")
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, _ = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 0 if condition == 0 else 0)
         
                 #print(resultNumeric)
                 #print(resultNumeric.size())
                 #print(itos[2033-3])
                 if condition == 0:
                   appended = ["won", "was", "true", "."]
                 else:
                   appended = ["was", "true", "."]
                 appended = torch.LongTensor([stoi[x]+3 for x in appended]).view(1, -1).expand(args.NUMBER_OF_REPLICATES*args.batchSize, -1).cuda()
                 #print(appended.size())
                 resultNumeric = torch.cat([resultNumeric, appended], dim=1)
                 resultNumeric = resultNumeric[:, -(1+args.sequence_length):]
                 #print(resultNumeric.size())
                 
                 #quit()
                 totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False)
                 print(totalSurprisal.size())
#                 quit()
  #               print("predictionsPlainLM", predictionsPlainLM.size())
   #              print(".", predictionsPlainLM[-1, :, stoi["."]+3].mean())
    #             print("was", predictionsPlainLM[-1, :, stoi["was"]+3].mean())
                 print(samplesFromLM)
      #           print(predictionsPlainLM.size())
                 (nounFraction, thatFraction) = fractions
                 thatFractions[condition].append(math.log(thatProbs))


    #             assert RUN == 0
                 if condition == 0:
    #                print(totalSurprisal)
    #                print(totalSurprisal[-4:])
    #                print(totalSurprisal[-4:].sum(dim=0))
    #                quit()
                    surprisals[condition].append(float(totalSurprisal[-1:, :].sum(dim=0).mean()))
                 else:
                    surprisals[condition].append(float(totalSurprisal[-1:, :].sum(dim=0).mean()))



                 
              print("NOUNS SO FAR", topNouns.index(NOUN))
         surprisals0 = sum(surprisals[0])/len(surprisals[0])
         surprisals1 = sum(surprisals[1])/len(surprisals[1])
         surprisalsPerNoun.append((NOUN, surprisals1, surprisals0))
         
         thatFractions0 = sum(thatFractions[0])/len(thatFractions[0])
         thatFractions1 = sum(thatFractions[1])/len(thatFractions[1])
         thatFractionsPerNoun.append((NOUN, thatFractions1, thatFractions0))
    print("SURPRISALS_PER_NOUN, WITH VERB, SANITY NEW")
    print(surprisalsPerNoun)
    print(thatFractionsPerNoun)
    print("surpUngramm = c("+",".join([str(x[1]) for x in surprisalsPerNoun])+")")
    print("surpGramm = c("+",".join([str(x[2]) for x in surprisalsPerNoun])+")")
    print("thatUngramm = c("+",".join([str(x[1]) for x in thatFractionsPerNoun])+")")
    print("thatGramm = c("+",".join([str(x[2]) for x in thatFractionsPerNoun])+")")
    quit()
   



def getPerNounReconstructionsSanity2VerbsUsingPlainLM2(): # Surprisal for EOS after 2 or 3 verbs
    print(plain_lm) 
    surprisalsPerNoun = []
    thatFractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = { 0 : [], 1 : []}
           surprisals = { 0 : [], 1 : []}

    
           for condition in [0,1]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} won was true . ."
              else:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} was true . ."
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("###########")
              for RUN in range(1): #args.NUMBER_OF_RUNS):
                 numeric, _ = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 5 if condition == 0 else 4)
#                 print(resultNumeric)
 #                print(resultNumeric.size())
                 totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False)
                 print(totalSurprisal.size())
#                 quit()
  #               print("predictionsPlainLM", predictionsPlainLM.size())
   #              print(".", predictionsPlainLM[-1, :, stoi["."]+3].mean())
    #             print("was", predictionsPlainLM[-1, :, stoi["was"]+3].mean())
                 print(samplesFromLM)
      #           print(predictionsPlainLM.size())
                 (nounFraction, thatFraction) = fractions
                 thatFractions[condition].append(math.log(thatProbs))


    #             assert RUN == 0
                 if condition == 0:
    #                print(totalSurprisal)
    #                print(totalSurprisal[-4:])
    #                print(totalSurprisal[-4:].sum(dim=0))
    #                quit()
                    surprisals[condition].append(float(totalSurprisal[-3:, :].sum(dim=0).mean()))
                 else:
                    surprisals[condition].append(float(totalSurprisal[-3:, :].sum(dim=0).mean()))



                 
              print("NOUNS SO FAR", topNouns.index(NOUN))
         surprisals0 = sum(surprisals[0])/len(surprisals[0])
         surprisals1 = sum(surprisals[1])/len(surprisals[1])
         surprisalsPerNoun.append((NOUN, surprisals1, surprisals0))
         
         thatFractions0 = sum(thatFractions[0])/len(thatFractions[0])
         thatFractions1 = sum(thatFractions[1])/len(thatFractions[1])
         thatFractionsPerNoun.append((NOUN, thatFractions1, thatFractions0))
    print("SURPRISALS_PER_NOUN, WITH VERB, SANITY NEW")
    print(surprisalsPerNoun)
    print(thatFractionsPerNoun)
    print("surpUngramm = c("+",".join([str(x[1]) for x in surprisalsPerNoun])+")")
    print("surpGramm = c("+",".join([str(x[2]) for x in surprisalsPerNoun])+")")
    print("thatUngramm = c("+",".join([str(x[1]) for x in thatFractionsPerNoun])+")")
    print("thatGramm = c("+",".join([str(x[2]) for x in thatFractionsPerNoun])+")")
    quit()
   


  
def getPerNounReconstructionsSanity2VerbsUsingPlainLM(): # Next-word prediction after observing 2 verbs
    print(plain_lm) 
    surprisalsPerNoun = []
    thatFractionsPerNoun = []
    for NOUN in topNouns:
    #     NOUN = "belief"
         
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
           thatFractions = []
    
           for condition in [0]:
              if condition == 0:
                 sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} won was"
              numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
              print(len(numerified))
              numerified = numerified[-args.sequence_length-1:]
              assert len(numerified) == args.sequence_length+1, len(numerified)
              numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
              print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
              print("###########")
              surprisals = []
              for RUN in range(20): #args.NUMBER_OF_RUNS):
                 numeric, _ = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
                 result, resultNumeric, fractions, thatProbs = sampleReconstructions((numeric, None), numeric_noised, NOUN, 2)
#                 print(resultNumeric)
 #                print(resultNumeric.size())
                 totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False)
  #               print("predictionsPlainLM", predictionsPlainLM.size())
   #              print(".", predictionsPlainLM[-1, :, stoi["."]+3].mean())
    #             print("was", predictionsPlainLM[-1, :, stoi["was"]+3].mean())
     #            print(samplesFromLM)
      #           print(predictionsPlainLM.size())
                 (nounFraction, thatFraction) = fractions
                 thatFractions.append(math.log(thatProbs))


    #             assert RUN == 0
                 surprisals.append((float(predictionsPlainLM[-1, :, stoi["."]+3].mean()) - float(predictionsPlainLM[-1, :, stoi["was"]+3].mean())))


                 
              print("NOUNS SO FAR", topNouns.index(NOUN))
                  
         surprisalsPerNoun.append((NOUN, sum(surprisals)/len(surprisals)))
         thatFractionsPerNoun.append((NOUN, sum(thatFractions)/len(thatFractions)))
    print("SURPRISALS_PER_NOUN, WITH VERB, SANITY NEW")
    print(surprisalsPerNoun)
    print(thatFractionsPerNoun)
    print([x[1] for x in surprisalsPerNoun])
    print([x[1] for x in thatFractionsPerNoun])
    quit()
   

    
#getPerNounReconstructionsSanity2VerbsUsingPlainLM()
#getPerNounReconstructionsSanity2VerbsUsingPlainLM2()
getPerNounReconstructionsSanity2VerbsUsingPlainLM3()
quit()
  
getPerNounReconstructionsSanity()
getPerNounReconstructionsSanityVerb()

for epoch in range(1000):
   print(epoch)
   training_data = corpusIteratorWikiWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   lm.rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   if updatesCount >= maxUpdates:
     break
   while updatesCount <= maxUpdates:
      counter += 1
      updatesCount += 1
      if updatesCount % 10000 == 0:
         getPerNounReconstructionsSanity()
         getPerNounReconstructionsSanityVerb()
         getPerNounReconstructions()
         getPerNounReconstructionsVerb()
         getPerNounReconstructions2Verbs()
         print("=========================")
         showAttention("the")
         showAttention("was")
         showAttention("that")
         showAttention("fact")
         showAttention("information")
         showAttention("report")
         showAttention("belief")
         showAttention("finding")
         showAttention("prediction")
         showAttention("of")
         showAttention("by")
         showAttention("about")


#      if updatesCount % 10000 == 0:
#         optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
#         optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9
#
      try:
         numeric = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
#      if loss.data.cpu().numpy() > 15.0:
#          lossHasBeenBad += 1
#      else:
#          lossHasBeenBad = 0
      if lossHasBeenBad > 100:
          print("Loss exploding, has been bad for a while")
          print(loss)
          assert False
      trainChars += charCounts 
      if printHere:
          print(("Loss here", loss))
          print((epoch,counter, trainChars))
          print("Dev losses")
          print(devLosses)
          print("Words per sec "+str(trainChars/(time.time()-startTime)))
          print(args.learning_rate_memory, args.learning_rate_autoencoder)
          print(lastSaved)
          print(__file__)
          print(args)

      if (time.time() - totalStartTime)/60 > 4000:
          print("Breaking early to get some result within 72 hours")
          totalStartTime = time.time()
          break

# #     break
#   rnn_drop.train(False)
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
#   #if args.save_to is not None:
# #     torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), MODELS_HOME+"/"+args.save_to+".pth.tar")
#
#   with open("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+args.language+"_"+__file__+"_model_"+str(args.myID)+"_"+model+".txt", "w") as outFile:
#       print(str(args), file=outFile)
#       print(" ".join([str(x) for x in devLosses]), file=outFile)
#
#   if len(devLosses) > 1 and devLosses[-1] > devLosses[-2]:
#      break
#
#   state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
#   torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")
#
#
#
#
#
#
#   learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
#   optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9




#      global runningAverageBaselineDeviation
#      global runningAveragePredictionLoss
#


with open("/u/scr/mhahn/reinforce-logs-both/results/"+__file__+"_"+str(args.myID), "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward, file=outFile)
   print(expectedRetentionRate, file=outFile)
   print(runningAverageBaselineDeviation, file=outFile)
   print(runningAveragePredictionLoss, file=outFile)
   print(runningAverageReconstructionLoss, file=outFile)


print("=========================")
showAttention("the")
showAttention("was")
showAttention("that")
showAttention("fact")
showAttention("information")
showAttention("report")
showAttention("belief")
showAttention("finding")
showAttention("prediction")
showAttention("of")
showAttention("by")
showAttention("about")


