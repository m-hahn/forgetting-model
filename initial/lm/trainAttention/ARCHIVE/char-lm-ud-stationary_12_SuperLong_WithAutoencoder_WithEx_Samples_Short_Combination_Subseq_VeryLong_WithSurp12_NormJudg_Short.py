# Based on:
#  char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos12_Long.py (loss model & code for language model)
# And autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Both_Saving.py (autoencoder)
# And (for the plain LM): ../autoencoder/autoencoder2_mlp_bidir_AND_languagemodel_sample.py
print("Character aware!")

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys
import random
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=964163553) # language model taking noised input # Amortized Prediction Posterior
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=random.choice([647336050, 516252642, 709961927, 727001672])) # Amortized Reconstruction Posterior
parser.add_argument("--load-from-plain-lm", dest="load_from_plain_lm", type=str, default=random.choice([27553360, 935649231])) # plain language model without noise (Prior)


# Unique ID for this model run
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))


# Sequence length
parser.add_argument("--sequence_length", type=int, default=random.choice([10]))

# Parameters of the neural network models
parser.add_argument("--batchSize", type=int, default=random.choice([1]))
parser.add_argument("--NUMBER_OF_REPLICATES", type=int, default=random.choice([12,20]))

## Layer size
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_lm", type=int, default=random.choice([1024]))
parser.add_argument("--hidden_dim_autoencoder", type=int, default=random.choice([512]))

## Layer number
parser.add_argument("--layer_num", type=int, default=random.choice([2]))

## Regularization
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))

## Learning Rates
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.000002, 0.00001, 0.00002, 0.00005])) #, 0.0001, 0.0002 # 1e-7, 0.000001, 0.000002, 0.000005, 0.000007, 
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.001, 0.01, 0.1, 0.2])) # 0.0001, 
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--momentum", type=float, default=random.choice([0.0, 0.3, 0.5, 0.7, 0.9]))
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.0,  0.005, 0.01, 0.1, 0.4]))



# Control
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--tuning", type=int, default=1) #random.choice([0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.001])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

# Lambda and Delta Parameters
parser.add_argument("--deletion_rate", type=float, default=0.5)
parser.add_argument("--predictability_weight", type=float, default=random.choice([0.0, 0.25, 0.5, 0.75, 1.0]))


TRAIN_LM = False
assert not TRAIN_LM



model = "REAL_REAL"

import math

args=parser.parse_args()

############################

assert args.predictability_weight >= 0
assert args.predictability_weight <= 1
assert args.deletion_rate > 0.0
assert args.deletion_rate < 0.8



#############################

assert args.tuning in [0,1]
assert args.batchSize == 1
print(args.myID)
import sys
STDOUT = sys.stdout
print(sys.argv)

print(args)
print(args, file=sys.stderr)



import corpusIteratorWikiWords



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



class PlainLanguageModel(torch.nn.Module):
  """ Prior: a sequence LSTM network """
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

#  def sample(self, numeric):
#     print("FOR SAMPLING", numeric.size())
#     embedded = self.word_embeddings(numeric.unsqueeze(0))
#     results = ["" for _ in range(args.NUMBER_OF_REPLICATES*args.batchSize)]     
#     for _ in range(10): 
#        out, self.hidden = self.rnn(embedded, self.hidden)
#        logits = self.output(out) 
#        probs = self.softmax(logits)
##        print(probs.size())
#        dist = torch.distributions.Categorical(probs=probs)
#         
#        nextWord = (dist.sample())
#        nextWordStrings = [itos_total[x] for x in nextWord.cpu().numpy()[0]]
#        for i in range(args.NUMBER_OF_REPLICATES*args.batchSize):
#            results[i] += " "+nextWordStrings[i]
#        embedded = self.word_embeddings(nextWord)
#     return results
#

  def forward(self, numeric, train=True, printHere=False, computeSurprisals=True, returnLastSurprisal=False, numberOfBatches=args.NUMBER_OF_REPLICATES*args.batchSize):
       """ Forward pass
           @param self
           @param numeric
           @param train
           @param printHere
           @param computeSurprisals
           @param returnLastSurprisal
           @param numberOfBatches

           @return lossTensor
           @return target_tensor.view(-1).size()[0]
           @return None
           @return log_probs
       """
       if self.hidden is None or True:
           self.hidden = None
           self.beginning = self.zeroBeginning
#       elif self.hidden is not None:
#           hidden1 = Variable(self.hidden[0]).detach()
#           hidden2 = Variable(self.hidden[1]).detach()
#           forRestart = bernoulli.sample()
#           hidden1 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden1)
#           hidden2 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden2)
#           self.hidden = (hidden1, hidden2)
#           self.beginning = torch.where(forRestart.unsqueeze(0) == 1, zeroBeginning, self.beginning)
       print("BEGINNING", "NUMERIC", self.beginning.size(), numeric.size())
       assert numeric.size()[1] == numberOfBatches, ("numberOfBatches", numberOfBatches)
       assert numeric.size()[0] == args.sequence_length+1
       self.beginning = numeric[numeric.size()[0]-1].view(1, numberOfBatches)
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
       if computeSurprisals: 
          logits = self.output(out) 
          log_probs = self.logsoftmax(logits)
           
          loss = self.train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))
     
          lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, numberOfBatches)
   
          if printHere:
             lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
             losses = lossTensor.data.cpu().numpy()
             numericCPU = numeric.cpu().data.numpy()
             #print(("NONE", itos_total[numericCPU[0][0]]))
             #for i in range((args.sequence_length)):
             #   print((losses[i][0], itos_total[numericCPU[i+1][0]]))
       elif returnLastSurprisal:
          logits = self.output(out[-1:]) 
          log_probs = self.logsoftmax(logits)
          loss = self.train_loss(log_probs.view(-1, len(itos)+3), target_tensor[-1].view(-1))
#          print([itos_total[int(x)] for x in target_tensor[-1].cpu()])
 #         quit()
          lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor[-1].view(-1)).view(1,numberOfBatches)
       return lossTensor, target_tensor.view(-1).size()[0], None, log_probs
   



class Autoencoder:
  """ Amortized Reconstruction Posterior """
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
 
  def sampleReconstructions(self, numeric, numeric_noised, NOUN, offset, numberOfBatches=args.batchSize*args.NUMBER_OF_REPLICATES):
      """ Draws samples from the amortized reconstruction posterior """
      if True:
          beginning = zeroBeginning


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      #target_tensor = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = self.word_embeddings(input_tensor)

      embedded_noised = self.word_embeddings(input_tensor_noised)

      out_encoder, _ = self.rnn_encoder(embedded_noised, None)



      hidden = None
      result  = ["" for _ in range(numberOfBatches)]
      result_numeric = [[] for _ in range(numberOfBatches)]
      embeddedLast = embedded[0].unsqueeze(0)
      for i in range(args.sequence_length+1):
          out_decoder, hidden = self.rnn_decoder(embeddedLast, hidden)
    
          attention = torch.bmm(self.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = self.attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)

 #         print(input_tensor.size())


          logits = self.output(self.relu(self.output_mlp(out_full) )) 
          probs = self.softmax(logits)
          if i == 15-offset:
            assert args.sequence_length == 20
            thatProbs = None #float(probs[0,:, stoi["that"]+3].mean())
#          print(i, probs[0,:, stoi["that"]+3].mean())
 #         quit()

          dist = torch.distributions.Categorical(probs=probs)
       
#          nextWord = (dist.sample())
          nextWord = torch.where(numeric_noised[i] == 0, (dist.sample()), numeric[i:i+1])
  #        print(nextWord.size())
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for i in range(numberOfBatches):
             result[i] += " "+nextWordStrings[i]
             result_numeric[i].append( nextWordDistCPU[i] )
          embeddedLast = self.word_embeddings(nextWord)
#          print(embeddedLast.size())
      for r in result[:2]:
         print(r)
      if NOUN is not None:
         nounFraction = (float(len([x for x in result if NOUN in x]))/len(result))
         thatFraction = (float(len([x for x in result if NOUN+" that" in x]))/len(result))
      else:
         nounFraction = -1
         thatFraction = -1
      result_numeric = torch.LongTensor(result_numeric).cuda()
      assert result_numeric.size()[0] == numberOfBatches
      return result, result_numeric, (nounFraction, thatFraction), thatFraction

    

autoencoder = Autoencoder()

class LanguageModel:
   """ Amortized Prediction Posterior """
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
   def forward(self, input_tensor_noised, target_tensor_full, NUMBER_OF_REPLICATES):
       lm_embedded = self.word_embeddings(input_tensor_noised)
       lm_out, lm_hidden = self.rnn_drop(lm_embedded, None)
       lm_out = lm_out[-1:]
       lm_logits = self.output(lm_out) 
       lm_log_probs = self.logsoftmax(lm_logits)
 
       # Prediction Loss 
       lm_lossTensor = self.print_loss(lm_log_probs.view(-1, len(itos)+3), target_tensor_full[-1].view(-1)).view(-1, NUMBER_OF_REPLICATES) # , args.batchSize is 1
       return lm_lossTensor 

lm = LanguageModel()

#character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total)+3, embedding_dim=args.char_emb_dim).cuda()

class MemoryModel():
  """ Noise Model """
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

parameters_memory_cached = [x for x in parameters_memory()]


# Set up optimization

dual_weight = torch.cuda.FloatTensor([1.0])
dual_weight.requires_grad=True






def parameters_autoencoder():
   for module in autoencoder.modules_autoencoder:
       for param in module.parameters():
            yield param



def parameters_lm():
   for module in lm.modules_lm:
       for param in module.parameters():
            yield param

parameters_lm_cached = [x for x in parameters_lm()]


assert not TRAIN_LM
optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9

###############################################3


# Load pretrained prior and amortized posteriors

# Amortized Reconstruction Posterior
if args.load_from_autoencoder is not None:
  print(args.load_from_autoencoder)
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["components"][i])
  
# Amortized Prediction Posterior
if args.load_from_lm is not None:
  lm_file = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure.py"
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+lm_file+"_code_"+str(args.load_from_lm)+".txt")
  for i in range(len(checkpoint["components"])):
      lm.modules_lm[i].load_state_dict(checkpoint["components"][i])

from torch.autograd import Variable



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
#         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])

       if len(numerified) > (args.batchSize*(args.sequence_length+1)):
         sequenceLengthHere = args.sequence_length+1

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

def product(x):
   r = 1
   for i in x:
     r *= i
   return r

def forward(numeric, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES, expandReplicates=True):
      """ Forward pass through the entire model
        @param numeric
      """
      global hidden
      global beginning
      global beginning_chars
      if True:
          hidden = None
          beginning = zeroBeginning


      ######################################################
      ######################################################
      # Run Loss Model
      if expandReplicates:
         numeric = numeric.expand(-1, NUMBER_OF_REPLICATES)
#      print(numeric.size(), beginning.size(), NUMBER_OF_REPLICATES)
#      numeric = torch.cat([beginning, numeric], dim=0)
      embedded_everything = lm.word_embeddings(numeric)

      # Positional embeddings
      numeric_positions = torch.LongTensor(range(args.sequence_length+1)).cuda().unsqueeze(1)
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
      assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)


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
      autoencoder_lossTensor = autoencoder.print_loss(autoencoder_log_probs.view(-1, len(itos)+3), target_tensor_onlyNoised[:-1].view(-1)).view(-1, NUMBER_OF_REPLICATES*args.batchSize)

      ##########################################
      ##########################################
      # RUN LANGUAGE MODEL (amortized prediction of next word)
      if args.predictability_weight > 0:
       lm_lossTensor = lm.forward(input_tensor_noised, target_tensor_full, NUMBER_OF_REPLICATES)
      ##########################################
      ##########################################

      # Reward, term 1
      if args.predictability_weight > 0:
        negativeRewardsTerm1 = 2*args.predictability_weight * lm_lossTensor.mean(dim=0) + 2*(1-args.predictability_weight) * autoencoder_lossTensor.mean(dim=0)
      else:
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
 #        print(("NONE", itos_total[numericCPU[0][0]]))
#         for i in range((args.sequence_length+1)):
            #print(autoencoder_losses[i][0] if i < args.sequence_length else "--", "\t", lm_losses[0][0] if args.predictability_weight > 0 and i == args.sequence_length else "---" , "\t", itos_total[numericCPU[i+1][0]],"\t", itos_total[numeric_noisedCPU[i+1][0]],"\t", memory_hidden_CPU[i+1],"\t", float(baselineValues[0]) if i == args.sequence_length else "","\t", float(numeric_embedded_cpu[i+1,0,0]),"\t", float(memory_hidden_logit_per_wordtype_cpu[i+1,0,0]),"\t", float(attention_bilinear_term[i+1,0,0]))
#            print((, itos_total[numericCPU[i+1][0]], itos_total[numeric_noisedCPU[i+1][0]], memory_hidden_CPU[i+1]))


         #if args.predictability_weight > 0:
         # print(lm_lossTensor.view(-1))
         #print(baselineValues.view(-1))
 #        if args.predictability_weight > 0:
#          print("EMPIRICAL DEVIATION FROM BASELINE", (lm_lossTensor-baselineValues).abs().mean())
               
         print("PREDICTION_LOSS", runningAveragePredictionLoss, "RECONSTRUCTION_LOSS", runningAverageReconstructionLoss, "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", expectedRetentionRate, "\tDEVIATION FROM BASELINE", runningAverageBaselineDeviation, "\tREWARD", runningAverageReward, "\tENTROPY", float(entropy))
         print(dual_weight)
      if updatesCount % 5000 == 0:
         print("updatesCount", updatesCount, updatesCount/maxUpdates)
         print("\t".join([str(x) for x in ("PREDICTION_LOSS", runningAveragePredictionLoss, "RECONSTRUCTION_LOSS", runningAverageReconstructionLoss, "\tTERM2", round(float(negativeRewardsTerm2.mean()),3), "\tAVERAGE_RETENTION", expectedRetentionRate, "\tDEVIATION FROM BASELINE", runningAverageBaselineDeviation, "\tREWARD", runningAverageReward, "\tENTROPY", float(entropy))]), file=sys.stderr)

      #runningAveragePredictionLoss = 0.95 * runningAveragePredictionLoss + (1-0.95) * float(negativeRewardsTerm1.mean())
      
      return loss, product(target_tensor_full.size())





def backward(loss, printHere):
      """ An optimization step for the resource-rational objective function """
      # Set stored gradients to zero
      optim_autoencoder.zero_grad()
      optim_memory.zero_grad()

      if dual_weight.grad is not None:
         dual_weight.grad.data.fill_(0.0)
      if printHere:
         print(loss)
      # Calculate new gradients
      loss.backward()
      # Gradient clipping
      torch.nn.utils.clip_grad_value_(parameters_memory_cached, 5.0) #, norm_type="inf")
      if TRAIN_LM:
         assert False
         torch.nn.utils.clip_grad_value_(parameters_lm_cached, 5.0) #, norm_type="inf")

      # Adapt parameters
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

maxUpdates = 200000 if args.tuning == 1 else 10000000000

def showAttention(word):
    attention = forward(torch.cuda.LongTensor([stoi[word]+3 for _ in range(args.sequence_length+1)]).view(-1, 1), train=True, printHere=True, provideAttention=True)
    attention = attention[:,0,0]
    print(*(["SCORES", word, "\t"]+[round(x,2) for x in list(attention.cpu().data.numpy())]))







nounsAndVerbs = []
nounsAndVerbs.append(["the principal",       "the teacher",        "kissed",      "was fired",                     "was quoted in the newspaper", "Was the XXXX quoted in the newspaper?", "Y"])
nounsAndVerbs.append(["the sculptor",        "the painter",        "admired",    "was n't talented",   "was completely untrue", "Was the XXXX untrue?", "Y"])
nounsAndVerbs.append(["the consultant",      "the artist",         "hired",      "was a fraud",       "shocked everyone", "Did the XXXX shock everyone?", "Y"])
nounsAndVerbs.append(["the runner",          "the psychiatrist",   "treated",    "was doping",        "was ridiculous", "Was the XXXX ridiculous?", "Y"])
nounsAndVerbs.append(["the child",           "the medic",          "rescued",    "was unharmed",      "relieved everyone", "Did the XXXX relieve everyone?", "Y"])
nounsAndVerbs.append(["the criminal",        "the officer",        "arrested",   "was guilty",        "was entirely bogus", "Was the XXXX bogus?", "Y"])
nounsAndVerbs.append(["the student",         "the professor",      "hated",      "dropped out",       "made the professor happy", "Did the XXXX make the professor happy?", "Y"])
nounsAndVerbs.append(["the mobster",         "the media",          "portrayed",  "had disappeared",    "turned out to be true", "Did the XXXX turn out to be true?", "Y"])
nounsAndVerbs.append(["the actor",           "the starlet",        "loved",      "was missing",       "made her cry", "Did the XXXX almost make her cry?", "Y"])
nounsAndVerbs.append(["the preacher",        "the parishioners",   "fired",      "stole money",        "proved to be true", "Did the XXXX prove to be true?", "Y"])
nounsAndVerbs.append(["the violinist",       "the sponsors",       "backed",     "abused drugs",                       "is likely true", "Was the XXXX likely true?", "Y"])
nounsAndVerbs.append(["the senator",         "the diplomat",       "opposed",    "was winning",                   "really made him angry", "Did the XXXX make him angry?", "Y"])
nounsAndVerbs.append(["the commander",       "the president",      "appointed",  "was corrupt",         "troubled people", "Did the XXXX trouble people?", "Y"])
nounsAndVerbs.append(["the victim",         "the criminal",       "assaulted",  "were surviving",         "calmed everyone down", "Did the XXXX calm everyone down?", "Y"])
nounsAndVerbs.append(["the politician",      "the banker",         "bribed",     "laundered money",         "came as a shock to his supporters", "Did the XXXX come as a shock?", "Y"])
nounsAndVerbs.append(["the surgeon",         "the patient",        "thanked",    "had no degree",         "was not a surprise", "Was the XXXX unsurprising?", "Y"])
nounsAndVerbs.append(["the extremist",       "the agent",          "caught",     "got an award",         "was disconcerting", "Was the XXXX disconcerting?", "Y"])
nounsAndVerbs.append(["the clerk",           "the customer",       "called",     "was a hero",         "seemed absurd", "Did the XXXX seem absurd?", "Y"])
nounsAndVerbs.append(["the trader",          "the businessman",    "consulted",  "had insider information",         "was confirmed", "Was the XXXX confirmed?", "Y"])
nounsAndVerbs.append(["the CEO",             "the employee",       "impressed",  "was retiring",         "was entirely correct", "Was the XXXX correct?", "Y"])



#nounsAndVerbs.append(["the senator",        "the diplomat",       "opposed"])

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

topNouns.append("assessment")
topNouns.append("truth")
topNouns.append("declaration")
topNouns.append("complaint")
topNouns.append("admission")
topNouns.append("disclosure")
topNouns.append("confirmation")
topNouns.append("guess")
topNouns.append("remark")
topNouns.append("news")
topNouns.append("proof")
topNouns.append("message")
topNouns.append("announcement")
topNouns.append("statement")
topNouns.append("thought")
topNouns.append("allegation")
topNouns.append("indication")
topNouns.append("recognition")
topNouns.append("speculation")
topNouns.append("accusation")
topNouns.append("reminder")
topNouns.append("rumor")
topNouns.append("finding")
topNouns.append("idea")
topNouns.append("feeling")
topNouns.append("conjecture")
topNouns.append("perception")
topNouns.append("certainty")
topNouns.append("revelation")
topNouns.append("understanding")
topNouns.append("claim")
topNouns.append("view")
topNouns.append("observation")
topNouns.append("conviction")
topNouns.append("presumption")
topNouns.append("intuition")
topNouns.append("opinion")
topNouns.append("conclusion")
topNouns.append("notion")
topNouns.append("suggestion")
topNouns.append("sense")
topNouns.append("suspicion")
topNouns.append("assurance")
topNouns.append("insinuation")
topNouns.append("realization")
topNouns.append("assertion")
topNouns.append("impression")
topNouns.append("contention")
topNouns.append("assumption")
topNouns.append("belief")
topNouns.append("fact")

topNouns = list(set(topNouns))



with open("../../../../forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", "r") as inFile:
   counts = [x.replace('"', '').split("\t") for x in inFile.read().strip().split("\n")]
   header = ["LineNum"] + counts[0]
   assert len(header) == len(counts[1])
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[header["Noun"]] : line for line in counts[1:]}


print(len(topNouns))
print([x for x in topNouns if x not in counts])
#topNouns = [x for x in topNouns if x in counts]

def thatBias(noun):
    return 1.0
#   return math.log(float(counts[noun][header["CountThat"]]))-math.log(float(counts[noun][header["CountBare"]]))

#topNouns = sorted(list(set(topNouns)), key=lambda x:thatBias(x))

print(topNouns)
print(len(topNouns))
#quit()


    
    
plain_lm = PlainLanguageModel()
plain_lmFileName = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars.py"

if args.load_from_plain_lm is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+plain_lmFileName+"_code_"+str(args.load_from_plain_lm)+".txt")
  for i in range(len(checkpoint["components"])):
      plain_lm.modules[i].load_state_dict(checkpoint["components"][i])


# Helper Functions

def correlation(x, y):
   variance_x = (x.pow(2)).mean() - x.mean().pow(2)
   variance_y = (y.pow(2)).mean() - y.mean().pow(2)
   return ((x-x.mean())* (y-y.mean())).mean()/(variance_x*variance_y).sqrt()


def rindex(x, y):
   return max([i for i in range(len(x)) if x[i] == y])


def encodeContextCrop(inp, context):
     sentence = context.strip() + " " + inp.strip()
     print("ENCODING", sentence)
     numerified = [stoi_total[char] if char in stoi_total else 2 for char in sentence.split(" ")]
     print(len(numerified))
     numerified = numerified[-args.sequence_length-1:]
     numerified = torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
     return numerified

def flatten(x):
   l = []
   for y in x:
     for z in y:
        l.append(z)
   return l


calibrationSentences = []
calibrationSentences.append("she is a good runner but a bad swimmer")
calibrationSentences.append("she are a good runners but a bad swimmers")
calibrationSentences.append("the children ate a bowl of international trade for breakfast")
calibrationSentences.append("the children ate a bowl of tasty cereals for breakfast")
calibrationSentences.append("they know how to solve the problem")
calibrationSentences.append("they knows how to solve the problem")

# word salad
calibrationSentences.append("his say later with to use electrons")
calibrationSentences.append("and a enlarged in transformed into")
calibrationSentences.append("available a solvated in the cosmetics")
calibrationSentences.append("it alkali a basis used metal been")
calibrationSentences.append("has result were are pear masters")
calibrationSentences.append("for are include course additional barrow")
calibrationSentences.append("to school a and apple flavors cremations")

# acceptable sentences, mostly adapted from COLA, some words exchanged to ensure InVocab
calibrationSentences.append("she went there yesterday and saw a movie")
calibrationSentences.append("perseus saw the gorgon in his shield")
calibrationSentences.append("what did you say ( that ) the poet had written ?")
calibrationSentences.append("i saw people playing there on the beach")
calibrationSentences.append("i did n't want any cake")
calibrationSentences.append("that i should kiss pigs is my fondest dream")
calibrationSentences.append("ron failed biology , unfortunately")
calibrationSentences.append("the men chuckle")
calibrationSentences.append("i expected there to be a problem")
calibrationSentences.append("gilgamesh wanted to seduce ishtar , and seduce ishtar he did")
calibrationSentences.append("harry collapsed")
calibrationSentences.append("i asked who saw what")
calibrationSentences.append("he has been happy")
calibrationSentences.append("poseidon had run away , before the executioner murdered hera")
calibrationSentences.append("merlin is a dangerous sorcerer")
calibrationSentences.append("anson saw anson")
calibrationSentences.append("i am to eat macaroni")
calibrationSentences.append("poseidon had escaped , before the executioner arrived")
calibrationSentences.append("owners love truffles")
calibrationSentences.append("humans love to eat owners of pigs")
calibrationSentences.append("those pigs love truffles")
calibrationSentences.append("i 'd planned to have finished by now")
calibrationSentences.append("has the potion not worked ?")
calibrationSentences.append("what i love is toast and sun dried tomatoes")
calibrationSentences.append("mary ran")
calibrationSentences.append("he replied that he was happy")
calibrationSentences.append("no one could remove the blood from the wall")
calibrationSentences.append("julie maintained that the barman was sober")
calibrationSentences.append("benjamin gave lee the cloak")
calibrationSentences.append("aphrodite wanted hera to persuade athena to leave")
calibrationSentences.append("gilgamesh is fighting the dragon")
calibrationSentences.append("i claimed she was pregnant")
calibrationSentences.append("for jenny , i intended to be present")
calibrationSentences.append("gilgamesh missed aphrodite")
calibrationSentences.append("she might be pregnant")
calibrationSentences.append("anson demonized david at the club")
calibrationSentences.append("jason asked whether the potion was ready")
calibrationSentences.append("frieda closed the door")
calibrationSentences.append("medea might have given jason a poisoned robe ( just treat a poisoned robe as an np")
calibrationSentences.append("quickly kiss anson !")
calibrationSentences.append("julie felt hot")
calibrationSentences.append("agamemnon expected esther to seem to be happy")
calibrationSentences.append("that the answer is obvious upset hermes")
calibrationSentences.append("homer recited the poem about achilles ?")
calibrationSentences.append("no vampire can survive sunrise")
calibrationSentences.append("under the bed is the best place to hide")
calibrationSentences.append("anson appeared")
calibrationSentences.append("there seems to be a problem")
calibrationSentences.append("i intoned that she was happy")
calibrationSentences.append("medea saw who ?")
calibrationSentences.append("no one expected that agamemnon would win")
calibrationSentences.append("believing that the world is flat gives one some solace")
calibrationSentences.append("kick them !")
calibrationSentences.append("medea wondered if the potion was ready")
calibrationSentences.append("who all did you meet when you were in derry ?")
calibrationSentences.append("who did you hear an oration about ?")
calibrationSentences.append("alison ran")
calibrationSentences.append("romeo sent letters to juliet")
calibrationSentences.append("richard 's gift of the helicopter to the hospital and of the bus to the school")
calibrationSentences.append("nathan caused benjamin to see himself in the mirror")
calibrationSentences.append("a. madeleine planned to catch the sardines and she did")
calibrationSentences.append("i did not understand")
calibrationSentences.append("gilgamesh loved ishtar and aphrodite did too")
calibrationSentences.append("we believed him to be omnipotent")
calibrationSentences.append("david ate mangoes and raffi should too")
calibrationSentences.append("julie and fraser ate those delicious pies in julie 's back garden")
calibrationSentences.append("the old pigs love truffles")
calibrationSentences.append("the boys all should could go")
calibrationSentences.append("aphrodite quickly freed the animals")
calibrationSentences.append("paul had two affairs")
calibrationSentences.append("what alison and david did was soak their feet in a bucket")
calibrationSentences.append("anson demonized david almost constantly")
calibrationSentences.append("anson 's hen nibbled his ear")
calibrationSentences.append("before the executioner arrived , poseidon had escaped")
calibrationSentences.append("gilgamesh did n't leave")
calibrationSentences.append("genie intoned that she was tired")
calibrationSentences.append("look at all these books which book would you like ?")
calibrationSentences.append("i do n't remember what i said all ?")
calibrationSentences.append("the pig grunts")
calibrationSentences.append("people are stupid")
calibrationSentences.append("what i arranged was for jenny to be present")
calibrationSentences.append("i compared ginger to fred")
calibrationSentences.append("which poet wrote which ode ?")
calibrationSentences.append("how did julie ask if jenny left ?")
calibrationSentences.append("dracula thought him to be the prince of darkness")
calibrationSentences.append("i must eat macaroni")
calibrationSentences.append("i asked who john would introduce to who")
calibrationSentences.append("reading shakespeare satisfied me")
calibrationSentences.append("humans love to eat owners")
calibrationSentences.append("gilgamesh fears death and achilles does as well")
calibrationSentences.append("how did julie say that jenny left ?")
calibrationSentences.append("show me letters !")
calibrationSentences.append("the readings of shakespeare satisfied me")
calibrationSentences.append("anson demonized david every day")
calibrationSentences.append("the students demonstrated this morning")
calibrationSentences.append("we believed aphrodite to be omnipotent")
calibrationSentences.append("emily caused benjamin to see himself in the mirror")
calibrationSentences.append("nothing like that would i ever eat again")
calibrationSentences.append("where has he put the cake ?")
calibrationSentences.append("jason persuaded medea to desert her family")
calibrationSentences.append("gilgamesh perhaps should be leaving")
calibrationSentences.append("gilgamesh has n't kissed ishtar")
calibrationSentences.append("it is easy to slay the gorgon")
calibrationSentences.append("i had the strangest feeling that i knew you")
calibrationSentences.append("what all did you get for christmas ?")

# unacceptable sentences from COLA
calibrationSentences.append("they drank the pub")
calibrationSentences.append("the professor talked us")
calibrationSentences.append("we yelled ourselves")
calibrationSentences.append("we yelled harry hoarse")
calibrationSentences.append("harry coughed himself")
calibrationSentences.append("harry coughed us into a fit")
calibrationSentences.append("they caused him to become angry by making him")
calibrationSentences.append("they caused him to become president by making him")
calibrationSentences.append("they made him to exhaustion")
calibrationSentences.append("the car honked down the road")
calibrationSentences.append("the dog barked out of the room")
calibrationSentences.append("the witch went into the forest by vanishing")
calibrationSentences.append("the building is tall and tall")
calibrationSentences.append("this building is taller and taller")
calibrationSentences.append("this building got than that one")
calibrationSentences.append("this building is than that one")
calibrationSentences.append("bill floated into the cave for hours")
calibrationSentences.append("bill pushed harry off the sofa for hours")
calibrationSentences.append("bill cried sue to sleep")
calibrationSentences.append("the elevator rumbled itself to the ground")
calibrationSentences.append("she yelled hoarse")
calibrationSentences.append("ted cried to sleep")
calibrationSentences.append("the ball wriggled itself loose")
calibrationSentences.append("the most you want , the least you eat")
calibrationSentences.append("i demand that the more john eat , the more he pay")
calibrationSentences.append("i demand that john pays more , the more he eat")
calibrationSentences.append("you get angrier , the more we eat , do n't we")
calibrationSentences.append("the harder it has rained , how much faster a flow that appears in the river ?")
calibrationSentences.append("the harder it rains , how much faster that do you run ?")

#in and is the it
#time maintain metal commercially the was salt cut as two-year college flavorings solutions
#a alkali and and four-year into twice programme cation and \tduring addition ammonia burials
#often and used butyllithiums perfumery as this amide



def getTotalSentenceSurprisalsCalibration(SANITY="Sanity", VERBS=2): # Surprisal for EOS after 2 or 3 verbs
    assert SANITY in ["Sanity", "Model"]
    assert VERBS in [1,2]
    print(plain_lm) 
    surprisalsPerNoun = {}
    thatFractionsPerNoun = {}
    numberOfSamples = 12
    with torch.no_grad():
     with open("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w") as outFile:
      print("Sentence", "Region", "Word", "Surprisal", file=outFile)

      for sentence in calibrationSentences:
            print(sentence)
            context = "later , the nurse suggested they treat the patient with an antibiotic, but in the end , this did not happen . "
            remainingInput = sentence.split(" ")
            for i in range(len(remainingInput)):
              numerified = encodeContextCrop(" ".join(remainingInput[:i+1]), context)
              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
              # Run the noise model
              numeric, numeric_noised = forward(numerified, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=numberOfSamples)
              numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
              numeric = numeric.unsqueeze(2).expand(-1, -1, 24).view(-1, numberOfSamples*24)
              numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
              # Get samples from the reconstruction posterior
              result, resultNumeric, fractions, thatProbs = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*24)
 #             print(resultNumeric.size())
              resultNumeric = resultNumeric.transpose(0,1).contiguous()
              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*24)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()
              # get next-word surprisals using the prior
              totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False, computeSurprisals=False, returnLastSurprisal=True, numberOfBatches=numberOfSamples*24)
#              print(totalSurprisal.size())
              totalSurprisal = totalSurprisal.view(numberOfSamples, 24)
              surprisalOfNextWord = totalSurprisal.exp().mean(dim=1).log().mean()
              print("SURPRISAL", sentence, i, remainingInput[i],float( surprisalOfNextWord))
              print(sentence, i, remainingInput[i], float( surprisalOfNextWord), file=outFile)

#    with open("/u/scr/mhahn/reinforce-logs-both/full-logs-tsv/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w") as outFile:
#      print("Noun", "Region", "Condition", "Surprisal", "ThatFraction", file=outFile)
#      for noun in topNouns:
#       for region in ["V3", "V2", "V1", "EOS"]:
#         for condition in ["u", "g"]:
#           print(noun, region, condition, surprisalsPerNoun[noun][condition][region], thatFractionsPerNoun[noun][condition][region], file=outFile)
#    for region in ["V3", "V2", "V1", "EOS"]:
#       print(SANITY, "CORR", region, correlation(torch.FloatTensor([(float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]])) for x in topNouns]), torch.FloatTensor([surprisalsPerNoun[x]["g"][region]-surprisalsPerNoun[x]["u"][region] for x in topNouns])), correlation(torch.FloatTensor([(float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]])) for x in topNouns]), torch.FloatTensor([thatFractionsPerNoun[x]["g"][region]-thatFractionsPerNoun[x]["u"][region] for x in topNouns])))
#    overallSurprisalForCompletion = torch.FloatTensor([sum([surprisalsPerNoun[noun]["u"][region] - surprisalsPerNoun[noun]["g"][region] for region in ["V2", "V1", "EOS"]]) for noun in topNouns])
#    print(SANITY, "CORR total", correlation(torch.FloatTensor([(float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]])) for x in topNouns]), overallSurprisalForCompletion), "note this is inverted!")




def getTotalSentenceSurprisals(SANITY="Model", VERBS=2): # Surprisal for EOS after 2 or 3 verbs
    assert SANITY in ["Model"]
    assert VERBS in [1,2]
    print(plain_lm) 
    surprisalsPerNoun = {}
    thatFractionsPerNoun = {}
    numberOfSamples = 12
    with torch.no_grad():
      for NOUN in topNouns:
         print(NOUN, "Time:", time.time() - startTimePredictions, file=sys.stderr)
         for sentenceList in nounsAndVerbs:
           print(sentenceList)
           context = "later , the nurse suggested they treat the patient with an antibiotic, but in the end , this did not happen . " + f"the {NOUN}"
           thatFractions = {x : defaultdict(float) for x in ["SC", "NoSC"]}
           surprisalByRegions = {x : defaultdict(float) for x in ["SC", "NoSC"]}

           for condition in ["SC","NoSC"]:
            if condition == "SC":
               context = "later , the nurse suggested they treat the patient with an antibiotic, but in the end , this did not happen . " + f"the {NOUN} that {sentenceList[0]} {sentenceList[3]}"
               remainingInput = f"{sentenceList[4]} .".split(" ")[:1]
               regions = [f"V1_{c}" for c, _ in enumerate(remainingInput)]
               assert len(remainingInput) == len(regions)
            else:
               context = "later , the nurse suggested they treat the patient with an antibiotic, but in the end , this did not happen . " + f"the {NOUN}"
               remainingInput = f"{sentenceList[4]} .".split(" ")[:1]
               regions = [f"V1_{c}" for c, _ in enumerate(remainingInput)]
               assert len(remainingInput) == len(regions)
            print("INPUT", context, remainingInput)
            for i in range(len(remainingInput)):
              numerified = encodeContextCrop(" ".join(remainingInput[:i+1]), context)
              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
     #         print(i, " ########### ", SANITY, VERBS)
    #          print(numerified.size())
              # Run the memory model. We collect 'numberOfSamples' many replicates.
              if SANITY == "Sanity":
                 numeric = numerified
                 numeric = numeric.expand(-1, numberOfSamples)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
              else:
                 numeric, numeric_noised = forward(numerified, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=numberOfSamples)
                 numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
              # Next, expand the tensor to get 24 samples from the reconstruction posterior for each replicate
              numeric = numeric.unsqueeze(2).expand(-1, -1, 24).view(-1, numberOfSamples*24)
              numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
              # Now get samples from the amortized reconstruction posterior
              result, resultNumeric, fractions, thatProbs = autoencoder.sampleReconstructions(numeric, numeric_noised, NOUN, 2, numberOfBatches=numberOfSamples*24)
              if condition == "SC" and i == 0:
                 locationThat = context.split(" ")[::-1].index("that")
                 thatFractions[condition][regions[i]]=float((resultNumeric[-locationThat-2] == stoi_total["that"]).float().mean())
              resultNumeric = resultNumeric.transpose(0,1).contiguous()
              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*24)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()
              # Evaluate the prior on these samples to estimate next-word surprisal
              totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False, computeSurprisals=False, returnLastSurprisal=True, numberOfBatches=numberOfSamples*24)
#              print(totalSurprisal.size())
              # For each of the `numberOfSamples' many replicates, evaluate (i) the probability of the next word under the Monte Carlo estimate of the next-word posterior, (ii) the corresponding surprisal, (iii) the average of those surprisals across the 'numberOfSamples' many replicates.
              totalSurprisal = totalSurprisal.view(numberOfSamples, 24)
              surprisalOfNextWord = totalSurprisal.exp().mean(dim=1).log().mean()
              print("PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in numerified[:,0]]), surprisalOfNextWord)
              print("DENOISED PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in resultNumeric[:,0]]), surprisalOfNextWord)
              print("SURPRISAL", NOUN, sentenceList[0], condition, i, regions[i], remainingInput[i],float( surprisalOfNextWord))
              surprisalByRegions[condition][regions[i]] += float( surprisalOfNextWord)
#              if i == 0 or regions[i] != regions[i-1]:
 #                 thatFractions[condition][regions[i]]=thatProbs
         print(surprisalByRegions)
         print(thatFractions)
         print("NOUNS SO FAR", topNouns.index(NOUN))
         surprisalsPerNoun[NOUN] = surprisalByRegions
         thatFractionsPerNoun[NOUN] = thatFractions
    print("SURPRISALS BY NOUN", surprisalsPerNoun)
    print("THAT (fixed) BY NOUN", thatFractionsPerNoun)
    print("SURPRISALS_PER_NOUN PLAIN_LM, WITH VERB, NEW")
    with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w") as outFile:
      print("Noun", "Region", "Condition", "Surprisal", "ThatFraction", file=outFile)
      for noun in topNouns:
       for condition in ["SC", "NoSC"]:
         for region in surprisalsPerNoun[noun][condition]:
           print(noun, region, condition, surprisalsPerNoun[noun][condition][region], thatFractionsPerNoun[noun][condition][region], file=outFile)
    # For sanity-checking: Prints correlations between surprisal and that-bias
 #   for region in ["V3_0", "V2_0", "V1_0", "EOS_0"]:
#       print(SANITY, "CORR", region, correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([surprisalsPerNoun[x]["g"][region]-surprisalsPerNoun[x]["u"][region] for x in topNouns])), correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([thatFractionsPerNoun[x]["g"][region]-thatFractionsPerNoun[x]["u"][region] for x in topNouns])))
    overallSurprisalForCompletion = torch.FloatTensor([sum([surprisalsPerNoun[noun]["SC"][region] - surprisalsPerNoun[noun]["NoSC"][region] for region in surprisalsPerNoun[noun]["SC"]]) for noun in topNouns])
    print(SANITY, "CORR total", correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), overallSurprisalForCompletion), "note this is inverted!")


#getTotalSentenceSurprisalsCalibration(SANITY="Model")
#quit()


startTimePredictions = time.time()

#getTotalSentenceSurprisals()
#quit()

#incrementallySampleCompletions(SANITY="Model", VERBS=1)
#getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Model", VERBS=1)
#quit()

#getTotalSentenceSurprisals(SANITY="Sanity")
#quit()
#getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Model")
#getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Sanity")
#quit()
#  
#getPerNounReconstructionsSanity()
#getPerNounReconstructionsSanityVerb()
startTimeTotal = time.time()

for epoch in range(1000):
   print(epoch)

   # Get training data
   training_data = corpusIteratorWikiWords.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)


   # Set the model up for training
   lm.rnn_drop.train(True)
   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, None
   # End optimization when maxUpdates is reached
   if updatesCount > maxUpdates:
     break
   while updatesCount <= maxUpdates:
      counter += 1
      updatesCount += 1
      # Get model predictions at the end of optimization
      if updatesCount == maxUpdates:

       # Record calibration for the acceptability judgments
       getTotalSentenceSurprisalsCalibration(SANITY="Model")
       
       # Record reconstructions and surprisals
       with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs/"+__file__+"_"+str(args.myID), "w") as outFile:
         startTimePredictions = time.time()

         sys.stdout = outFile
         print(updatesCount)
         print(args)
         getTotalSentenceSurprisals(SANITY="Model")
  #       getTotalSentenceSurprisals(SANITY="Sanity")

#         getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Model", VERBS=1)
 #        getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Sanity", VERBS=2)
 #        getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Model", VERBS=2)
#         getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Sanity", VERBS=2)
#  

#         getPerNounReconstructionsSanity()
#         getPerNounReconstructionsSanityVerb()
#         getPerNounReconstructions()
#         getPerNounReconstructionsVerb()
#         getPerNounReconstructions2Verbs()
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
         sys.stdout = STDOUT

#      if updatesCount % 10000 == 0:
#         optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
#         optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9
#
      # Get a batch from the training set
      try:
         numeric, _ = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      # Run this through the model: forward pass of the resource-rational objective function
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      # Calculate gradients and update parameters
      backward(loss, printHere)

#      if loss.data.cpu().numpy() > 15.0:
#          lossHasBeenBad += 1
#      else:
#          lossHasBeenBad = 0

      # Bad learning rate parameters might make the loss explode. In this case, stop.
      if lossHasBeenBad > 100:
          print("Loss exploding, has been bad for a while")
          print(loss)
          assert False
      trainChars += charCounts 
      if printHere:
          print(("Loss here", loss))
          print((epoch, "Updates", updatesCount, str((100.0*updatesCount)/maxUpdates)+" %", maxUpdates, counter, trainChars, "ETA", ((time.time()-startTimeTotal)/updatesCount * (maxUpdates-updatesCount))/3600.0, "hours"))
          print("Dev losses")
          print(devLosses)
          print("Words per sec "+str(trainChars/(time.time()-startTime)))
          print(args.learning_rate_memory, args.learning_rate_autoencoder)
          print(lastSaved)
          print(__file__)
          print(args)

      if False and (time.time() - totalStartTime)/60 > 4000:
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


with open("/u/scr/mhahn/reinforce-logs-both-short/results/"+__file__+"_"+str(args.myID), "w") as outFile:
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


