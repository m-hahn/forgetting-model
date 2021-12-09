# Derived from char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_Lo.py

# Example: ~/python-py37-mhahn char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_TPLo.py --load_from_joint=163008048 --stimulus_file=Staub2006
# ~/python-py37-mhahn char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_Lo.py --load_from_joint=163008048 --stimulus_file=Staub2006 --criticalRegions=NP1_0,NP1_1,OR,NP2_0,NP2_1

# --stimulus_file=BartekEtal --criticalRegions=Critical_0

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
#parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=964163553) # language model taking noised input # Amortized Prediction Posterior
#parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=random.choice([647336050, 516252642, 709961927, 727001672, 712478284, 524811876])) # Amortized Reconstruction Posterior
#parser.add_argument("--load-from-plain-lm", dest="load_from_plain_lm", type=str, default=random.choice([27553360, 935649231])) # plain language model without noise (Prior)
parser.add_argument("--load_from_joint", type=str)


# Unique ID for this model run
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))


# Sequence length
parser.add_argument("--sequence_length", type=int, default=random.choice([20]))

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
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.00002, 0.00005, 0.0001, 0.0001, 0.0001]))  # Can also use 0.0001, which leads to total convergence to deterministic solution withtin maximum iterations (March 25, 2021)   #, 0.0001, 0.0002 # 1e-7, 0.000001, 0.000002, 0.000005, 0.000007, 
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.001, 0.01, 0.1, 0.1, 0.1, 0.1])) # 0.0001, 
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--momentum", type=float, default=random.choice([0.5, 0.7, 0.7, 0.7, 0.7, 0.9])) # Momentum is helpful in facilitating convergence to a low-loss solution (March 25, 2021). It might be even more important for getting fast convergence than a high learning rate
parser.add_argument("--entropy_weight", type=float, default=random.choice([0.0])) # 0.0,  0.005, 0.01, 0.1, 0.4]))



# Control
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--tuning", type=int, default=1) #random.choice([0.00001, 0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.0008, 0.001])) # 0.0,  0.005, 0.01, 0.1, 0.4]))

# Lambda and Delta Parameters
parser.add_argument("--deletion_rate", type=float, default=0.5)
parser.add_argument("--predictability_weight", type=float, default=random.choice([0.0, 0.25, 0.5, 0.75, 1.0]))

parser.add_argument("--stimulus_file", type=str)
parser.add_argument("--criticalRegions", type=str)


TRAIN_LM = False
assert not TRAIN_LM



model = "REAL_REAL"

import math

args=parser.parse_args()

if args.criticalRegions is not None:
   args.criticalRegions = args.criticalRegions.split(",")
############################

assert args.predictability_weight >= 0
assert args.predictability_weight <= 1
assert args.deletion_rate > 0.0
assert args.deletion_rate < 1.0



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


  def forward(self, input_tensor_pure, input_tensor_noised, target_tensor_onlyNoised, NUMBER_OF_REPLICATES):
      # INPUTS: input_tensor_pure, input_tensor_noised
      # OUTPUT: autoencoder_lossTensor
#      autoencoder_lossTensor =  autoencoder.forward(input_tensor_pure, input_tensor_noised, NUMBER_OF_REPLICATES)

      autoencoder_embedded = self.word_embeddings(input_tensor_pure[:-1])
      autoencoder_embedded_noised = self.word_embeddings(input_tensor_noised[:-1])
      autoencoder_out_encoder, _ = self.rnn_encoder(autoencoder_embedded_noised, None)
      autoencoder_out_decoder, _ = self.rnn_decoder(autoencoder_embedded, None)
      assert autoencoder_embedded.size()[0] == args.sequence_length-1, (autoencoder_embedded.size()[0], args.sequence_length-1) # Note that this is different from autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py. Would be good if they were unified.
      assert autoencoder_embedded_noised.size()[0] == args.sequence_length-1, (autoencoder_embedded.size()[0], args.sequence_length-1) # Note that this is different from autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py.

      autoencoder_attention = torch.bmm(self.attention_proj(autoencoder_out_encoder).transpose(0,1), autoencoder_out_decoder.transpose(0,1).transpose(1,2))
      autoencoder_attention = self.attention_softmax(autoencoder_attention).transpose(0,1)
      autoencoder_from_encoder = (autoencoder_out_encoder.unsqueeze(2) * autoencoder_attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      autoencoder_out_full = torch.cat([autoencoder_out_decoder, autoencoder_from_encoder], dim=2)


      autoencoder_logits = self.output(self.relu(self.output_mlp(autoencoder_out_full) ))
      autoencoder_log_probs = self.logsoftmax(autoencoder_logits)

      # Prediction Loss 
      autoencoder_lossTensor = self.print_loss(autoencoder_log_probs.view(-1, len(itos)+3), target_tensor_onlyNoised[:-1].view(-1)).view(-1, NUMBER_OF_REPLICATES*args.batchSize)
      return autoencoder_lossTensor
 

  def sampleReconstructions(self, numeric, numeric_noised, NOUN, offset, numberOfBatches=args.batchSize*args.NUMBER_OF_REPLICATES, fillInBefore=-1, computeProbabilityStartingFrom=0):
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
      amortizedPosterior = torch.zeros(numberOfBatches, device='cuda')
      zeroLogProb = torch.zeros(numberOfBatches, device='cuda')
      for i in range(args.sequence_length+1):
          out_decoder, hidden = self.rnn_decoder(embeddedLast, hidden)
#          assert embeddedLast.size()[0] == args.sequence_length-1, (embeddedLast.size()[0] , args.sequence_length)


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
          if i < fillInBefore:
             nextWord = numeric[i:i+1]
          else:
            sampledFromDist = dist.sample()
            logProbForSampledFromDist = dist.log_prob(sampledFromDist).squeeze(0)
 #           print(logProbForSampledFromDist.size(), numeric_noised[i].size(), zeroLogProb.size())
            assert numeric_noised.size()[0] == args.sequence_length+1
            if i < args.sequence_length: # IMPORTANT make sure the last word -- which is (due to a weird design choice) cut off -- doesn't contribute to the posterior
               amortizedPosterior += torch.where(numeric_noised[i] == 0, logProbForSampledFromDist, zeroLogProb)

            nextWord = torch.where(numeric_noised[i] == 0, sampledFromDist, numeric[i:i+1])
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
      return result, result_numeric, (nounFraction, thatFraction), thatFraction, amortizedPosterior

    

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
     # Modules of the memory model
     self.modules_memory = [self.memory_mlp_inner, self.memory_mlp_outer, self.memory_mlp_inner_from_pos, self.positional_embeddings, self.perword_baseline_inner, self.perword_baseline_outer, self.memory_word_pos_inter, self.memory_bilinear, self.memory_mlp_inner_bilinear]
  def forward(self, numeric):
      embedded_everything_mem = lm.word_embeddings(numeric).detach()

      # Positional embeddings
      numeric_positions = torch.LongTensor(range(args.sequence_length+1)).cuda().unsqueeze(1)
      embedded_positions = self.positional_embeddings(numeric_positions)
      numeric_embedded = self.memory_word_pos_inter(embedded_positions)

      # Retention probabilities
      memory_byword_inner = self.memory_mlp_inner(embedded_everything_mem)
      memory_hidden_logit_per_wordtype = self.memory_mlp_outer(self.relu(memory_byword_inner))

  #    print(embedded_positions.size(), embedded_everything.size())
 #     print(self.memory_bilinear(embedded_positions).size())
#      print(self.relu(self.memory_mlp_inner_bilinear(embedded_everything.detach())).transpose(1,2).size())
      attention_bilinear_term = torch.bmm(self.memory_bilinear(embedded_positions), self.relu(self.memory_mlp_inner_bilinear(embedded_everything_mem)).transpose(1,2)).transpose(1,2)

      memory_hidden_logit = numeric_embedded + memory_hidden_logit_per_wordtype + attention_bilinear_term
      memory_hidden = self.sigmoid(memory_hidden_logit)
      return memory_hidden, embedded_everything_mem



  def compute_likelihood(self, numeric, numeric_noised, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES, expandReplicates=True, computeProbabilityStartingFrom=0):
      """ Forward pass through the entire model
        @param numeric
      """
      global hidden
      global beginning
      global beginning_chars
      if True:
          hidden = None
          beginning = zeroBeginning

      assert numeric.size() == numeric_noised.size(), (numeric.size(), numeric_noised.size())

      ######################################################
      ######################################################
      # Run Loss Model
      if expandReplicates:
         assert False
         numeric = numeric.expand(-1, NUMBER_OF_REPLICATES)
#      print(numeric.size(), beginning.size(), NUMBER_OF_REPLICATES)
#      numeric = torch.cat([beginning, numeric], dim=0)
      embedded_everything_mem = lm.word_embeddings(numeric)

      # Positional embeddings
      numeric_positions = torch.LongTensor(range(args.sequence_length+1)).cuda().unsqueeze(1)
      embedded_positions = self.positional_embeddings(numeric_positions)
      numeric_embedded = self.memory_word_pos_inter(embedded_positions)

      # Retention probabilities
      memory_byword_inner = self.memory_mlp_inner(embedded_everything_mem)
      memory_hidden_logit_per_wordtype = self.memory_mlp_outer(self.relu(memory_byword_inner))

  #    print(embedded_positions.size(), embedded_everything.size())
 #     print(self.memory_bilinear(embedded_positions).size())
#      print(self.relu(self.memory_mlp_inner_bilinear(embedded_everything.detach())).transpose(1,2).size())
      attention_bilinear_term = torch.bmm(self.memory_bilinear(embedded_positions), self.relu(self.memory_mlp_inner_bilinear(embedded_everything_mem)).transpose(1,2)).transpose(1,2)

      memory_hidden_logit = numeric_embedded + memory_hidden_logit_per_wordtype + attention_bilinear_term
      memory_hidden = self.sigmoid(memory_hidden_logit)
 #     if provideAttention:
#         return memory_hidden

#      # Baseline predictions for prediction loss
 #     baselineValues = 10*self.sigmoid(self.perword_baseline_outer(self.relu(self.perword_baseline_inner(embedded_everything[-1].detach())))).squeeze(1)
  #    assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)


      # NOISE MEMORY ACCORDING TO MODEL
      memory_filter = (numeric_noised != 0)
#      print(memory_filter.size(), memory_hidden.size())
      bernoulli_logprob = torch.where(memory_filter, torch.log(memory_hidden.squeeze(2)+1e-10), torch.log(1-memory_hidden.squeeze(2)+1e-10))

      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(12, 1, 1)).long().sum(dim=0)).bool())

      # Disregard likelihood computation on punctuation
      bernoulli_logprob = torch.where(punctuation, 0*bernoulli_logprob, bernoulli_logprob)
      # Penalize forgotten punctuation
      bernoulli_logprob = torch.where(torch.logical_and(punctuation, memory_filter==0), 0*bernoulli_logprob-10.0, bernoulli_logprob)

#      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)

     # Run the following lines as a sanity check
#      print(numeric.size(), numeric_noised.size())
#      for i in range(computeProbabilityStartingFrom, bernoulli_logprob.size()[0]):
#        print(i, itos_total[int(numeric[i,0])], itos_total[int(numeric_noised[i,0])], bernoulli_logprob[i,0])


      # SPECIFICALLY FOR THIS APPLICATION (where the last element in the sequence is the first future word) CUT OFF, TO REDUCE EXTRANEOUS VARIANCE, OR POTENTIALLY PRECLUDE WEIRRD VALUES AS THAT IS ALWAYS OBLIGATORILY NOISED: I'm cutting of the final value by restricting up to -1.
      return bernoulli_logprob[computeProbabilityStartingFrom:-1].sum(dim=0)






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


#assert not TRAIN_LM
#optim_autoencoder = torch.optim.SGD(parameters_autoencoder(), lr=args.learning_rate_autoencoder, momentum=0.0) # 0.02, 0.9
#optim_memory = torch.optim.SGD(parameters_memory(), lr=args.learning_rate_memory, momentum=args.momentum) # 0.02, 0.9

###############################################3

checkpoint = torch.load(glob.glob("/u/scr/mhahn/CODEBOOKS_MEMORY/*"+str(args.load_from_joint)+"*")[0])
# Load pretrained prior and amortized posteriors

print("ARGUMENTS OF ORIGINAL MODEL ", checkpoint["arguments"])
# Amortized Reconstruction Posterior
if True or args.load_from_autoencoder is not None:
#  if checkpoint["arguments"].deletion_rate not in [0.05,0.5] or checkpoint["arguments"].predictability_weight != 0.5:
#      assert False, "rejecting models outside of relevant parameter space"

  print(checkpoint["arguments"].load_from_autoencoder)
  checkpoint_ = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"+"_code_"+str(checkpoint["arguments"].load_from_autoencoder)+".txt")
  for i in range(len(checkpoint_["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint_["components"][i])
  del checkpoint_
 
# Amortized Prediction Posterior
if True or args.load_from_lm is not None:
  lm_file = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure.py"
  checkpoint_ = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+lm_file+"_code_"+str(checkpoint["arguments"].load_from_lm)+".txt")
  for i in range(len(checkpoint_["components"])):
      lm.modules_lm[i].load_state_dict(checkpoint_["components"][i])
  del checkpoint_

from torch.autograd import Variable

if "lm_embeddings" in checkpoint:
  print(lm.word_embeddings.weight)
  assert (checkpoint["lm_embeddings"]["weight"] == lm.word_embeddings.weight).all()
  del checkpoint["lm_embeddings"]
assert set(list(checkpoint)) == set(["arguments", "words", "memory", "autoencoder"]), list(checkpoint)
assert itos == checkpoint["words"]
for i in range(len(checkpoint["memory"])):
   memory.modules_memory[i].load_state_dict(checkpoint["memory"][i])
#for i in range(len(checkpoint["lm"])):
#   lm.modules_lm[i].load_state_dict(checkpoint["lm"][i])
for i in range(len(checkpoint["autoencoder"])):
   autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["autoencoder"][i])
#state = {"arguments" : args, "words" : itos, "memory" : memory, "lm" : lm, "autoencoder" : autoencoder}
#torch.save(state, "/u/scr/mhahn/CODEBOOKS_MEMORY/{__file__}_{args.myID}.model")




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


def getPunctuationMask(masks):
   assert len(masks) > 0
   if len(masks) == 1:
      return masks[0]
   else:
      punc1 = punctuation[:int(len(punctuation)/2)]
      punc2 = punctuation[int(len(punctuation)/2):]
      return torch.logical_or(getPunctuationMask(punc1), getPunctuationMask(punc2))

def product(x):
   r = 1
   for i in x:
     r *= i
   return r

PUNCTUATION = torch.LongTensor([stoi_total[x] for x in [".", "OOV", '"', "(", ")", "'", '"', ":", ",", "'s", "[", "]"]]).cuda()

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

      assert numeric.size()[0] == args.sequence_length+1, numeric.size()[0]
      ######################################################
      ######################################################
      # Run Loss Model
      if expandReplicates:
         numeric = numeric.expand(-1, NUMBER_OF_REPLICATES)
#      print(numeric.size(), beginning.size(), NUMBER_OF_REPLICATES)
#      numeric = torch.cat([beginning, numeric], dim=0)

      # Input: numeric
      # Output: memory_hidden

      memory_hidden, embedded_everything_mem = memory.forward(numeric)

      if provideAttention:
         return memory_hidden

      # Baseline predictions for prediction loss
#      baselineValues = 10*memory.sigmoid(memory.perword_baseline_outer(memory.relu(memory.perword_baseline_inner(embedded_everything_mem[-1].detach())))).squeeze(1)
 #     assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)


      # NOISE MEMORY ACCORDING TO MODEL
      memory_filter = torch.bernoulli(input=memory_hidden)
      bernoulli_logprob = torch.where(memory_filter == 1, torch.log(memory_hidden+1e-10), torch.log(1-memory_hidden+1e-10))
      bernoulli_logprob_perBatch = bernoulli_logprob.mean(dim=0)
      if args.entropy_weight > 0:
         entropy = -(memory_hidden * torch.log(memory_hidden+1e-10) + (1-memory_hidden) * torch.log(1-memory_hidden+1e-10)).mean()
      else:
         entropy=-1.0
      memory_filter = memory_filter.squeeze(2)

      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(12, 1, 1)).long().sum(dim=0)).bool())
        
      ####################################################################################
      numeric_noised = torch.where(torch.logical_or(punctuation, memory_filter==1), numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]
      numeric_onlyNoisedOnes = torch.where(memory_filter == 0, numeric, 0*numeric) # target is 0 in those places where no noise has happened

      if onlyProvideMemoryResult:
        return numeric, numeric_noised
      assert False, "this version of the code uses an unintialized lm"
      input_tensor_pure = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      target_tensor_full = Variable(numeric[1:], requires_grad=False)

      target_tensor_onlyNoised = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)
      #####################################################################################


      ##########################################
      ##########################################
      # RUN AUTOENCODER (approximately inverting loss model)
      autoencoder_lossTensor =  autoencoder.forward(input_tensor_pure, input_tensor_noised, target_tensor_onlyNoised, NUMBER_OF_REPLICATES)

      ##########################################
      ##########################################
      # RUN LANGUAGE MODEL (amortized prediction of next word)
#      if args.predictability_weight > 0:
 #      lm_lossTensor = lm.forward(input_tensor_noised, target_tensor_full, NUMBER_OF_REPLICATES)
      ##########################################
      ##########################################

      # Reward, term 1
#      if args.predictability_weight > 0:
#        negativeRewardsTerm1 = 2*args.predictability_weight * lm_lossTensor.mean(dim=0) + 2*(1-args.predictability_weight) * autoencoder_lossTensor.mean(dim=0)
#      else:
#        negativeRewardsTerm1 = autoencoder_lossTensor.mean(dim=0)


      # Reward, term 2
      # Regularization towards lower retention rates
      negativeRewardsTerm2 = memory_filter.mean(dim=0)
      retentionTarget = 1-args.deletion_rate
      loss = 0

      # Autoencoder Loss
      loss += autoencoder_lossTensor.mean()

      # Overall Reward
      #negativeRewardsTerm = negativeRewardsTerm1 + dual_weight * (negativeRewardsTerm2-retentionTarget)
      # for the dual weight
      #loss += (dual_weight * (negativeRewardsTerm2-retentionTarget).detach()).mean()
      #if printHere:
       #   print(negativeRewardsTerm1.mean(), dual_weight, negativeRewardsTerm2.mean(), retentionTarget)
      #print(loss)

      # baselineValues: the baselines for the prediction loss (term 1)
      # memory_hidden: baseline for term 2
      # Important to detach all but the baseline values

      # Reward Minus Baseline
      # Detached surprisal and mean retention
#      rewardMinusBaseline = (negativeRewardsTerm.detach() - baselineValues - args.RATE_WEIGHT * memory_hidden.mean(dim=0).squeeze(dim=1).detach())
#      rewardMinusBaseline = (negativeRewardsTerm.detach() - baselineValues - (dual_weight * (memory_hidden.mean(dim=0).squeeze(dim=1) - retentionTarget)).detach())

      # Important to detach from the baseline!!! 
#      loss += (rewardMinusBaseline.detach() * bernoulli_logprob_perBatch.squeeze(1)).mean()
 #     if args.entropy_weight > 0:
  #       loss -= args.entropy_weight  * entropy

      # Loss for trained baseline
   #   loss += args.reward_multiplier_baseline * rewardMinusBaseline.pow(2).mean()


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
 #     runningAverageBaselineDeviation = factor * runningAverageBaselineDeviation + (1-factor) * float((rewardMinusBaseline).abs().mean())

 #     if args.predictability_weight > 0:
#       runningAveragePredictionLoss = factor * runningAveragePredictionLoss + (1-factor) * round(float(lm_lossTensor.mean()),3)
      
      runningAverageReconstructionLoss = factor * runningAverageReconstructionLoss + (1-factor) * round(float(autoencoder_lossTensor.mean()),3)
 #     runningAverageReward = factor * runningAverageReward + (1-factor) * float(negativeRewardsTerm.mean())
      ############################

      if printHere:
  #       if args.predictability_weight > 0:
   #       lm_losses = lm_lossTensor.data.cpu().numpy()
         autoencoder_losses = autoencoder_lossTensor.data.cpu().numpy()

         numericCPU = numeric.cpu().data.numpy()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy()
         memory_hidden_CPU = memory_hidden[:,0,0].cpu().data.numpy()
#         memory_hidden_logit_per_wordtype_cpu = memory_hidden_logit_per_wordtype.cpu().data
 #        attention_bilinear_term = attention_bilinear_term.cpu().data
         #numeric_embedded_cpu = numeric_embedded.cpu().data
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
      assert False
      """ An optimization step for the resource-rational objective function """
      assert False, "this version of the code uses an unintialized lm"
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
def showAttention(word, POS=""):
    attention = forward(torch.cuda.LongTensor([stoi[word]+3 for _ in range(args.sequence_length+1)]).view(-1, 1), train=True, printHere=True, provideAttention=True)
    attention = attention[:,0,0]
    print(*(["SCORES", word, "\t"]+[round(x,2) for x in list(attention.cpu().data.numpy())] + (["POS="+POS] if POS != "" else [])))



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


#in and is the it
#time maintain metal commercially the was salt cut as two-year college flavorings solutions
#a alkali and and four-year into twice programme cation and \tduring addition ammonia burials
#often and used butyllithiums perfumery as this amide

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
    import scoreWithGPT2Medium as scoreWithGPT2
    sentences = [(1, [{"Item" : 1, "Condition" : 1, "Region" : i, "Word" : x} for i, x in  enumerate(["coach", "smiled", "at", "the", "player", "tossed"])])]
    with torch.no_grad():
      outFile = sys.stdout
#     with open("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/"+__file__+"_"+args.stimulus_file.replace("/", "-")+"_"+str(args.load_from_joint if SANITY != "ZeroLoss" else "ZERO")+"_"+SANITY, "w") as outFile:
      print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "Surprisal", "SurprisalReweighted", "Repetition"]), file=outFile)
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
            
            if not (args.criticalRegions is None) and regions[i] not in args.criticalRegions:
              continue
            for repetition in range(2):
              numerified = encodeContextCrop(" ".join(remainingInput[:i+1]), "later the nurse suggested they treat the patient with an antibiotic but in the end this did not happen . " + context)
              pointWhereToStart = max(0, args.sequence_length - len(context.split(" ")) - i - 1) # some sentences are too long
              assert pointWhereToStart >= 0, (args.sequence_length, i, len(context.split(" ")))
              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
     #         print(i, " ########### ", SANITY, VERBS)
    #          print(numerified.size())
              # Run the memory model. We collect 'numberOfSamples' many replicates.
              if SANITY == "Sanity":
                 numeric = numerified
                 numeric = numeric.expand(-1, numberOfSamples)
                 numeric_noised = torch.where(numeric == stoi["that"]+3, 0*numeric, numeric)
              elif SANITY == "ZeroLoss":
                 numeric = numerified
                 numeric = numeric.expand(-1, numberOfSamples)
                 numeric_noised = numeric
              else:
                 assert SANITY in ["Model", "ModelTmp"]
                 numeric, numeric_noised = forward(numerified, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=numberOfSamples)
                 numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
              # Next, expand the tensor to get 24 samples from the reconstruction posterior for each replicate
              numeric = numeric.unsqueeze(2).expand(-1, -1, 24).view(-1, numberOfSamples*24)
              numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)



              # First, get reconstructions when the new word is UNKNOWN
              numeric_noised[args.sequence_length] = 0 # A simple hack for dealing with the issue that the last word 
              # Now get samples from the amortized reconstruction posterior
              print("NOISED: ", " ".join([itos_total[int(x)] for x in numeric_noised[:,0].cpu()]))
              result, resultNumeric, fractions, thatProbs, amortizedPosterior = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*24, fillInBefore=pointWhereToStart)
              resultNumeric = resultNumeric.transpose(0,1).contiguous()
              likelihood = memory.compute_likelihood(resultNumeric, numeric_noised, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=1, computeProbabilityStartingFrom=pointWhereToStart, expandReplicates=False)
              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*24)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()

              # Second, get reconstructions when the new word is KNOWN
              print(numeric_noised.size(), nextWord.size(), resultNumeric.size())
              numeric_noised_second = torch.cat([numeric_noised[:-1], nextWord], dim=0).contiguous()
              result2, resultNumeric2, fractions2, thatProbs2, amortizedPosterior2 = autoencoder.sampleReconstructions(numeric, numeric_noised_second, None, 2, numberOfBatches=numberOfSamples*24, fillInBefore=pointWhereToStart)
              resultNumeric2 = resultNumeric2.transpose(0,1).contiguous()
              print(resultNumeric.size(), resultNumeric2.size())
              likelihood2 = memory.compute_likelihood(resultNumeric2, numeric_noised_second, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=1, computeProbabilityStartingFrom=pointWhereToStart, expandReplicates=False)
              print(likelihood.size(), likelihood2.size())

              likelihood = likelihood.view(numberOfSamples, 24)
              likelihood2 = likelihood2.view(numberOfSamples, 24)


              resultNumeric = resultNumeric.view(21, numberOfSamples, 24)
              resultNumeric2 = resultNumeric2.view(21, numberOfSamples, 24)

              numeric_noised = numeric_noised.view(21, numberOfSamples, 24)
              numeric_noised_second = numeric_noised_second.view(21, numberOfSamples, 24)

              likelihood = torch.cat([likelihood, likelihood2], dim=2)
              resultNumeric = torch.cat([resultNumeric, resultNumeric2], dim=2)
              numeric_noised = torch.cat([numeric_noised, numeric_noised_second], dim=2)


              quit()


              # Evaluate the prior on these samples to estimate next-word surprisal

              resultNumeric_cpu = resultNumeric.detach().cpu()
              batch = [" ".join([itos_total[resultNumeric_cpu[r,s]] for r in range(pointWhereToStart+1, resultNumeric.size()[0])]) for s in range(resultNumeric.size()[1])]
              for h in range(len(batch)):
                 batch[h] = batch[h][:1].upper() + batch[h][1:]
                 assert batch[h][0] != " ", batch[h]
#              # Add the preceding context
#              batchPreceding = [" ".join([itos_total[resultNumeric_cpu[r,s]] for r in range(0,pointWhereToStart+1)]) for s in range(resultNumeric.size()[1])]
#              batch = [x.replace(" .", ".")+" "+y for x, y in zip(batchPreceding, batch)]
#              print(batch)
              totalSurprisal = scoreWithGPT2.scoreSentences(batch)
              surprisals_past = torch.FloatTensor([x["past"] for x in totalSurprisal]).cuda().view(numberOfSamples, 24)
              surprisals_nextWord = torch.FloatTensor([x["next"] for x in totalSurprisal]).cuda().view(numberOfSamples, 24)

#              totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False, computeSurprisals=True, returnLastSurprisal=False, numberOfBatches=numberOfSamples*24)
#              assert resultNumeric.size()[0] == args.sequence_length+1
#              assert totalSurprisal.size()[0] == args.sequence_length
#              # For each of the `numberOfSamples' many replicates, evaluate (i) the probability of the next word under the Monte Carlo estimate of the next-word posterior, (ii) the corresponding surprisal, (iii) the average of those surprisals across the 'numberOfSamples' many replicates.
#              totalSurprisal = totalSurprisal.view(args.sequence_length, numberOfSamples, 24)
#              surprisals_past = totalSurprisal[:-1].sum(dim=0)
#              surprisals_nextWord = totalSurprisal[-1]

              # where numberOfSamples is how many samples we take from the noise model, and 24 is how many samples are drawn from the amortized posterior for each noised sample
              amortizedPosterior = amortizedPosterior.view(numberOfSamples, 24)
              likelihood = likelihood.view(numberOfSamples, 24)
    #          print(surprisals_past.size(), surprisals_nextWord.size(), amortizedPosterior.size(), likelihood.size())
   #           print(amortizedPosterior.mean(), likelihood.mean(), surprisals_past.mean(), surprisals_nextWord.mean())
              unnormalizedLogTruePosterior = likelihood - surprisals_past
  #            print(unnormalizedLogTruePosterior)
 #             print(amortizedPosterior.mean())
              assert float(unnormalizedLogTruePosterior.max()) <= 1e-5
              assert float(amortizedPosterior.max()) <= 1e-5
              log_importance_weights = unnormalizedLogTruePosterior - amortizedPosterior
              log_importance_weights_maxima, _ = log_importance_weights.max(dim=1, keepdim=True)
#              assert False, "the importance weights seem wacky"
              print(log_importance_weights[0])
              for j in range(24): # TODO the importance weights seem wacky
                 if j % 3 != 0:
                    continue
                 print(j, "@@", result[j], float(surprisals_past[0, j]), float(surprisals_nextWord[0, j]), float(log_importance_weights[0, j]), float(likelihood[0, j]), float(amortizedPosterior[0, j]))
              print(" ".join([itos_total[int(x)] for x in numeric_noised[:, 0].detach().cpu()]))
#              quit()
#              print(log_importance_weights_maxima)
              log_importance_weighted_probs_unnormalized = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima - surprisals_nextWord).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              log_importance_weights_sum = torch.log(torch.exp(log_importance_weights - log_importance_weights_maxima).sum(dim=1)) + log_importance_weights_maxima.squeeze(1)
              reweightedSurprisals = -(log_importance_weighted_probs_unnormalized - log_importance_weights_sum)
              #print(reweightedSurprisals.size())
              #quit()
#              print(log_importance_weighted_probs_unnormalized.size(), log_importance_weights_maxima.size())
              reweightedSurprisalsMean = reweightedSurprisals.mean()
#              quit()

              surprisalOfNextWord = surprisals_nextWord.exp().mean(dim=1).log().mean()
  #            print("PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in numerified[:,0]]), surprisalOfNextWord, reweightedSurprisalsMean)
   #           quit()
              # for printing
              nextWordSurprisal_cpu = surprisals_nextWord.view(-1).detach().cpu()
#              reweightedSurprisal_cpu = reweightedSurprisals.detach().cpu()
#              print(nextWordSurprisal_cpu.size())


              for q in range(0, min(3*24, resultNumeric.size()[1]),  24):
                  print("DENOISED PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in resultNumeric[:,q]]), float(nextWordSurprisal_cpu[q])) #, float(reweightedSurprisal_cpu[q//24]))
              print("SURPRISAL", i, regions[i], remainingInput[i],float( surprisalOfNextWord), float(reweightedSurprisalsMean))
              print("\t".join([str(w) for w in [sentenceID, ITEM, CONDITION, regions[i], remainingInput[i], round(float( surprisalOfNextWord),3), round(float( reweightedSurprisalsMean),3), repetition]]), file=outFile)


getSurprisalsStimuli(SANITY=("Model" if args.deletion_rate > 0 else "ZeroLoss"))
