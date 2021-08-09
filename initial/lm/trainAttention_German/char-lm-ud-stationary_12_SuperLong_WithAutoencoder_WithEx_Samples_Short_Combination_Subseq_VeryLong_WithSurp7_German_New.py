# German code constructed using English  code
# ...bination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_W.py


# Based on:
#  char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos12_Long.py (loss model & code for language model)
# And autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Both_Saving.py (autoencoder)
# And (for the plain LM): ../autoencoder/autoencoder2_mlp_bidir_AND_languagemodel_sample.py
print("Character aware!")
import os
# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys
import random
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="german")
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=random.choice([177741044])) # language model taking noised input
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=random.choice([971549136, 79606396])) #518982544, 188436350, 518982544, 310179465, 12916800])) # 310179465, has a corrupted file
#parser.add_argument("--load-from-plain-lm", dest="load_from_plain_lm", type=str, default=random.choice([67760999, 977691881])) #, 129313017])) #136525999])) #244706489, 273846868])) # plain language model without noise


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
parser.add_argument("--learning_rate_memory", type = float, default= random.choice([0.000002, 0.00001, 0.00002, 0.00005]))  # Can also use 0.0001, which leads to total convergence to deterministic solution withtin maximum iterations (March 25, 2021)   #, 0.0001, 0.0002 # 1e-7, 0.000001, 0.000002, 0.000005, 0.000007, 
parser.add_argument("--learning_rate_autoencoder", type = float, default= random.choice([0.001, 0.01, 0.1, 0.2])) # 0.0001, 
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--reward_multiplier_baseline", type=float, default=0.1)
parser.add_argument("--dual_learning_rate", type=float, default=random.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3]))
parser.add_argument("--momentum", type=float, default=random.choice([0.0, 0.3, 0.5, 0.7, 0.9])) # Momentum is helpful in facilitating convergence to a low-loss solution (March 25, 2021). It might be even more important for getting fast convergence than a high learning rate
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


#############################################################
# Vocabulary


vocab_path = f"/u/scr/mhahn/FAIR18/{args.language.lower()}-wiki-word-vocab.txt"
bpe_vocab_path = f"/u/scr/mhahn/FAIR18/{args.language.lower()}-wiki-word-vocab_BPE_50000_Parsed.txt"

itos_autoencoder = [None for _ in range(5000000)]
i2BPE = [None for _ in range(5000000)]
with open(vocab_path, "r") as inFile:
  with open(bpe_vocab_path, "r") as inFileBPE:
     for i in range(5000000):
        if i % 50000 == 0:
           print(i)
        word = next(inFile).strip().split("\t")
        bpe = next(inFileBPE).strip().split("\t")
        itos_autoencoder[i] = word[0]
        i2BPE[i] = bpe[0].split("@@ ")
stoi_autoencoder = dict([(itos_autoencoder[i],i) for i in range(len(itos_autoencoder))])


itos_autoencoder_total = ["<SOS>", "<EOS>", "OOV"] + itos_autoencoder
stoi_autoencoder_total = dict([(itos_autoencoder_total[i],i) for i in range(len(itos_autoencoder_total))])

with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
     itos_BPE = [x for x in inFile.read().strip().split("\n")]
with open("/u/scr/mhahn/FAIR18/german-wiki-word-vocab_BPE_50000.txt", "r") as inFile:
     itos_BPE += [x.replace(" ",  "") for x in inFile.read().strip().split("\n")]
assert len(itos_BPE) > 50000
stoi_BPE = dict([(itos_BPE[i],i) for i in range(len(itos_BPE))])
itos_BPE_total = ["SOS", "EOS", "OOV"] + itos_BPE





# Load Vocabulary
char_vocab_path = f"/u/scr/mhahn/FAIR18/{args.language.lower()}-wiki-word-vocab.txt"

itos = []
with open(char_vocab_path, "r") as inFile:
  for i in range(500000):
     if i % 20000 == 0:
       print(i)
     itos.append(next(inFile).strip().split("\t")[0])
#     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


itos_total = ["<SOS>", "<EOS>", "OOV"] + itos
stoi_total = dict([(itos_total[i],i) for i in range(len(itos_total))])


import random
import torch

print(torch.__version__)



class Autoencoder:
  """ Amortized Reconstruction Posterior """
  def __init__(self):
    self.rnn_encoder = torch.nn.LSTM(2*args.word_embedding_size, int(args.hidden_dim_autoencoder/2.0), args.layer_num, bidirectional=True).cuda()
    self.rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim_autoencoder, args.layer_num).cuda()
    self.output = torch.nn.Linear(args.hidden_dim_autoencoder, len(itos_BPE_total)).cuda()
    self.word_embeddings = torch.nn.Embedding(num_embeddings=len(itos_BPE_total), embedding_dim=2*args.word_embedding_size).cuda()
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

  def convertToBPE(self, input_tensor_pure, input_tensor_noised, withTarget=True):
      input_tensor_pure_cpu = input_tensor_pure.detach().cpu()
      input_tensor_noised_cpu = input_tensor_noised.detach().cpu()
      input_tensor_pure_bpe = [[] for _ in range(input_tensor_pure_cpu.size()[1])]
      input_tensor_noised_bpe = [[] for _ in range(input_tensor_pure_cpu.size()[1])]
      target_tensor_onlyNoised_bpe = [[] for _ in range(input_tensor_pure_cpu.size()[1])] if withTarget else None
      for i1 in range(input_tensor_pure_cpu.size()[0]-1):
        for i2 in range(input_tensor_pure_cpu.size()[1]):
            #print(itos_total[input_tensor_pure_cpu[i1,i2]])
            word = (itos_total[input_tensor_pure_cpu[i1,i2]])
            bpeRep = ([stoi_BPE[x]+3 if x in stoi_BPE else 2 for x in i2BPE[stoi_autoencoder[word]]] if word in stoi_autoencoder else [2,1])
            
            erased = (input_tensor_noised_cpu[i1,i2] == 0)
            for x in bpeRep:
              #print(i1, i2, word, itos_BPE_total[x])
              input_tensor_pure_bpe[i2].append(x)
              if erased:
                 input_tensor_noised_bpe[i2].append(0)
                 if withTarget:
                    target_tensor_onlyNoised_bpe[i2].append(x)
              else:
                 input_tensor_noised_bpe[i2].append(x)
                 if withTarget:
                    target_tensor_onlyNoised_bpe[i2].append(0)
      #print(input_tensor_pure_bpe) 
      maxLength = max(len(q) for q in input_tensor_pure_bpe)
      for i1 in range(len(input_tensor_pure_bpe)):
         input_tensor_pure_bpe[i1] = [0 for _ in range(maxLength - len(input_tensor_pure_bpe[i1]))] + input_tensor_pure_bpe[i1]
         input_tensor_noised_bpe[i1] = [0 for _ in range(maxLength - len(input_tensor_noised_bpe[i1]))] + input_tensor_noised_bpe[i1]
         if withTarget:
            target_tensor_onlyNoised_bpe[i1] = [0 for _ in range(maxLength - len(target_tensor_onlyNoised_bpe[i1]))] + target_tensor_onlyNoised_bpe[i1]
      input_tensor_pure_bpe = torch.cuda.LongTensor(input_tensor_pure_bpe).t().contiguous()
      input_tensor_noised_bpe = torch.cuda.LongTensor(input_tensor_noised_bpe).t().contiguous()
      if withTarget:
         target_tensor_onlyNoised_bpe = torch.cuda.LongTensor(target_tensor_onlyNoised_bpe).t().contiguous()
      return input_tensor_pure_bpe, input_tensor_noised_bpe, target_tensor_onlyNoised_bpe

  def forward(self, input_tensor_pure, input_tensor_noised, NUMBER_OF_REPLICATES):
      # INPUTS: input_tensor_pure, input_tensor_noised
      # OUTPUT: autoencoder_lossTensor
      # now convert to BPE
#      print(input_tensor_pure.size())
 #     print(input_tensor_noised.size())
      input_tensor_pure_bpe, input_tensor_noised_bpe, target_tensor_onlyNoised_bpe = self.convertToBPE(input_tensor_pure, input_tensor_noised)
      #print("614", input_tensor_pure_bpe.size(), input_tensor_noised_bpe.size(), target_tensor_onlyNoised_bpe.size())
      #print([len(q) for q in input_tensor_pure_bpe])
      #print(maxLength, input_tensor_pure.size())

      autoencoder_embedded = self.word_embeddings(input_tensor_pure_bpe)
      autoencoder_embedded_noised = self.word_embeddings(input_tensor_noised_bpe)
      autoencoder_out_encoder, _ = self.rnn_encoder(autoencoder_embedded_noised, None)
      autoencoder_out_decoder, _ = self.rnn_decoder(autoencoder_embedded, None)
      assert autoencoder_embedded.size()[0] >= args.sequence_length-1, (input_tensor_pure.size(),input_tensor_pure.size(), autoencoder_embedded.size(), args.sequence_length-1) # Note that this is different from autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py. Would be good if they were unified.
      assert autoencoder_embedded_noised.size()[0] >= args.sequence_length-1, (autoencoder_embedded.size()[0], args.sequence_length-1) # Note that this is different from autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py.

      autoencoder_attention = torch.bmm(self.attention_proj(autoencoder_out_encoder).transpose(0,1), autoencoder_out_decoder.transpose(0,1).transpose(1,2))
      autoencoder_attention = self.attention_softmax(autoencoder_attention).transpose(0,1)
      autoencoder_from_encoder = (autoencoder_out_encoder.unsqueeze(2) * autoencoder_attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      autoencoder_out_full = torch.cat([autoencoder_out_decoder, autoencoder_from_encoder], dim=2)


      autoencoder_logits = self.output(self.relu(self.output_mlp(autoencoder_out_full) ))
      autoencoder_log_probs = self.logsoftmax(autoencoder_logits)

      # Prediction Loss 
      #print(autoencoder_log_probs.size(), target_tensor_onlyNoised_bpe.size(), len(itos_BPE_total), NUMBER_OF_REPLICATES, args.batchSize)
      autoencoder_lossTensor = self.print_loss(autoencoder_log_probs.view(-1, len(itos_BPE_total)), target_tensor_onlyNoised_bpe.view(-1)).view(-1, NUMBER_OF_REPLICATES*args.batchSize)
      return autoencoder_lossTensor
 

 
  def sampleReconstructions(self, numeric, numeric_noised, NOUN, offset, numberOfBatches=args.batchSize*args.NUMBER_OF_REPLICATES, fillInBefore=-1, computeProbabilityStartingFrom=0):
      """ Draws samples from the amortized reconstruction posterior """
      #assert False, "not yet implemented with BPE"


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)

      input_tensor_pure_bpe, input_tensor_noised_bpe, target_tensor_onlyNoised_bpe = self.convertToBPE(input_tensor, input_tensor_noised, withTarget=False)



      #target_tensor = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)


 #     input_tensor = Variable(numeric[:-1], requires_grad=False)
#      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


#      embedded = self.word_embeddings(input_tensor)

      embedded_noised = self.word_embeddings(input_tensor_noised_bpe)

      out_encoder, _ = self.rnn_encoder(embedded_noised, None)



      hidden = None
      result  = ["" for _ in range(numberOfBatches)]
      result_numeric = [[] for _ in range(numberOfBatches)]
      embeddedLast = embedded_noised[0].unsqueeze(0)
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
             nextWord = input_tensor_pure_bpe[i:i+1]
          else:
            sampledFromDist = dist.sample()
            logProbForSampledFromDist = dist.log_prob(sampledFromDist).squeeze(0)
 #           print(logProbForSampledFromDist.size(), numeric_noised[i].size(), zeroLogProb.size())
            assert numeric_noised.size()[0] == args.sequence_length+1
            if i < args.sequence_length: # IMPORTANT make sure the last word -- which is (due to a weird design choice) cut off -- doesn't contribute to the posterior
               amortizedPosterior += torch.where(input_tensor_noised_bpe[i] == 0, logProbForSampledFromDist, zeroLogProb)

            nextWord = torch.where(input_tensor_noised_bpe[i] == 0, sampledFromDist, input_tensor_pure_bpe[i:i+1])
  #        print(nextWord.size())

          nextWordDistCPU = nextWord.cpu().numpy()[0]
#          print("line 335")
#          print(probs.size())
#          print(input_tensor_pure_bpe[i:i+1])
#          print(sampledFromDist)
#          print(nextWord)
          nextWordStrings = [itos_BPE_total[x] for x in nextWordDistCPU]
          for i in range(numberOfBatches):
             result[i] += " "+nextWordStrings[i]
             result_numeric[i].append( nextWordDistCPU[i] )
          embeddedLast = self.word_embeddings(nextWord)
#          print(embeddedLast.size())
      for r in result[:2]:
         print(r)
      if NOUN is not None and False:
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
      self.output = torch.nn.Linear(args.hidden_dim_lm, 50000+3).cuda()
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

       #print(target_tensor_full.size())
       #quit()
       target_tensor_full_relevant = target_tensor_full[-1]
#       if NUMBER_OF_REPLICATES == args.NUMBER_OF_REPLICATES:
       oovTensor = torch.zeros(target_tensor_full_relevant.size()).cuda().long() + 2
       target_tensor_full_relevant = torch.where(target_tensor_full_relevant < 50000, target_tensor_full_relevant, oovTensor)


 
       # Prediction Loss 
       lm_lossTensor = self.print_loss(lm_log_probs.view(-1, 50000+3), target_tensor_full_relevant.view(-1)).view(-1, NUMBER_OF_REPLICATES) # , args.batchSize is 1
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
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+"autoencoder2_mlp_bidir_Erasure_SelectiveLoss_WithoutComma_BPE.py"+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      autoencoder.modules_autoencoder[i].load_state_dict(checkpoint["components"][i])
  del checkpoint
 
# Amortized Prediction Posterior
if args.load_from_lm is not None:
  lm_file = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_NoComma_LargeVocab.py"
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+lm_file+"_code_"+str(args.load_from_lm)+".txt")
  for i in range(len(checkpoint["components"])):
      lm.modules_lm[i].load_state_dict(checkpoint["components"][i])
  del checkpoint

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
         if char == ",":
           continue
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

PUNCTUATION = torch.LongTensor([stoi_total[x] for x in [".", "OOV", '"', "(", ")", "'", '"', ":", ",", "[", "]"]]).cuda()

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

      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(11, 1, 1)).long().sum(dim=0)).bool())
        
      ####################################################################################
      numeric_noised = torch.where(torch.logical_or(punctuation, memory_filter==1), numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]
      numeric_onlyNoisedOnes = torch.where(memory_filter == 0, numeric, 0*numeric) # target is 0 in those places where no noise has happened

      if onlyProvideMemoryResult:
        return numeric, numeric_noised

      input_tensor_pure = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      target_tensor_full = Variable(numeric[1:], requires_grad=False)

      #target_tensor_onlyNoised = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)
      #####################################################################################


      ##########################################
      ##########################################
      # RUN AUTOENCODER (approximately inverting loss model)

      autoencoder_lossTensor =  autoencoder.forward(input_tensor_pure, input_tensor_noised, NUMBER_OF_REPLICATES)

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

# This could easily be made a function in the MemoryModel class
def compute_likelihood(numeric, numeric_noised, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False, NUMBER_OF_REPLICATES=args.NUMBER_OF_REPLICATES, expandReplicates=True, computeProbabilityStartingFrom=0):
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
 #     if provideAttention:
#         return memory_hidden

#      # Baseline predictions for prediction loss
 #     baselineValues = 10*memory.sigmoid(memory.perword_baseline_outer(memory.relu(memory.perword_baseline_inner(embedded_everything[-1].detach())))).squeeze(1)
  #    assert tuple(baselineValues.size()) == (NUMBER_OF_REPLICATES,)


      # NOISE MEMORY ACCORDING TO MODEL
      memory_filter = (numeric_noised != 0)
#      print(memory_filter.size(), memory_hidden.size())
      bernoulli_logprob = torch.where(memory_filter, torch.log(memory_hidden.squeeze(2)+1e-10), torch.log(1-memory_hidden.squeeze(2)+1e-10))

      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(11, 1, 1)).long().sum(dim=0)).bool())

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

def showAttention(word, POS=""):
    attention = forward(torch.cuda.LongTensor([stoi[word]+3 for _ in range(args.sequence_length+1)]).view(-1, 1), train=True, printHere=True, provideAttention=True)
    attention = attention[:,0,0]
    print(*(["SCORES", word, "\t"]+[round(x,2) for x in list(attention.cpu().data.numpy())] + (["POS="+POS] if POS != "" else [])))






nounsAndVerbs = []
nounsAndVerbs.append(["der Schulleiter",    "den der Lehrer",    "kritisierte",           "gefeuert wurde",           "erschien in der Zeitung", "Was the XXXX quoted in the newspaper?", "Y"])
nounsAndVerbs.append(["der Bildhauer",    "den der Maler",    "bewunderte",      "Talent hatte",          "war falsch", "Was the XXXX untrue?", "Y"])
nounsAndVerbs.append(["der Spezialist", "den der Künstler",     "kannte",         "ein Betrüger war",            "schockiert alle", "Did the XXXX shock everyone?", "Y"])
nounsAndVerbs.append(["der Marathonläufer",     "den der Psychiater",    "behandelte",        "betrogen hatte",      "schien lächerlich", "Was the XXXX ridiculous?", "Y"])
nounsAndVerbs.append(["das Kind",      "das der Sanitäter",     "rettete",  "unbeschadet war",      "erleichterte alle", "Did the XXXX relieve everyone?", "Y"])
nounsAndVerbs.append(["der Kriminelle",    "den die Polizisten",    "festnahmen",         "unschuldig war",       "war völlig falsch", "Was the XXXX bogus?", "Y"])
nounsAndVerbs.append(["der Student",     "der Professor",   "hasst",           "das Studium abbrach",       "machte den Professor glücklich", "Did the XXXX make the professor happy?", "Y"])
nounsAndVerbs.append(["der Boss",     "den der Journalist",     "darstellte",        "geflohen war",           "stellte sich als wahr heraus", "Did the XXXX turn out to be true?", "Y"])
nounsAndVerbs.append(["die Schauspielerin",            "den der Star",    "liebte",          "die Show verpasste",        "brachte sie zum Weinen", "Did the XXXX almost make her cry?", "Y"])
nounsAndVerbs.append(["der Pfarrer",    "den die Gemeindemitglieder",  "wählten",           "Geld stahl",    "war korrekt", "Did the XXXX prove to be true?", "Y"])
nounsAndVerbs.append(["der Musiker",   "den die Sponsoren",    "unterstützten",          "Drogen nahm",            "stimmt wahrscheinlich", "Was the XXXX likely true?", "Y"])
nounsAndVerbs.append(["der Abgeordnete",    "den der Diplomat",    "kannte",          "in der Stichwahl gewann",          "machte ihn wütend", "Did the XXXX make him angry?", "Y"])
nounsAndVerbs.append(["der Kommandant",    "den der Präsident",    "einsetzte",  "einen Krieg begann",     "beunruhigt die Leute", "Did the XXXX trouble people?", "Y"])
nounsAndVerbs.append(["das Opfer",    "das der Verbrecher",    "angriff",  "überlebten",     "beruhigt alle", "Did the XXXX calm everyone down?", "Y"])
nounsAndVerbs.append(["der Politiker",    "den der Bankier",    "unterstützte",  "Geldwäsche betrieb",     "war ein Schock für seine Anhänger", "Did the XXXX come as a shock?", "Y"])
nounsAndVerbs.append(["der Chirurg",    "den der Patient",    "bezahlte",  "keinen Doktor hatte",     "war keine Überraschung", "Was the XXXX unsurprising?", "Y"])
nounsAndVerbs.append(["der Spion",    "den der Agent",    "verfolgte",          "einen Preis bekam",     "war traurig", "Was the XXXX disconcerting?", "Y"])
nounsAndVerbs.append(["der Angestellte",    "den der Kunde",    "rief",  "ein Held war",     "schien absurd", "Did the XXXX seem absurd?", "Y"])
nounsAndVerbs.append(["der Händler",    "der Unternehmer",    "befragte",  "geheime Informationen hatte",     "wurde bestätigt", "Was the XXXX confirmed?", "Y"])
nounsAndVerbs.append(["der Chef",    "den der Angestellte",    "beeindruckte",  "in Rente ging",     "war korrekt", "Was the XXXX correct?", "Y"])
nounsAndVerbs.append(["der Taxifahrer", "den der Tourist", "fragte", "blind war", "schien schwer zu glauben", "", "Y"])
nounsAndVerbs.append(["der Buchhändler", "den der Dieb", "überfiel", "einen Herzinfarkt bekam", "schockiert seine Familie", "", "Y"])
nounsAndVerbs.append(["der Nachbar", "den die Frau", "verdächtigte", "ihren Hund ermordete", "war eine Lüge", "", "Y"])
nounsAndVerbs.append(["der Wissenschaftler", "dem der Bürgermeister", "vertraute", "verrückt war", "war eine böse Verleumdung", "", "Y"])
nounsAndVerbs.append(["der Schüler", "den der Junge", "schlug", "betrogen hatte", "schockiert seine Eltern", "", "Y"])
nounsAndVerbs.append(["der Betrüger", "den die Frau", "erkannte", "gefasst wurde", "beruhigt die Leute", "", "Y"])
nounsAndVerbs.append(["der Unternehmer", "den der Wohltäter", "finanzierte", "das Geld ausgab", "war eine Enttäuschung", "", "Y"])
nounsAndVerbs.append(["der Retter", "den der Schwimmer", "rief", "die Kinder rettete", "beeindruckte die ganze Stadt", "", "Y"])

for x in nounsAndVerbs:
   for i in [0,1,2,3,4]:
       for y in x[i].lower().split(" "):
          if stoi_total.get(y, 500000000) > 50000:
             print(y)

#nounsAndVerbs.append(["the senator",        "the diplomat",       "opposed"])

#nounsAndVerbs = nounsAndVerbs[:1]

topNouns = []



topNouns.append('Die Klage')
topNouns.append('Der Zweifel')
topNouns.append('Der Bericht')
topNouns.append('Die Kritik')
topNouns.append('Der Punkt')
topNouns.append('Die Sicherheit')
topNouns.append('Die Anordnung')
topNouns.append('Die Entscheidung')
topNouns.append('Das Zeichen')
topNouns.append('Die Schätzung')
topNouns.append('Die Aufforderung')
topNouns.append('Die Entdeckung')
topNouns.append('Der Beleg')
topNouns.append('Die Idee')
topNouns.append('Die Möglichkeit')
topNouns.append('Der Vorwurf')
topNouns.append('Die Erfahrung')
topNouns.append('Die Erklärung')
topNouns.append('Die Bestätigung')
topNouns.append('Die Spekulation')
topNouns.append('Die Information')
topNouns.append('Die Ankündigung')
topNouns.append('Der Glaube')
topNouns.append('Die Andeutung')
topNouns.append('Der Gedanke')
topNouns.append('Die Aussage')
topNouns.append('Das Gefühl')
topNouns.append('Der Eindruck')
topNouns.append('Der Beweis')
topNouns.append('Der Verdacht')
topNouns.append('Das Fazit')
topNouns.append('Die Hoffnung')
topNouns.append('Die Nachricht')
topNouns.append('Die Behauptung')
topNouns.append('Das Gerücht')
topNouns.append('Die Mitteilung')
topNouns.append('Die Wahrscheinlichkeit')
topNouns.append('Der Hinweis')
topNouns.append('Die Mutmaßung')
topNouns.append('Die Erkenntnis')
topNouns.append('Die Feststellung')
topNouns.append('Die Annahme')
topNouns.append('Die Vermutung')
topNouns.append('Die Befürchtung')
topNouns.append('Die Ansicht')
topNouns.append('Die Auffassung')
topNouns.append('Die Überzeugung')
topNouns.append('Der Schluss')
topNouns.append('Die Tatsache')


topNouns = [x for x in topNouns if stoi_total.get(x.split(" ")[1].lower(), 100000000) <= 50000]
topNouns = [x.lower().strip().split(" ") for x in topNouns]
articles = dict([x[::-1] for x in topNouns])
topNouns = [x[1] for x in topNouns]


with open("../../../../forgetting/corpus_counts/german/output/counts_ordered.tsv", "r") as inFile:
   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = counts[0]
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[1].lower() : line for line in counts}

topNouns = [x for x in topNouns if x in counts]
topNouns = sorted(list(set(topNouns)), key=lambda x:float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]]))

print(topNouns)
print(len(topNouns))



    
#plain_lm = PlainLanguageModel()
#plain_lmFileName = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_NoComma.py"
#
#if args.load_from_plain_lm is not None:
#  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+plain_lmFileName+"_code_"+str(args.load_from_plain_lm)+".txt")
#  for i in range(len(checkpoint["components"])):
#      plain_lm.modules[i].load_state_dict(checkpoint["components"][i])
#  del checkpoint
#
#
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


def getTotalSentenceSurprisalsCalibration(SANITY="Sanity", VERBS=2): # Surprisal for EOS after 2 or 3 verbs
    assert SANITY in ["ModelTmp", "Model", "Sanity", "ZeroLoss"]
    assert VERBS in [1,2]
#    print(plain_lm) 
    numberOfSamples = 12
    import scoreWithGPT2Medium as scoreWithGPT2
    with torch.no_grad():
     with open("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv-german/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w") as outFile:
      print("\t".join(["Sentence", "Region", "Word", "Surprisal", "SurprisalReweighted"]), file=outFile)
      TRIALS_COUNT = 0
      for sentenceID in range(len(calibrationSentences)):
          print(sentenceID)
          sentence = calibrationSentences[sentenceID].lower().replace(".", "").replace(",", "").replace("n't", " n't").split(" ")
          context = sentence[0]
          remainingInput = sentence[1:]
          regions = range(len(sentence))
          print("INPUT", context, remainingInput)
          assert len(remainingInput) > 0
          for i in range(len(remainingInput)):
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
                 numeric_noised = torch.where(numeric == stoi["dass"]+3, 0*numeric, numeric)
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
              numeric_noised[args.sequence_length] = 0 # A simple hack for dealing with the issue that the last word 
              # Now get samples from the amortized reconstruction posterior
              print("NOISED: ", " ".join([itos_total[int(x)] for x in numeric_noised[:,0].cpu()]))
              result, resultNumeric, fractions, thatProbs, amortizedPosterior = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*24, fillInBefore=pointWhereToStart)
              # get THAT fractions

              resultNumeric = resultNumeric.transpose(0,1).contiguous()


#              print(resultNumeric.size(), numeric_noised.size())
              likelihood = compute_likelihood(resultNumeric, numeric_noised, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=1, computeProbabilityStartingFrom=pointWhereToStart, expandReplicates=False)




              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*24)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()
              # Evaluate the prior on these samples to estimate next-word surprisal

              resultNumeric_cpu = resultNumeric.detach().cpu()
              batch = [" ".join([itos_total[resultNumeric_cpu[r,s]] for r in range(pointWhereToStart+1, resultNumeric.size()[0])]) for s in range(resultNumeric.size()[1])]
              for h in range(len(batch)):
                 batch[h] = batch[h][:1].upper() + batch[h][1:]
                 assert batch[h][0] != " ", batch[h]
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
              print("\t".join([str(w) for w in [sentenceID, regions[i], remainingInput[i], round(float( surprisalOfNextWord),3), round(float( reweightedSurprisalsMean),3)]]), file=outFile)

def divideDicts(y, z):
   r = {}
   for x in y:
     r[x] = y[x]/z[x]
   return r

def getTotalSentenceSurprisals(SANITY="Model", VERBS=2): # Surprisal for EOS after 2 or 3 verbs
    assert SANITY in ["ModelTmp", "Model", "Sanity", "ZeroLoss"]
    assert VERBS in [1,2]
    surprisalsPerNoun = {}
    surprisalsReweightedPerNoun = {}
    thatFractionsPerNoun = {}
    thatFractionsReweightedPerNoun = {}
    numberOfSamples = 12
    import scoreWithGPT2Medium as scoreWithGPT2
    global topNouns
#    topNouns = ["fact", "report"]
    with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem-german/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w") if SANITY != "ModelTmp" else sys.stdout as outFile:
     print("\t".join(["Noun", "Item", "Region", "Condition", "Surprisal", "SurprisalReweighted", "ThatFraction", "ThatFractionReweighted", "SurprisalsWithThat", "SurprisalsWithoutThat", "Word"]), file=outFile)
     with torch.no_grad():
      TRIALS_COUNT = 0
      TOTAL_TRIALS = len(topNouns) * len(nounsAndVerbs) * 2 * 1
      for nounIndex, NOUN in enumerate(topNouns):
        print(NOUN, "Time:", time.time() - startTimePredictions, nounIndex/len(topNouns), file=sys.stderr)
        thatFractions = {x : defaultdict(float) for x in ["grammatical", "ungrammatical"]}
        thatFractionsReweighted = {x : defaultdict(float) for x in ["grammatical", "ungrammatical"]}
        thatFractionsCount = {x : defaultdict(float) for x in ["grammatical", "ungrammatical"]}
        surprisalReweightedByRegions = {x : defaultdict(float) for x in ["grammatical", "ungrammatical"]}
        surprisalByRegions = {x : defaultdict(float) for x in ["grammatical", "ungrammatical"]}
        surprisalCountByRegions = {x : defaultdict(float) for x in ["grammatical", "ungrammatical"]}
        itemIDs = set()
        for sentenceID in range(len(nounsAndVerbs)):
          print(sentenceID)
          context = None
          for condition in ["grammatical", "ungrammatical"]:
            TRIALS_COUNT += 1
            print("TRIALS", TRIALS_COUNT/TOTAL_TRIALS)
            sentenceList = [x.lower() for x in nounsAndVerbs[sentenceID]]
            itemID = sentenceID
            compatible = "incompatible" #[None, "compatible", "incompatible"][sentenceListDict["compatible"]] if condition != "NoSC" else "neither"#
#       nounsAndVerbs.append(["der Schulleiter",    "den der Lehrer",    "kritisierte",           "gefeuert wurde",           "erschien in der Zeitung", "Was the XXXX quoted in the newspaper?", "Y"])    
            assert len(sentenceList) >= 4, sentenceList
            if condition == "grammatical":
               context = f"{articles[NOUN]} {NOUN} dass {sentenceList[0]} {sentenceList[1]} {sentenceList[2]}"
               regionsToDo = [(sentenceList[3], "V2"), (sentenceList[4].split(" ")[0], "V1")]
               remainingInput = flatten([x[0].split(" ") for x in regionsToDo])
               regions = flatten([[f"{region}_{c}" for c, _ in enumerate(words.split(" "))] for words, region in regionsToDo])
               assert len(remainingInput) == len(regions), (regionsToDo, remainingInput, regions)
            elif condition == "ungrammatical":
               context = f"{NOUN} dass {sentenceList[0]} {sentenceList[1]}"
               regionsToDo = [(sentenceList[4].split(" ")[0], "V1")]
               remainingInput = flatten([x[0].split(" ") for x in regionsToDo])
               regions = flatten([[f"{region}_{c}" for c, _ in enumerate(words.split(" "))] for words, region in regionsToDo])
               assert len(remainingInput) == len(regions), (regionsToDo, remainingInput, regions)
            else:
               assert False
            print("INPUT", context, remainingInput)
            assert len(remainingInput) > 0
            for i in range(len(remainingInput)):
              if regions[i] not in ["V1_0", "V1_1"]: #.startswith("V2"):
                continue
              numerified = encodeContextCrop(" ".join(remainingInput[:i+1]), "die schwester schlug vor den patienten mit einem antibiotikum zu behandeln aber dazu kam es nicht . " + context)
              pointWhereToStart = args.sequence_length - len(context.split(" ")) - i - 1
              assert pointWhereToStart >= 0, (args.sequence_length, i, len(context.split(" ")))
              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
     #         print(i, " ########### ", SANITY, VERBS)
    #          print(numerified.size())
              # Run the memory model. We collect 'numberOfSamples' many replicates.
              if SANITY == "Sanity":
                 numeric = numerified
                 numeric = numeric.expand(-1, numberOfSamples)
                 numeric_noised = torch.where(numeric == stoi["dass"]+3, 0*numeric, numeric)
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
              numeric_noised[args.sequence_length] = 0 # A simple hack for dealing with the issue that the last word 
              # Now get samples from the amortized reconstruction posterior
              print("NOISED: ", " ".join([itos_total[int(x)] for x in numeric_noised[:,0].cpu()]))
              result, resultNumeric, fractions, thatProbs, amortizedPosterior = autoencoder.sampleReconstructions(numeric, numeric_noised, NOUN, 2, numberOfBatches=numberOfSamples*24, fillInBefore=pointWhereToStart)
              # get THAT fractions
              if "NoSC" not in condition: # and i == 0:
                 resultNumericPrevious = resultNumeric
                 locationThat = context.split(" ")[::-1].index("dass")+i+2
                 thatFractionHere = float((resultNumeric[:, -locationThat] == stoi_total["dass"]).float().mean())
                 thatFractions[condition][regions[i]]+=thatFractionHere
                 thatFractionsCount[condition][regions[i]]+=1
              else:
                 thatFractionHere = -1
#                 print("\n".join(result))
 #                print(float((resultNumeric[:,-locationThat-2] == stoi_total["that"]).float().mean()))
                 
  #               print(locationThat, thatFractions[condition][regions[i]])
   #              quit()

              resultNumeric = resultNumeric.transpose(0,1).contiguous()


#              print(resultNumeric.size(), numeric_noised.size())
              likelihood = compute_likelihood(resultNumeric, numeric_noised, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=1, computeProbabilityStartingFrom=pointWhereToStart, expandReplicates=False)




              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*24)]).unsqueeze(0).cuda()
              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()
              # Evaluate the prior on these samples to estimate next-word surprisal

              resultNumeric_cpu = resultNumeric.detach().cpu()
              batch = [" ".join([itos_total[resultNumeric_cpu[r,s]] for r in range(pointWhereToStart+1, resultNumeric.size()[0])]) for s in range(resultNumeric.size()[1])]
              for h in range(len(batch)):
                 batch[h] = batch[h][:1].upper() + batch[h][1:]
                 assert batch[h][0] != " ", batch[h]
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

              if "NoSC" not in condition: # and i == 0:
#                 print((resultNumericPrevious[:, -locationThat] == stoi_total["that"]).size(), log_importance_weights.size(), log_importance_weights_sum.size())
 #                print(torch.exp(log_importance_weights - log_importance_weights_sum.unsqueeze(1)))
  #               print(torch.exp(log_importance_weights - log_importance_weights_sum.unsqueeze(1)).sum(dim=1))
#                 print(surprisals_nextWord.size(), (((resultNumericPrevious[:, -locationThat] == stoi_total["that"]).float().view(-1, 24) .size())))
                 surprisalsWithThat = float(surprisals_nextWord[(resultNumericPrevious[:, -locationThat] == stoi_total["dass"]).view(-1, 24)].mean())
                 surprisalsWithoutThat = float(surprisals_nextWord[(resultNumericPrevious[:, -locationThat] != stoi_total["dass"]).view(-1, 24)].mean())
                 print("Surp with and without that", surprisalsWithThat, surprisalsWithoutThat)               
                 thatFractionReweightedHere = float((((resultNumericPrevious[:, -locationThat] == stoi_total["dass"]).float().view(-1, 24) * torch.exp(log_importance_weights - log_importance_weights_sum.unsqueeze(1))).sum(dim=1)).mean())
                 thatFractionsReweighted[condition][regions[i]]+=thatFractionReweightedHere
   #              print((((resultNumericPrevious[:, -locationThat] == stoi_total["that"]).float().view(-1, 24) * torch.exp(log_importance_weights - log_importance_weights_sum.unsqueeze(1))).sum(dim=1)).mean())
    #             print(((resultNumericPrevious[:, -locationThat] == stoi_total["that"]).float().mean()))
     #            quit()

              else:
                 thatFractionReweightedHere = -1


              for q in range(0, min(3*24, resultNumeric.size()[1]),  24):
                  print("DENOISED PREFIX + NEXT WORD", " ".join([itos_total[int(x)] for x in resultNumeric[:,q]]), float(nextWordSurprisal_cpu[q])) #, float(reweightedSurprisal_cpu[q//24]))
              print("SURPRISAL", NOUN, sentenceList[0], condition, i, regions[i], remainingInput[i],float( surprisalOfNextWord), float(reweightedSurprisalsMean))
              surprisalReweightedByRegions[condition][regions[i]] += float( reweightedSurprisalsMean)
              surprisalByRegions[condition][regions[i]] += float( surprisalOfNextWord)
              surprisalCountByRegions[condition][regions[i]] += 1

              #assert sentenceList[-1] in ["o","v"]
              print("\t".join([str(w) for w in [NOUN, itemID, regions[i], condition, round(float( surprisalOfNextWord),3), round(float( reweightedSurprisalsMean),3), int(100*thatFractionHere), int(100*thatFractionReweightedHere), surprisalsWithThat, surprisalsWithoutThat, remainingInput[i]]]), file=outFile)
#                 print("Surp with and without that", surprisalsWithThat, surprisalsWithoutThat)               


           #   if compatible == "compatible":
            #    hasSeenCompatible = True
#              if i == 0 or regions[i] != regions[i-1]:
        print(surprisalByRegions)
        print(surprisalReweightedByRegions)
        print(thatFractions)
        print("NOUNS SO FAR", topNouns.index(NOUN))
        assert NOUN not in surprisalsPerNoun # I think that in previous versions of these scripts the indentation was wrong, and this was overwitten multiple times
        assert NOUN not in surprisalsReweightedPerNoun # I think that in previous versions of these scripts the indentation was wrong, and this was overwitten multiple times
        print(surprisalByRegions)
        surprisalsReweightedPerNoun[NOUN] = {x : divideDicts(surprisalReweightedByRegions[x], surprisalCountByRegions[x]) for x in surprisalReweightedByRegions}
        surprisalsPerNoun[NOUN] = {x : divideDicts(surprisalByRegions[x], surprisalCountByRegions[x]) for x in surprisalByRegions}
        thatFractionsReweightedPerNoun[NOUN] = {x : divideDicts(thatFractionsReweighted[x], thatFractionsCount[x]) for x in thatFractionsReweighted}
        thatFractionsPerNoun[NOUN] = {x : divideDicts(thatFractions[x], thatFractionsCount[x]) for x in thatFractions}
        print(thatFractionsPerNoun[NOUN])
        #quit()
        #quit()
        #assert hasSeenCompatible
    print("SURPRISALS BY NOUN", surprisalsPerNoun)
    print("THAT (fixed) BY NOUN", thatFractionsPerNoun)
    print("SURPRISALS_PER_NOUN PLAIN_LM, WITH VERB, NEW")
    with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-german/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w")  if SANITY != "ModelTmp" else sys.stdout as outFile:
      print("Noun", "Region", "Condition", "Surprisal", "SurprisalReweighted", "ThatFraction", "ThatFractionReweighted", file=outFile)
      for noun in topNouns:
 #      assert "SCRC_incompatible" in surprisalsPerNoun[noun], list(surprisalsPerNoun[noun])
#       assert "SCRC_compatible" in surprisalsPerNoun[noun], list(surprisalsPerNoun[noun])
#       assert len(surprisalsPerNoun[noun]["SCRC_compatible"]) > 0
       for condition in surprisalsPerNoun[noun]:
#         assert "V1_0" in thatFractionsPerNoun[noun][condition], list(thatFractionsPerNoun[noun][condition])
#         assert "V1_0" in surprisalsPerNoun[noun][condition], list(surprisalsPerNoun[noun][condition])
         for region in surprisalsPerNoun[noun][condition]:
           print(noun, region, condition, surprisalsPerNoun[noun][condition][region], surprisalsReweightedPerNoun[noun][condition][region], thatFractionsPerNoun[noun][condition][region] if "NoSC" not in condition else "NA", thatFractionsReweightedPerNoun[noun][condition][region] if "NoSC" not in condition else "NA", file=outFile)
    # For sanity-checking: Prints correlations between surprisal and that-bias
    for region in ["V1_0"]:
      for condition in surprisalsPerNoun["fact"]:
       if region not in surprisalsPerNoun["fact"][condition]:
          continue
       print(SANITY, condition, "CORR", region, correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([surprisalsPerNoun[x][condition][region] for x in topNouns])), correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([surprisalsReweightedPerNoun[x][condition][region] for x in topNouns])), correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([thatFractionsPerNoun[x][condition][region] for x in topNouns])) if "NoSC" not in condition else 0 , correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), torch.FloatTensor([thatFractionsReweightedPerNoun[x][condition][region] for x in topNouns])) if "NoSC" not in condition else 0 )
       surprisals = torch.FloatTensor([surprisalsPerNoun[x][condition][region] for x in topNouns])
       print(condition, surprisals.mean(), "SD", math.sqrt(surprisals.pow(2).mean() - surprisals.mean().pow(2)))
#    overallSurprisalForCompletion = torch.FloatTensor([sum([surprisalsPerNoun[noun]["SC"][region] - surprisalsPerNoun[noun]["NoSC"][region] for region in surprisalsPerNoun[noun]["SC"]]) for noun in topNouns])
 #   print(SANITY, "CORR total", correlation(torch.FloatTensor([thatBias(x) for x in topNouns]), overallSurprisalForCompletion), "note this is inverted!")


#getTotalSentenceSurprisalsCalibration(SANITY="ModelTmp")
#quit()


startTimePredictions = time.time()

#getTotalSentenceSurprisals(SANITY="ZeroLoss")
#getTotalSentenceSurprisals(SANITY="Sanity")
#getTotalSentenceSurprisals(SANITY="Model")
getTotalSentenceSurprisals(SANITY="ModelTmp")
quit()


#getTotalSentenceSurprisals()
#quit()

#incrementallySampleCompletions(SANITY="Model", VERBS=1)
#getPerNounReconstructions2VerbsUsingPlainLM(SANITY="Model", VERBS=1)
#quit()

startTimeTotal = time.time()
startTimePredictions = time.time()
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


       with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs-german/"+__file__+"_"+str(args.myID), "w") as outFile:
         print(updatesCount, "Slurm", os.environ["SLURM_JOB_ID"], file=outFile)
         print(args, file=outFile)


       # Record calibration for the acceptability judgments
       #getTotalSentenceSurprisalsCalibration(SANITY="Model")
       
       # Record reconstructions and surprisals
       with open("/u/scr/mhahn/reinforce-logs-both-short/full-logs-german/"+__file__+"_"+str(args.myID), "w") as outFile:
         startTimePredictions = time.time()

         sys.stdout = outFile
         print(updatesCount, "Slurm", os.environ["SLURM_JOB_ID"])
         print(args)
         getTotalSentenceSurprisals(SANITY="Model")
  

#         getPerNounReconstructionsSanity()
#         getPerNounReconstructionsSanityVerb()
#         getPerNounReconstructions()
#         getPerNounReconstructionsVerb()
#         getPerNounReconstructions2Verbs()
         print("=========================")
         showAttention("der")
         showAttention("war")
         showAttention("ist")
         showAttention("dass")
         showAttention("tatsache")
         showAttention("information")
         showAttention("bericht")
         showAttention("von")
         

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
          print("Slurm", os.environ["SLURM_JOB_ID"])
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


with open("/u/scr/mhahn/reinforce-logs-both-short/results-german/"+__file__+"_"+str(args.myID), "w") as outFile:
   print(args, file=outFile)
   print(runningAverageReward, file=outFile)
   print(expectedRetentionRate, file=outFile)
   print(runningAverageBaselineDeviation, file=outFile)
   print(runningAveragePredictionLoss, file=outFile)
   print(runningAverageReconstructionLoss, file=outFile)


print("=========================")
showAttention("der")
showAttention("war")
showAttention("ist")
showAttention("dass")
showAttention("tatsache")
showAttention("information")
showAttention("bericht")
showAttention("von")


