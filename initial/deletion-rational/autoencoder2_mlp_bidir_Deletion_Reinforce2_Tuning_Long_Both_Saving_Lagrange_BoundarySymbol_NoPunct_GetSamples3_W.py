print("Character aware!")
import random


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys
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


#sys.stdout = open("/u/scr/mhahn/reinforce-logs/full-logs/"+__file__+"_"+str(args.myID), "w")

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

#          print(hasSampledSOSTwice.size(), hasSampledSOSTwice.size(), logProbForSampledFromDist.size(), zeroLogProb.size())
          amortizedPosterior += torch.where(hasSampledSOSOnce, logProbForSampledFromDist, zeroLogProb)
#          print(i, itos_total[int(nextWord[0,0])], hasSampledSOSOnce[0], float(amortizedPosterior[0]))
#          hasSampledSOSTwice = torch.logical_and(hasSampledSOSOnce, nextWord.squeeze(0) == stoi_total["<EOS>"])
          hasSampledSOSOnce = torch.logical_xor(hasSampledSOSOnce, nextWord.squeeze(0) == stoi_total["<EOS>"])
  #        print(nextWord.size())
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for i in range(numberOfBatches):
             result[i] += " "+nextWordStrings[i]
             result_numeric[i].append( nextWordDistCPU[i] )
          embeddedLast = self.word_embeddings(nextWord)
#          print(embeddedLast.size())
      for r in result[:2]:
         print("RECONSTRUCTION", r)
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


import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit =True)

from collections import defaultdict

results = {}
                                                                                                                                                                          
import scoreWithGPT2           

def normalize_dict(d):
   t = sum(y for _, y in d.items())
   return {x : y/t for x, y in d.items()}

with open("/u/scr/mhahn/stimuli-scr/gibson2013/E11_all_implaus_raw_data_2013.csv", "r") as inFile:
   data = [x.split(",") for x in inFile.read().strip().split("\n")]
header = data[0]
header = dict(list(zip(header, range(len(header)))))
data = data[1:]
processedSentences = set()
with torch.no_grad():
  for line in data:
      sentence = line[header['"Input.trial_"']].strip('"').strip(".").replace("'s ", " 's ").lower()
      if sentence in processedSentences:
        continue
      processedSentences.add(sentence)
      print(len(processedSentences), file=sys.stderr)
      condition = line[header['"Condition"']].strip('"')
      if "filler" in condition:
        continue
      sentence_proc = sentence
      OOVs = []
      for x in sentence_proc.split(" "):
        if x not in stoi_total:
          print("OOV", x, file=sys.stderr)
          OOVs.append(x)
      context = "later the nurse suggested they treat the patient with an antibiotic but in the end this did not happen <EOS> "+sentence_proc+ " <EOS> after this something else happened instead and she went away but nobody noticed anything about it"
      numberOfSamples = 12
      numerified = encodeContextCrop(context, "", replicates=numberOfSamples)
      assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
      # Run the noise model
      numeric, numeric_noised = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
      print(numeric.size(), numeric_noised.size(), numerified.size())

      numeric = numeric.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
      numeric_noised_original = numeric_noised
      numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
      print("numeric_noised_original", numeric_noised_original.size(), numeric_noised.size())
      # Get samples from the reconstruction posterior

      _, resultNumeric, fractions, thatProbs, amortizedPosterior = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*24)
      
      resultNumeric = resultNumeric.transpose(0,1).contiguous()
      resultNumeric_gpu = resultNumeric
      resultNumeric = resultNumeric.cpu()

      print(resultNumeric.size(), numeric_noised.size())
      keep_probabilities = forward((resultNumeric_gpu, None), train=False, printHere=False, provideAttention=False, onlyProvideKeepProbabilities=True)
      keep_probabilities = keep_probabilities.squeeze(2).detach().cpu()
      sampled_results = []
      likelihood = torch.zeros(numberOfSamples*24)
      for i in range((numberOfSamples*24)):
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
#      quit()
      resultNumeric_cpu = resultNumeric.detach().cpu()                                                                                                                                              
      batch = [" ".join([itos_total[x] for x in sampled_results[s]]) for s in range(resultNumeric.size()[1])]                                 
      totalSurprisal = scoreWithGPT2.scoreSentences(batch)                                                                                                                                          
      surprisals_past = torch.FloatTensor([x["past"] for x in totalSurprisal]).cuda().view(numberOfSamples, 24)                                                                                     

      surprisals_past = -surprisals_past.view(numberOfSamples, 24)
      likelihood = likelihood.view(numberOfSamples, 24)
      amortizedPosterior = amortizedPosterior.view(numberOfSamples, 24)
      #print(likelihood.size())
      #print(amortizedPosterior.size())
      #print(torch.max(surprisals_past, dim=1))
      #print(likelihood)
      surprisals_past = surprisals_past - torch.max(surprisals_past, dim=1)[0].unsqueeze(1)
      amortizedPosterior = amortizedPosterior - torch.max(amortizedPosterior, dim=1)[0].unsqueeze(1)
      importanceWeights_unnormalized = torch.exp(surprisals_past - amortizedPosterior).detach().cpu() * likelihood
      importanceWeights_sums = importanceWeights_unnormalized.sum(dim=1).unsqueeze(1)+1e-12
      importanceWeights = importanceWeights_unnormalized / importanceWeights_sums
      importanceWeights = importanceWeights.view(numberOfSamples*24).numpy().tolist()

#      quit()


      #print(resultNumeric.size())
      sentences = defaultdict(int)
      for i in range(numberOfSamples*24):
            decoded = [itos_total[x] for x in sampled_results[i]]
            OOVs_Ind = [j for j in range(len(decoded)) if decoded[j] == "OOV"]
            if len(OOVs_Ind) == len(OOVs) and len(OOVs) > 0:
                for q, j in enumerate(OOVs_Ind):
                  decoded[j] = OOVs[q]
#                assert False, (decoded, (" ".join([itos_total[int(resultNumeric[j,i])] for j in range(resultNumeric.size()[0])])))
            decoded = " ".join(decoded)
            sentences[decoded]+=importanceWeights[i]
#            print(decoded)
      sentences_list = list(sentences)
      toparse = [x for x in sentences_list + [sentence]]
      processed = nlp("\n\n".join(toparse))
  #    print(processed.sentences[0].words[0])
      originalSentenceParsed = processed.sentences[-1]
#      print(originalSentenceParsed.dependencies)
 #     print(originalSentenceParsed.text)
      nsubjs = [x for x in originalSentenceParsed.dependencies if x[1] in ["nsubj", "nsubj:pass"]]
      objs = sorted([x for x in originalSentenceParsed.dependencies if x[1] in ["iobj", "obj"]], key=lambda x:int(x[2].id))
      obls = [x for x in originalSentenceParsed.dependencies if x[1] == "obl"]
#      print(nsubjs)
 #     print(objs)
  #    print(obls)
   #   print(condition)
      annotations = {}
      if condition.startswith("DO_for_implausible"):
        try:
           assert len(nsubjs) == 1
           assert len(objs) == 2
           assert len(obls) == 0
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        object2 = objs[1][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           dependencyOfObject2 = [x.deprel for x in processed.sentences[i].words if x.text == object2]
           if len(dependencyOfObject1) == 0 or len(dependencyOfObject2) == 0:
             answer = "unknown"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("obl",):
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("nmod",): # made the quilt of her granddaughter
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) in [("obj",), ("iobj",)] and tuple(dependencyOfObject2) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject2) == ("conj",):
             answer = "other_conj"
           elif tuple(dependencyOfObject1) == ("compound",):
             answer = "other_compound"
           else:
             answer = "unknown"
        #   print(dependencyOfSubject, dependencyOfObject1, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1, dependencyOfObject2)
#           quit()
      elif condition.startswith("DO_implausible"):
        try:
          assert len(nsubjs) == 1
          assert len(objs) == 2
          assert len(obls) == 0
        except AssertionError:
          print("Parsing Error!!!", sentence)
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        object2 = objs[1][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           dependencyOfObject2 = [x.deprel for x in processed.sentences[i].words if x.text == object2]
           if len(dependencyOfObject1) == 0 or len(dependencyOfObject2) == 0:
             answer = "unknown"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("obl",):
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("nmod",): # made the quilt of her granddaughter
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("iobj",) and tuple(dependencyOfObject2) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("nsubj",) and tuple(dependencyOfObject2) == ("obj",):
             answer = "other"
           elif tuple(dependencyOfObject1) == ("compound",):
             answer = "other_compound"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfObject1, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1, dependencyOfObject2)
#           quit()
      elif condition in ["PO_plausible", "PO_for_plausible"]: # the uncle sold the truck to the father
        try:
          assert len(nsubjs) == 1, sentence
          assert len(objs) == 1, sentence
          assert len(obls) == 1, sentence
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfObject1) in [("iobj",), ("obj",)] and  tuple(dependencyOfOblique) == ("obj",):
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           elif tuple(dependencyOfOblique) == ("conj",):
             answer = "other_conj"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
      elif condition.startswith("implaus_obj"):
        try:
          assert len(nsubjs) == 1, sentence
          assert len(objs) == 0, sentence
          assert len(obls) == 1, sentence
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the check was written on the defendant 's name
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfOblique) in [("nsubj:pass",), ("nsubj",)]:
             answer = "nonliteral"
           elif tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
      elif condition.startswith("Preposition_plausible"):
        try:
           assert len(nsubjs) == 1, sentence
           assert len(objs) == 0, sentence
           assert len(obls) == 1, sentence
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfOblique) == ("obj",):
             answer = "nonliteral"
           elif tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
      elif condition.startswith("Preposition_implausible"):
        try:
           assert len(nsubjs) == 1, (processed.sentences[-1].dependencies, sentence)
           assert len(objs) == 0, sentence
           assert len(obls) == 1, sentence
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfOblique) == ("obj",):
             answer = "nonliteral"
           elif tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
      elif condition.startswith("DO_for_plausible") or condition.startswith("DO_plausible"): # the father gave the son the car
        try:
          assert len(nsubjs) == 1, sentence
          assert len(objs) == 2, sentence
          assert len(obls) == 0, sentence
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        object2 = objs[1][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           dependencyOfObject2 = [x.deprel for x in processed.sentences[i].words if x.text == object2]
           if tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("obl",):
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("nmod",):
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("iobj",) and tuple(dependencyOfObject2) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("obj",) and tuple(dependencyOfObject2) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("compound",):
             answer = "other_compound"
           elif tuple(dependencyOfObject2) == ("conj",):
             answer = "other_conj"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfObject1, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1, dependencyOfObject2)
#           quit()
      elif condition == "implaus_subj":
        try:
          assert len(nsubjs)+len(objs) == 1, ([x[1] for x in originalSentenceParsed.dependencies], sentence)
          assert len(obls) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = (nsubjs+objs)[0][2].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfSubject) in [("obj",), ("nsubj",), ("nsubj:pass",)] and tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           elif tuple(dependencyOfSubject) == ("obl",) and tuple(dependencyOfOblique) in [("obj",), ("nsubj",), ("nsubj:pass",)]:
             answer = "nonliteral"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
           print(sentences_list[i])
      elif condition == "plaus_subj":
        try:
           assert len(nsubjs) == 1, ([x[1] for x in originalSentenceParsed.dependencies], sentence)
           assert len(obls) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfSubject) in [("nsubj",), ("nsubj:pass",)] and tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           elif tuple(dependencyOfSubject) == ("obl",) and tuple(dependencyOfOblique) == ("nsubj",):
             answer = "nonliteral"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
 #          print(sentences_list[i])
      elif condition in ["PO_for_implausible", "PO_implausible"]:
        try:
           assert len(nsubjs) == 1, [x[1] for x in originalSentenceParsed.dependencies]
           assert len(obls) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfSubject) in [("nsubj",), ("nsubj:pass",)] and tuple(dependencyOfOblique) == ("obl",) and tuple(dependencyOfObject1) in [("iobj",), ("obj",)]:
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("compound",):
             answer = "other_compound"
           elif tuple(dependencyOfOblique) == ("conj",):
             answer = "other_conj"
           elif tuple(dependencyOfOblique) == ("obj",):
             answer = "nonliteral"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1, dependencyOfOblique)
           print(sentences_list[i])
#           quit()

      elif condition in ["passive_v1", "passive_v2"]:
        try:
          assert len(nsubjs) == 1, [x[1] for x in originalSentenceParsed.dependencies]
          assert len(obls) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfSubject) in [("nsubj",), ("nsubj:pass",)] and tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           elif tuple(dependencyOfSubject) == ("nsubj",) and tuple(dependencyOfOblique) == ("obj",):
             answer = "nonliteral"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
 #          print(sentences_list[i])
#           quit()

      elif condition == "plaus_obj":
        try:
          assert len(nsubjs) + len(objs) == 1, [x[1] for x in originalSentenceParsed.dependencies]
          assert len(obls) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = (nsubjs+objs)[0][2].text # In locative inversion sentences, the parser labels the second NP as object
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfSubject) in [("obj",), ("nsubj",), ("nsubj:pass",)] and tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           elif tuple(dependencyOfSubject) == ("obl",) and tuple(dependencyOfOblique) in [("obj",), ("nsubj",)]:
             answer = "nonliteral"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfOblique, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
           print(sentences_list[i])
#           quit()

      elif condition == "Transitive_plausible":
        try:
          assert len(nsubjs) == 1
          assert len(objs) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           if tuple(dependencyOfSubject) == ("nsubj",) and tuple(dependencyOfObject1) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("obl",):
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("obj",):
             answer = "literal"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfObject1, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1)
 #          print(sentences_list[i])
#           quit()

      elif condition == "Transitive_implausible":
        try:
          assert len(nsubjs) == 1
          assert len(objs) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           if tuple(dependencyOfSubject) == ("nsubj",) and tuple(dependencyOfObject1) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("obl",):
             answer = "nonliteral"
           elif tuple(dependencyOfObject1) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfObject1) == ("conj",):
             answer = "other_conj"
           elif tuple(dependencyOfObject1) == ("nmod",):
             answer = "other_nmod"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfObject1, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1)
 #          print(sentences_list[i])
#           quit()

      elif condition in ["active_implausible", "active_v2", "active_v1"]:
        try:
          assert len(nsubjs) == 1
          assert len(objs) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           if tuple(dependencyOfSubject) == ("nsubj",) and tuple(dependencyOfObject1) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfSubject) == ("nsubj:pass",) and tuple(dependencyOfObject1) == ("obl",):
             answer = "nonliteral"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfObject1, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1)
 #          print(sentences_list[i])
#              elif condition == "active_v2":
      elif condition in ["passive_plausible", "passive_implausible"]:
        try:
          assert len(nsubjs) == 1
          assert len(obls) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        oblique = obls[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfOblique = [x.deprel for x in processed.sentences[i].words if x.text == oblique]
           if tuple(dependencyOfSubject) == ("nsubj:pass",) and tuple(dependencyOfOblique) == ("obl",):
             answer = "literal"
           elif tuple(dependencyOfSubject) == ("nsubj",) and tuple(dependencyOfOblique) == ("obj",):
             answer = "nonliteral"
           else:
             answer = "unknown"
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfOblique)
      elif condition == "active_plausible":
        try:
          assert len(nsubjs) == 1
          assert len(objs) == 1
        except AssertionError:
          print("PARSING ERROR", [(x.text, x.deprel) for x in processed.sentences[-1].words])
          continue
        subject = nsubjs[0][2].text
        verb = nsubjs[0][0].text
        object1 = objs[0][2].text
        for i in range(len(sentences_list)): # the grandma made a quilt her granddaughter
           dependencyOfSubject = [x.deprel for x in processed.sentences[i].words if x.text == subject]
           dependencyOfObject1 = [x.deprel for x in processed.sentences[i].words if x.text == object1]
           if tuple(dependencyOfSubject) == ("nsubj",) and tuple(dependencyOfObject1) == ("obj",):
             answer = "literal"
           elif tuple(dependencyOfSubject) == ("nsubj:pass",) and tuple(dependencyOfObject1) == ("obl",):
             answer = "nonliteral"
           else:
             answer = "unknown"
#           print(dependencyOfSubject, dependencyOfObject1, dependencyOfObject2, answer)
           annotations[sentences_list[i]] = (answer, dependencyOfSubject, dependencyOfObject1)
 #          print(sentences_list[i])
#           quit()

      else:
         assert False, (originalSentenceParsed.text, condition)
      question = line[header['"Input.question_1_"']].strip('"')
      sentences = sorted(list(sentences.items()), key=lambda x:x[1])
      for x, y in sentences:
        if y > 0.1:
          print(x, "\t", y, "\t", annotations.get(x, "?"), "\t", sentence, "\t", condition, "\t", question)
#          assert len(annotations.get(x, "?")) == 3, annotations[x]
        if condition not in results:
           results[condition] = defaultdict(int)
        results[condition][annotations[x][0]]+=y
      print("....")
      print("numeric_noised_original", numeric_noised_original.size(), numeric_noised.size())
      for i in range(10):
            print(" ".join([itos_total[int(numeric_noised_original[j,i])] for j in range(numeric_noised.size()[0])]))
      print(sentence, "\t", condition)
      for x in (sorted(list(results.items()))):
        print(x[0], normalize_dict(x[1]))
      print("=============")
   #   break
#      quit()

def flatten(d):
   print(d)
   r = []
   for x in d:
     for y in x:
        r.append(y)
   return r
print("ARGUMENTS FROM TRAINING", checkpoint["arguments"]) 
with open(f"/u/scr/mhahn/noisy-channel-logs/deletion-gibson/{__file__}_{args.load_from_joint}", "w") as outFile:
      columnNames = sorted(list(set(flatten([list(y) for _, y in results.items()]))))
      HEADER = ["Condition"] + columnNames
      print("\t".join(HEADER), file=outFile)
      for x, _ in (sorted(list(results.items()))):
         outline = [x]
         for c in columnNames:
           outline.append(results[x].get(c,0))
         print("\t".join([str(z) for z in outline]), file=outFile)

