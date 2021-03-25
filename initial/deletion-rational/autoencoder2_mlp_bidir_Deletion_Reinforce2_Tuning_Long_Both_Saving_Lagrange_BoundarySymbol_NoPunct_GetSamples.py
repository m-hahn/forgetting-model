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
parser.add_argument("--load-from-joint", dest="load_from_joint", type=str, default=random.choice([172216170]))
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
      return result, result_numeric, None, None

 
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

PUNCTUATION = torch.LongTensor([stoi_total[x] for x in [".", "OOV", '"', "(", ")", "'", '"', ":", ",", "'s", "[", "]"]]).cuda()

def forward(numeric, train=True, printHere=False, provideAttention=False, onlyProvideMemoryResult=False):
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

      punctuation = (((numeric.unsqueeze(0) == PUNCTUATION.view(12, 1, 1)).long().sum(dim=0)).bool())
        
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



if True:
      context = "later the nurse suggested they treat the patient with an antibiotic but in the end this did not happen <EOS> the girl kicked the ball <EOS> after this something else happened instead and she went away but nobody noticed anything about it"
      numerified = encodeContextCrop(context, "", replicates=24)
      assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
      # Run the noise model
      numberOfSamples = 24
      numeric, numeric_noised = forward((numerified, None), train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True)
      print(numeric.size(), numeric_noised.size(), numerified.size())

      numeric = numeric.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
      numeric_noised_original = numeric_noised
      numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
      # Get samples from the reconstruction posterior

      result, resultNumeric, fractions, thatProbs = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*24)
      resultNumeric = resultNumeric.transpose(0,1).contiguous().cpu()
      print(resultNumeric.size())
      sentences = defaultdict(int)
      for i in range(574):
            decoded = (" ".join([itos_total[int(resultNumeric[j,i])] for j in range(resultNumeric.size()[0])]))
            try:
              decoded = decoded[decoded.index("<EOS>")+6:]
            except ValueError:
                print("ERROR", decoded)
                continue
#            print(decoded)
            try:
              decoded = decoded[:decoded.index("<EOS>")]
            except ValueError:
                print("ERROR", decoded)
                continue
            sentences[decoded]+=1
#            print(decoded)
      sentences = sorted(list(sentences.items()), key=lambda x:x[1])
      for x, y in sentences:
          print(x, y)
      print("....")
      for i in range(10):
            print(" ".join([itos_total[int(numeric_noised_original[j,i])] for j in range(numeric_noised.size()[0])]))

