print("Character aware!")


# TODO can also do importance weighting using the language model -- maybe that makes it better


# Built using char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars.py


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default=878921872)
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=45661490)
parser.add_argument("--load-from-memory", dest="load_from_memory", type=str, default=968347198)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([16]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_autoencoder", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim_lm", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([30]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
#parser.add_argument("--char_emb_dim", type=int, default=128)
#parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
#parser.add_argument("--char_dec_hidden_dim", type=int, default=128)


parser.add_argument("--deletion_rate", type=float, default=0.2)

parser.add_argument("--surpsFile", type=str)


parser.add_argument("--SAMPLES_PER_BATCH", type=int,default=32)
parser.add_argument("--NUMBER_OF_RUNS", type=int, default=2)



model = "REAL_REAL"

import math

args=parser.parse_args()

#################################
#################################

#if "MYID" in args.save_to:
#   args.save_to = args.save_to.replace("MYID", str(args.myID))

#assert "word" in args.save_to, args.save_to

print(args)



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



class LanguageModel(torch.nn.Module):
  def __init__(self):
      super(LanguageModel, self).__init__()
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
      self.learning_rate = args.learning_rate
      self.optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.0) # 0.02, 0.9
      self.zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize*args.SAMPLES_PER_BATCH)]).cuda().view(1,args.batchSize*args.SAMPLES_PER_BATCH)
      self.beginning = None
      self.zeroBeginning_chars = torch.zeros(1, args.batchSize*args.SAMPLES_PER_BATCH, 16).long().cuda()
      self.zeroHidden = torch.zeros((args.layer_num, args.batchSize*args.SAMPLES_PER_BATCH, args.hidden_dim_lm)).cuda()
      self.bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize*args.SAMPLES_PER_BATCH)]).cuda())
      self.bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize*args.SAMPLES_PER_BATCH * 2 * args.word_embedding_size)]).cuda())
      self.bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize*args.SAMPLES_PER_BATCH * args.hidden_dim_lm)]).cuda())

      self.hidden = None

  def parameters(self):
     for module in self.modules:
         for param in module.parameters():
              yield param

  def sample(self, numeric):
  #   print(numeric.size())
     embedded = self.word_embeddings(numeric.unsqueeze(0))
     results = ["" for _ in range(args.batchSize*args.SAMPLES_PER_BATCH)]     
     for _ in range(10): 
        out, self.hidden = self.rnn(embedded, self.hidden)
        logits = self.output(out) 
        probs = self.softmax(logits)
        #print(probs.size())
        dist = torch.distributions.Categorical(probs=probs)
         
        nextWord = (dist.sample())

          


        nextWordStrings = [itos_total[x] for x in nextWord.cpu().numpy()[0]]
        for i in range(args.batchSize*args.SAMPLES_PER_BATCH):
            results[i] += " "+nextWordStrings[i]
        embedded = self.word_embeddings(nextWord)
     return results


  def forward(self, numeric, train=True, printHere=False):
       if self.hidden is None:
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
       numeric = torch.cat([self.beginning, numeric], dim=0)
       self.beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize*args.SAMPLES_PER_BATCH)
       input_tensor = Variable(numeric[:-1], requires_grad=False)
       target_tensor = Variable(numeric[1:], requires_grad=False)
       embedded = self.word_embeddings(input_tensor)
       if train:
          embedded = self.char_dropout(embedded)
          mask = self.bernoulli_input.sample()
          mask = mask.view(1, args.batchSize*args.SAMPLES_PER_BATCH, 2*args.word_embedding_size)
          embedded = embedded * mask
  
       out, self.hidden = self.rnn(embedded, self.hidden)
   
       if train:
         mask = self.bernoulli_output.sample()
         mask = mask.view(1, args.batchSize*args.SAMPLES_PER_BATCH, args.hidden_dim_lm)
         out = out * mask
   
       logits = self.output(out) 
       log_probs = self.logsoftmax(logits)
        
       loss = self.train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))
  
       if True or printHere:
          lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize*args.SAMPLES_PER_BATCH)
          losses = lossTensor.data.cpu()
          numericCPU = numeric.cpu().data
          print(("NONE", itos_total[numericCPU[0][0]]))
          for i in range(losses.size()[0]):
             print((float(losses[i][0]), itos_total[numericCPU[i+1][0]]))
       lastTokenSurprisal = losses[-1, :]
       return lastTokenSurprisal
    
from torch.autograd import Variable


memory_mlp_inner = torch.nn.Linear(2*args.word_embedding_size, 500).cuda()
memory_mlp_outer = torch.nn.Linear(500, 1).cuda()

sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()

modules_memory = [memory_mlp_inner, memory_mlp_outer]




class Autoencoder(torch.nn.Module):
  def __init__(self):
      super(Autoencoder, self).__init__()
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
      
      self.train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
      
      self.attention_proj = torch.nn.Linear(args.hidden_dim_autoencoder, args.hidden_dim_autoencoder, bias=False).cuda()
      self.attention_proj.weight.data.fill_(0)
      
      self.output_mlp = torch.nn.Linear(2*args.hidden_dim_autoencoder, args.hidden_dim_autoencoder).cuda()
      
      self.modules = [self.rnn_decoder, self.rnn_encoder, self.output, self.word_embeddings, self.attention_proj, self.output_mlp]
      
      self.learning_rate = args.learning_rate

      self.optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.0) # 0.02, 0.9
      self.relu = torch.nn.ReLU()
      
      self.hidden = None
      
      self.zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
      self.beginning = None
      
      self.zeroBeginning_chars = torch.zeros(1, args.batchSize*args.SAMPLES_PER_BATCH, 16).long().cuda()
      
      
      self.zeroHidden = torch.zeros((args.layer_num, args.batchSize*args.SAMPLES_PER_BATCH, args.hidden_dim_autoencoder)).cuda()
      
      self.bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize*args.SAMPLES_PER_BATCH)]).cuda())
      
      self.bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize*args.SAMPLES_PER_BATCH * 2 * args.word_embedding_size)]).cuda())
      self.bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize*args.SAMPLES_PER_BATCH * 2 * args.hidden_dim_autoencoder)]).cuda())
      
      
      


  def parameters(self):
      for module in self.modules:
          for param in module.parameters():
               yield param



  def forward(self, numeric, train=True, printHere=True):
      if True:
          self.beginning = self.zeroBeginning

      
      numeric = torch.cat([self.beginning, numeric], dim=0)

      embedded_everything = self.word_embeddings(numeric)
      memory_hidden = sigmoid(memory_mlp_outer(relu(memory_mlp_inner(embedded_everything))))
      memory_filter = torch.bernoulli(input=memory_hidden)
      memory_filter = memory_filter.squeeze(2)
      memory_filter[numeric == stoi_total["."]] = 1
      numeric_noised = torch.where(memory_filter==1, numeric, 0*numeric) #[[x if random.random() > args.deletion_rate else 0 for x in y] for y in numeric.cpu().t()]
      numeric_onlyNoisedOnes = torch.where(memory_filter == 0, numeric, 0*numeric) # target is 0 in those places where no noise has happened

      numeric = numeric.unsqueeze(2).expand(-1, -1, args.SAMPLES_PER_BATCH).contiguous().view(-1, args.batchSize*args.SAMPLES_PER_BATCH)
      numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, args.SAMPLES_PER_BATCH).contiguous().view(-1, args.batchSize*args.SAMPLES_PER_BATCH)
      numeric_onlyNoisedOnes = numeric_onlyNoisedOnes.unsqueeze(2).expand(-1, -1, args.SAMPLES_PER_BATCH).contiguous().view(-1, args.batchSize*args.SAMPLES_PER_BATCH)

      input_tensor = Variable(numeric[:-1], requires_grad=False)
      input_tensor_noised = Variable(numeric_noised[:-1], requires_grad=False)
      target_tensor = Variable(numeric_onlyNoisedOnes[1:], requires_grad=False)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = self.word_embeddings(input_tensor)

      embedded_noised = self.word_embeddings(input_tensor_noised)

      out_encoder, _ = self.rnn_encoder(embedded_noised, None)



      hidden = None
      result  = ["" for _ in range(args.batchSize*args.SAMPLES_PER_BATCH)]
      result_numeric = [[] for _ in range(args.batchSize*args.SAMPLES_PER_BATCH)]
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

#          print(probs.size(), probs.sum(dim=2))
 #         quit()

          dist = torch.distributions.Categorical(probs=probs)
       
#          nextWord = (dist.sample())
          nextWord = torch.where(numeric_noised[i] == 0, (dist.sample()), numeric[i:i+1])
  #        print(nextWord.size())
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for i in range(args.batchSize*args.SAMPLES_PER_BATCH):
             result[i] += " "+nextWordStrings[i]
             result_numeric[i].append( nextWordDistCPU[i] )
          embeddedLast = self.word_embeddings(nextWord)
#          print(embeddedLast.size())
      for r in result[:10]:
         print(r)
      nounFraction = (float(len([x for x in result if NOUN in x]))/len(result))

      thatFraction = (float(len([x for x in result if NOUN+" that" in x]))/len(result))

      return result, torch.LongTensor(result_numeric).cuda(), (nounFraction, thatFraction)


autoencoder = Autoencoder()
lm = LanguageModel()



#named_modules = {"rnn" : rnn, "output" : output, "word_embeddings" : word_embeddings, "optim" : optim}

#if args.load_from is not None:
#  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
#  for name, module in named_modules.items():
 #     module.load_state_dict(checkpoint[name])


lmFileName = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars.py"

if args.load_from_lm is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+lmFileName+"_code_"+str(args.load_from_lm)+".txt")
  for i in range(len(checkpoint["components"])):
      lm.modules[i].load_state_dict(checkpoint["components"][i])



autoencoderFileName = "autoencoder2_mlp_bidir_Erasure_SelectiveLoss.py"

if args.load_from_autoencoder is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+autoencoderFileName+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      autoencoder.modules[i].load_state_dict(checkpoint["components"][i])

memoryFileName = "autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Saving.py"

if args.load_from_memory is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS_memoryPolicy/"+args.language+"_"+memoryFileName+"_code_"+str(args.load_from_memory)+".txt")
  for i in range(len(checkpoint["components"])):
      modules_memory[i].load_state_dict(checkpoint["components"][i])




from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout








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

nounsAndVerbs.append(["the senator",        "the diplomat",       "opposed",                   "won",                   "shocked people"])

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



with open("../forgetting/fromCorpus_counts.csv", "r") as inFile:
   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = counts[0]
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[0] : line[1:] for line in counts}

topNouns = [x for x in topNouns if x in counts]
topNouns = sorted(list(set(topNouns)), key=lambda x:float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]]))

print(topNouns)
print(len(topNouns))

#quit()


results = []
with torch.no_grad():
  with open("temporary-stimuli/stimuli2.txt", "w") as outFile:

                                                                                                                                                                                                           


# ~/python-py37-mhahn autoencoder2_mlp_bidir_Erasure_constructStimulusSentences_RUN_languagemodel2.py --batchSize=800 --SAMPLES_PER_BATCH=1 --NUMBER_OF_RUNS=1 --surpsFile=surpsErasure1.txt                                                             
# IMPORTANT SANITY CHECK
# ~/python-py37-mhahn autoencoder2_mlp_bidir_Erasure_constructStimulusSentences_RUN_languagemodel2.py --batchSize=1 --SAMPLES_PER_BATCH=1 --NUMBER_OF_RUNS=1 --surpsFile=surpsErasure1b_00.txt --load-from-autoencoder=878921872 --deletion_rate=0.0

# ~/python-py37-mhahn autoencoder2_mlp_bidir_Erasure_constructStimulusSentences_RUN_languagemodel2.py --batchSize=800 --SAMPLES_PER_BATCH=1 --NUMBER_OF_RUNS=1 --surpsFile=surpsErasure1b.txt --load-from-autoencoder=878921872
# ~/python-py37-mhahn autoencoder2_mlp_bidir_Erasure_constructStimulusSentences_RUN_languagemodel2.py --batchSize=800 --SAMPLES_PER_BATCH=1 --NUMBER_OF_RUNS=1 --surpsFile=surpsErasure1c.txt --load-from-autoencoder=49986904
# ~/python-py37-mhahn autoencoder2_mlp_bidir_Erasure_constructStimulusSentences_RUN_languagemodel2.py --batchSize=800 --SAMPLES_PER_BATCH=1 --NUMBER_OF_RUNS=1 --surpsFile=surpsErasure1d.txt --load-from-autoencoder=69900543


   with open(f"temporary-stimuli/{args.surpsFile}", "w") as outFileSurps:
    for NOUN in topNouns:
#     NOUN = "belief"
     for sentenceList in nounsAndVerbs:
       print(sentenceList)
       context = ", the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . "
       for condition in [0,2]:
          if condition == 0:
             sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} {sentenceList[3]} {sentenceList[4]}"
          elif condition == 2:
             sentence = context + f"the {NOUN} that {sentenceList[0]} who {sentenceList[1]} {sentenceList[2]} {sentenceList[4]}"
          else:
             assert False
     #     print(sentence)
    #      continue
          numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
          print(len(numerified))
          numerified = numerified[-args.sequence_length:]
          assert len(numerified) == args.sequence_length
          numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
          print(" ".join([itos[int(x)-3] for x in numerified[:,0]]))
          print("===========")
          surprisalsPerRun = []
          thatFractions = []
          for RUN in range(args.NUMBER_OF_RUNS):
             result, resultNumeric, fractions = autoencoder.forward(numerified, train=False)
             (nounFraction, thatFraction) = fractions
             thatFractions.append(thatFraction)


             lm.hidden = None
             lm.beginning = None
           
             resultNumeric = torch.cat([resultNumeric, torch.LongTensor([stoi_total["."] for _ in range(args.batchSize*args.SAMPLES_PER_BATCH)]).cuda().view(-1, 1)], dim=1)
             print(resultNumeric.size())
      #       quit()
       #      quit()
             print(NOUN,  topNouns.index(NOUN), RUN)
             lastTokenSurprisal = lm.forward(resultNumeric.t(), train=False, printHere=True)
             lastTokenSurprisal = lastTokenSurprisal.view(-1, args.SAMPLES_PER_BATCH)
             #print(lastTokenSurprisal.mean(dim=1))
             #print(lastTokenSurprisal.view(args.SAMPLES_PER_BATCH, -1).mean(dim=0))

             #quit()
             probabilityOfEOS = torch.exp(-lastTokenSurprisal)
             #print(probabilityOfEOS)
             averageProbabilityOfEOS = probabilityOfEOS.mean(dim=1)
             #print(averageProbabilityOfEOS)
             surprisalPerBatch = -torch.log(averageProbabilityOfEOS)
             #print(surprisalPerBatch)
             surprisalsPerRun.append(surprisalPerBatch)
             for index, imputed in enumerate(result):
                print(f'{NOUN}\t{sentenceList[0].replace(" ","_")}\t{int(index / args.SAMPLES_PER_BATCH)}\t{condition}\t{imputed}', file=outFile)

   #       print(lastTokenSurprisal.view(args.SAMPLES_PER_BATCH, -1))
    #      quit()
          #for i in range(args.batchSize*args.SAMPLES_PER_BATCH):
          #   print(denoised[i]+" ### "+samples[i])
  #        results.append((NOUN, sentenceList[0], condition, result))
          print(surprisalsPerRun)
          surprisalsPerRun = torch.stack(surprisalsPerRun, dim=0)
          meanSurprisal = surprisalsPerRun.mean()
          varianceSurprisal = (surprisalsPerRun.pow(2)).mean() - (meanSurprisal**2)
          print(f'{NOUN}\t{sentenceList[0].replace(" ","_")}\t{condition}\t{meanSurprisal}\t{varianceSurprisal/math.sqrt(args.NUMBER_OF_RUNS*args.batchSize)}\t{sum(thatFractions)/len(thatFractions)}', file=outFileSurps)
          print("NOUNS SO FAR", topNouns.index(NOUN))

import sys
print(sys.argv)
