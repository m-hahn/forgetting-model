print("Character aware!")

# Built using char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars.py


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from-autoencoder", dest="load_from_autoencoder", type=str, default="264073608")
parser.add_argument("--load-from-lm", dest="load_from_lm", type=str, default=45661490)
#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([128]))
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


model = "REAL_REAL"

import math

args=parser.parse_args()

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
      self.zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
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
     print(numeric.size())
     embedded = self.word_embeddings(numeric.unsqueeze(0))
     results = ["" for _ in range(args.batchSize)]     
     for _ in range(10): 
        out, self.hidden = self.rnn(embedded, self.hidden)
        logits = self.output(out) 
        probs = self.softmax(logits)
        print(probs.size())
        dist = torch.distributions.Categorical(probs=probs)
         
        nextWord = (dist.sample())
        nextWordStrings = [itos_total[x] for x in nextWord.cpu().numpy()[0]]
        for i in range(args.batchSize):
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
       self.beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)
       input_tensor = Variable(numeric[:-1], requires_grad=False)
       target_tensor = Variable(numeric[1:], requires_grad=False)
       embedded = self.word_embeddings(input_tensor)
       if train:
          embedded = self.char_dropout(embedded)
          mask = self.bernoulli_input.sample()
          mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
          embedded = embedded * mask
  
       out, self.hidden = self.rnn(embedded, self.hidden)
   
       if train:
         mask = self.bernoulli_output.sample()
         mask = mask.view(1, args.batchSize, args.hidden_dim_lm)
         out = out * mask
   
       logits = self.output(out) 
       log_probs = self.logsoftmax(logits)
        
       loss = self.train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))
  
       if printHere:
          lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
          losses = lossTensor.data.cpu().numpy()
          numericCPU = numeric.cpu().data.numpy()
          print(("NONE", itos_total[numericCPU[0][0]]))
          for i in range((args.sequence_length)):
             print((losses[i][0], itos_total[numericCPU[i+1][0]]))
       samples = self.sample(numeric[-1])
       return loss, target_tensor.view(-1).size()[0], samples
    
from torch.autograd import Variable

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
      
      self.zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()
      
      
      self.zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim_autoencoder)).cuda()
      
      self.bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())
      
      self.bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
      self.bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * 2 * args.hidden_dim_autoencoder)]).cuda())
      
      
      


  def parameters(self):
      for module in self.modules:
          for param in module.parameters():
               yield param



  def forward(self, numeric, train=True, printHere=True):
      if True:
          self.beginning = self.zeroBeginning


      numeric_noised = [[x for x in y if random.random() > args.deletion_rate or itos_total[int(x)] == "."] for y in numeric.cpu().t()]

      numeric_noised = torch.LongTensor([[0 for _ in range(args.sequence_length-len(y))] + y for y in numeric_noised]).cuda().t()

      numeric = torch.cat([self.beginning, numeric], dim=0)
      numeric_noised = torch.cat([self.beginning, numeric_noised], dim=0)

      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = self.word_embeddings(input_tensor)

      embedded_noised = self.word_embeddings(input_tensor_noised)

      out_encoder, _ = self.rnn_encoder(embedded_noised, None)

      out_decoder, _ = self.rnn_decoder(embedded, None)

      attention = torch.bmm(self.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
      attention = self.attention_softmax(attention).transpose(0,1)
      from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
      out_full = torch.cat([out_decoder, from_encoder], dim=2)


      if train:
        mask = self.bernoulli_output.sample()
        mask = mask.view(1, args.batchSize, 2*args.hidden_dim_autoencoder)
        out_full = out_full * mask



      logits = self.output(self.relu(self.output_mlp(out_full) ))
      log_probs = self.logsoftmax(logits)

      
      loss = self.train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere:
         lossTensor = self.print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()
         numeric_noisedCPU = numeric_noised.cpu().data.numpy()

         print(("NONE", itos_total[numericCPU[0][0]]))
         for i in range((args.sequence_length)):
            print((losses[i][0], itos_total[numericCPU[i+1][0]], itos_total[numeric_noisedCPU[i+1][0]]))






      hidden = None
      result  = ["" for _ in range(args.batchSize)]
      result_numeric = [[] for _ in range(args.batchSize)]
      embeddedLast = embedded[0].unsqueeze(0)
      for i in range(args.sequence_length):
          out_decoder, hidden = self.rnn_decoder(embeddedLast, hidden)
    
          attention = torch.bmm(self.attention_proj(out_encoder).transpose(0,1), out_decoder.transpose(0,1).transpose(1,2))
          attention = self.attention_softmax(attention).transpose(0,1)
          from_encoder = (out_encoder.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
          out_full = torch.cat([out_decoder, from_encoder], dim=2)

          print(input_tensor.size())


          logits = self.output(self.relu(self.output_mlp(out_full) )) 
          probs = self.softmax(logits)

#          print(probs.size(), probs.sum(dim=2))
 #         quit()

          dist = torch.distributions.Categorical(probs=probs)
       
          nextWord = (dist.sample())
          print(nextWord.size())
          nextWordDistCPU = nextWord.cpu().numpy()[0]
          nextWordStrings = [itos_total[x] for x in nextWordDistCPU]
          for i in range(args.batchSize):
             result[i] += " "+nextWordStrings[i]
             result_numeric[i].append( nextWordDistCPU[i] )
          embeddedLast = self.word_embeddings(nextWord)
          print(embeddedLast.size())
      for r in result:
         print(r)
      print(float(len([x for x in result if NOUN in x]))/len(result))

      print(float(len([x for x in result if NOUN+" that" in x]))/len(result))

      print(float(len([x for x in result if NOUN2+" who" in x]))/len(result))

      return result, torch.LongTensor(result_numeric).cuda()


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



autoencoderFileName = "autoencoder2_mlp_bidir.py"

if args.load_from_autoencoder is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+autoencoderFileName+"_code_"+str(args.load_from_autoencoder)+".txt")
  for i in range(len(checkpoint["components"])):
      autoencoder.modules[i].load_state_dict(checkpoint["components"][i])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout









NOUN = "information"
NOUN2 = "janitor"
sentence = ", the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . the "+NOUN+" that the janitor who the doctor admired"
#sentence = ", the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . the "+NOUN+" which the janitor who the doctor admired"

#sentence = ", the nurse suggested to treat the patient with an antibiotic, but in the end , this did not happen . the city that the explorer who the inaccurate map had misled"
#sentence = " ".join(sentence.split(" ")[-args.sequence_length:])

numerified = [stoi[char]+3 if char in stoi else 2 for char in sentence.split(" ")]
print(len(numerified))
assert len(numerified) == args.sequence_length
numerified=torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
denoised, denoised_numeric = autoencoder.forward(numerified, train=False)
print(denoised_numeric.size())

lm.hidden = None
lm.beginning = None

_, _, samples = lm.forward(denoised_numeric.t(), train=False, printHere=True)

for i in range(args.batchSize):
   print(denoised[i]+" ### "+samples[i])
