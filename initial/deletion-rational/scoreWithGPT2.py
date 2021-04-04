import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).cuda()
print("Finished loading GPT2")

#text = "Replace me by any text you'd like."
#encoded_input = tokenizer.encode(text, return_tensors='pt')
#print(encoded_input)
#predictions, _ = model(encoded_input.cuda())
#print(predictions.size())
#
#sentences = {length : set() for length in range(20)}
#counter = 0
#with open("/jagupard27/scr0/mhahn/memory/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_CondGPT2.py_749792590_Model.txt", "r") as inFile:
#   next(inFile)
#   for line in inFile:
#     counter += 1
#     line = line.strip().split("\t")
#     _, _, _, _, _, _, _, sentence, _, _, nextWord = line
#     sentence = (sentence.strip().split(" ")[1:-1] + [nextWord.strip()])
#     sentences[len(sentence)].add(" ".join(sentence).strip())
#     if counter % 1000 == 0:
#        print(counter/9537985, sum([len(x) for _, x in sentences.items()])/counter)
#     #   break
#
#GPT2surprisals = {}

#with open("/jagupard27/scr0/mhahn/memory/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_CondGPT2.py_749792590_Model.txt_SURP", "w") as outFile:
#  for w in range(20):
def scoreSentences(batch_):
       fromStringToPosition = {}
       batch = []
       for s in batch_:
          if s not in fromStringToPosition:
              fromStringToPosition[s] = len(batch)
              batch.append(s)

       tensors = [tokenizer.encode(" "+text.strip()+".", return_tensors='pt') for text in batch] # below using bos, so should be no need for adding "<|endoftext|> "+
   #    print(tensors)
#       print(tokenizer.decode(50256))
 #      print(tokenizer.decode(262))
#       print(tokenizer.bos_token_id, tokenizer.eos_token_id) they all evaoluate to 50256, the ID of <|endoftext|>
  #     quit()
       #print([(x.size()) for x in tensors])
       maxLength = max([x.size()[1] for x in tensors])+1
       print("MAX LENGTH", maxLength)
       for i in range(len(tensors)):
          tensors[i] = torch.cat([torch.LongTensor([tokenizer.bos_token_id]).view(1,1), tensors[i], torch.LongTensor([tokenizer.eos_token_id for _ in range(maxLength - tensors[i].size()[1])]).view(1, -1)], dim=1)
       tensors = torch.cat(tensors, dim=0)
       predictions, _ = model(tensors.cuda())
#       print(predictions.size())      
       surprisals = torch.nn.CrossEntropyLoss(reduction='none')(predictions[:,:-1].contiguous().view(-1, 50257), tensors[:,1:].contiguous().view(-1).cuda()).view(len(batch), -1)
       surprisals = surprisals.detach().cpu()
 #      print(surprisals, surprisals.size())
       surprisalsCollected = []
       for batchElem in range(len(batch)):
         #print(tensors[batchElem])
         words = [[]]
         for q in range(1, maxLength):
            word = tokenizer.decode(int(tensors[batchElem][q]))
         #   print(word)
            if word.startswith(" ") or q == 0:
                words.append([])
            words[-1].append((word, float(surprisals[batchElem][q-1])))
            if word == '<|endoftext|>':
                break
        #    print(q, "#"+word+"#", surprisals[batchElem][q-1])
#         print(words, batch[batchElem])
 #        quit()
         # find where last word starts and separately get the surprisals
#         print(words)
         surprisalsPast = sum([sum(x[1] for x in y) for y in words])
         surprisalsCollected.append({"past" : surprisalsPast})
#       quit()
       #print(fromStringToPosition)
       surprisalsCollected_ = []
       for s in batch_:
          surprisalsCollected_.append(surprisalsCollected[fromStringToPosition[s]])
       return surprisalsCollected_
#         print("\t".join([batch[batchElem], str( sum([sum(x[1] for x in y) for y in words[:-1]])), str(sum(x[1] for x in words[-1]))]))
 #      quit()  
#         GPT2surprisals[batch[batchElem]] = ( sum([sum(x[1] for x in y) for y in words[:-1]]), sum(x[1] for x in words[-1]))
#       quit()
   #         print([tokenizer.decode(int(x)) for x in tensors[batchElem]], surprisals[batchElem])
    #     quit()
