import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).cuda()
text = "Replace me by any text you'd like."
encoded_input = tokenizer.encode(text, return_tensors='pt')
print(encoded_input)
predictions, _ = model(encoded_input.cuda())
print(predictions.size())

sentences = {length : set() for length in range(20)}
counter = 0
with open("/jagupard20/scr0/mhahn/memory/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_CondGPT2.py_749792590_Model.txt", "r") as inFile:
   next(inFile)
   for line in inFile:
     counter += 1
     line = line.strip().split("\t")
     _, _, _, _, _, _, _, sentence, _, _, nextWord = line
     sentence = (sentence.strip().split(" ")[1:-1] + [nextWord.strip()])
     sentences[len(sentence)].add(" ".join(sentence).strip())
     if counter % 1000 == 0:
        print(counter/9537985, sum([len(x) for _, x in sentences.items()])/counter)
        break

for i in range(20):
    sents = list(sentences[i])
    for b in range(0, len(sents), 128):
       batch = sents[b:b+128]
       tensors = [tokenizer.encode(text, return_tensors='pt') for text in batch]
       maxLength = max([x.size()[1] for x in tensors])
       for i in range(len(tensors)):
          tensors[i] = torch.cat([tensors[i], torch.LongTensor([tokenizer.eos_token_id for _ in range(maxLength - tensors[i].size()[1])]).view(1, -1)], dim=1)
       tensors = torch.cat(tensors, dim=0)
       predictions, _ = model(tensors.cuda())
       print(predictions.size())      
       surprisals = torch.nn.CrossEntropyLoss(reduction='none')(predictions[:,:-1].contiguous().view(-1, 50257), tensors[:,1:].contiguous().view(-1).cuda()).view(len(batch), -1)
       print(surprisals, surprisals.size())
       for batchElem in range(len(batch)):
         print(tensors[batchElem])
         for q in range(maxLength-1):
            print(q, "#"+tokenizer.decode(int(tensors[batchElem][q]))+"#", surprisals[batchElem][q])
         # find where last word starts and separately get the surprisals
         quit()
   #         print([tokenizer.decode(int(x)) for x in tensors[batchElem]], surprisals[batchElem])
    #     quit()

