# Based on finetune_xlnet_sst2_10_c_sentBreak_new_large_sample.py


# In part based on ~/scr/CODE/transformers/examples/run_glue.py

#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


# Based on https://raw.githubusercontent.com/huggingface/transformers/master/examples/run_generation.py

# Construct neighbors for SST-2

# ~/python-py37-mhahn generate_RoBERTa.py

import argparse
import logging

import numpy as np
import torch

from transformers import (
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos> """


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text



PREPROCESSING_FUNCTIONS = {
    "xlnet": prepare_xlnet_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length



import re
import random
import argparse
import glob
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


#import dataloader


from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

model = AutoModelWithLMHead.from_pretrained("roberta-large").cuda()
			
from transformers import pipeline

#unmasker = pipeline('fill-mask', model='bert-base-uncased')

#print(unmasker("hello i'm a <mask> model."))
assert tokenizer.mask_token == "<mask>"

sequences = []
sequences.append(f"Distilled {tokenizer.mask_token} are smaller than the models they mimic. Using them instead of the large versions would help lower our carbon footprint.")
sequences.append(f"This is a massive blockbuster crunchalic {tokenizer.mask_token} movie.")

inputs = [tokenizer.encode(x, return_tensors="pt")[0] for x in sequences]
maxLength = max([x.size()[0] for x in inputs])
for i in range(len(inputs)):
    inputs[i] = torch.cat([inputs[i], torch.LongTensor([tokenizer.convert_tokens_to_ids("<pad>") for _ in range(maxLength - inputs[i].size()[0])])])
    print([tokenizer.convert_ids_to_tokens(int(x)) for x in inputs[i]])




input = torch.stack(inputs, dim=0)
token_logits = model(input.cuda())
print(token_logits)
token_logits = token_logits[0]
print(token_logits)
print(token_logits.size())

import torch
print(torch.cuda.is_available())

mask_token_index = (input == tokenizer.mask_token_id)
print(mask_token_index)
print([[bool(y) for y in x] for x in mask_token_index])
print([True in [bool(y) for y in x] for x in mask_token_index])


def trueIndex(x):
   print(x)
   print(True in x)
   return (x.index(True))
mask_token_index = [trueIndex([bool(y) for y in x]) for x in mask_token_index]
print(mask_token_index)

mask_token_logits = token_logits[0, mask_token_index[0], :]
mask_token_logits = torch.stack([token_logits[i, mask_token_index[i], :] for i in range(len(mask_token_index))], dim=0)
print(mask_token_logits)


import torch.nn.functional as F

for i in range(2):
  top_5_tokens = torch.topk(mask_token_logits[i], 5, dim=0).indices.tolist()
  for token in top_5_tokens:
    print(sequences[i].replace(tokenizer.mask_token, tokenizer.decode([token])))

                                                                                                                   
probs = F.softmax(mask_token_logits, dim=-1).squeeze(1)      

for _ in range(10):                                                                                                                          
 next_token = torch.multinomial(probs, num_samples=1).squeeze(1)     
 print(next_token.size())
 for i in next_token:
    print(tokenizer.decode([int(i)]))


import re

#quit()



GENERATION_VOCABULARY_MASK = torch.cuda.FloatTensor([float("-inf") if tokenizer.convert_ids_to_tokens(x) in ["<unk>", "@"] else 0 for x in range(50265)]).view(1, 1, -1)

blankCandidates = []

from transformers import AdamW
LR = 5e-06#random.choice([5e-6, 1e-5, 2e-5, 5e-5, 1e-4])

optimizer = AdamW(model.parameters(),
                  lr = LR, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )


from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The RoBERTa authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 3 # random.choice([1,2,3,4])

BATCH = 4 #random.choice([4, 8, 16])

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).

model.eval()


def sampleNextWord(sentences, contractSampleDimension):
      inputs = [tokenizer.encode(f"{x} {tokenizer.mask_token} {tokenizer.mask_token} {tokenizer.mask_token} {tokenizer.mask_token}", return_tensors="pt")[0] for x in sentences]
      maxLength = max([int(x.size()[0]) for x in inputs])
      for i in range(len(inputs)):
       padding = " ".join([tokenizer.pad_token] * (maxLength - inputs[i].size()[0]))
       if len(padding) == 0:
          continue
       padding = tokenizer.encode(padding, return_tensors="pt")[0][1:-1]
       inputs[i] = torch.cat([inputs[i], padding], dim=0)
      input = torch.stack(inputs, dim=0)[:, :-1] # want to get rid of EOS symbol
      token_logits = model(input.cuda())
      token_logits = token_logits[0]
      
      mask_token_index = (input == tokenizer.mask_token_id)
     
      def trueIndex(x):
         return (x.index(True))
      mask_token_index = [trueIndex([bool(y) for y in x]) for x in mask_token_index]
      
      mask_token_logits = token_logits[0, mask_token_index[0], :]
      mask_token_logits = torch.stack([token_logits[i, mask_token_index[i], :] for i in range(len(mask_token_index))], dim=0)
      
      
      import torch.nn.functional as F
      
                                                                                                                        
      probs = F.softmax(GENERATION_VOCABULARY_MASK + mask_token_logits, dim=-1).squeeze(0)
      if contractSampleDimension != None:
         probs = probs.view(contractSampleDimension + (-1,)).mean(dim=1)
      else:
          contractSampleDimension = (len(inputs),)
      for _ in range(1):
       next_token = torch.multinomial(probs, num_samples=1).squeeze(1).cpu()
       encoded = []
       for i in range(contractSampleDimension[0]):
          encoded.append(tokenizer.decode(int(next_token[i])))
      return encoded 
