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
# https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py



# Construct neighbors for SST-2

# ~/python-py37-mhahn generate_RoBERTa.py

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


import argparse
import logging

import numpy as np
import torch

from transformers import (
#    CTRLLMHeadModel,
 #   CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
#    TransfoXLLMHeadModel,
 #   TransfoXLTokenizer,
  #  XLMTokenizer,
   # XLMWithLMHeadModel,
    #XLNetLMHeadModel,
    #XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
#    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
 #   "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
  #  "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
   # "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


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


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

class Args(object):
    condition = 'New'
    def __init__(self):
        pass
if True:
    args = Args()
    args.no_cuda=False
    args.model_type = "gpt2"
    args.model_name_or_path = "gpt2-large"

    args.prompt="The notion that the diplomat who"
    args.length=50
    args.stop_token=None
    args.temperature=1.0
    args.repetition_penalty=1.0
    args.k=0
    args.p=0.9
    args.prefix=""
    args.padding_text=""
    args.xlm_language=""
    args.seed=42
    args.no_cuda=False
    args.num_return_sequences=6
    args.fp16=False

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

def prepareModel():
    global tokenizer
    global model
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir="/u/scr/mhahn/CACHE/transformers/")
    model = model_class.from_pretrained(args.model_name_or_path, cache_dir="/u/scr/mhahn/CACHE/transformers/")
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)




def sample(prompt):
    global tokenizer
    global model
    prompt_text = prompt

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
        preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

        if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            tokenizer_kwargs = {"add_space_before_punct_symbol": True}
        else:
            tokenizer_kwargs = {}

        encoded_prompt = tokenizer.encode(
            preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
        )
    else:
        prefix = args.prefix if args.prefix else args.padding_text
        encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
 #       print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
#        print(total_sequence)

    return generated_sequences


#if __name__ == "__main__":
prepareModel()

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
nounsAndVerbs.append(["the victims",         "the criminal",       "assaulted",  "were surviving",         "calmed everyone down", "Did the XXXX calm everyone down?", "Y"])
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



with open("../../../../forgetting/fromCorpus_counts.csv", "r") as inFile:
   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = counts[0]
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[0] : line[1:] for line in counts}

topNouns = [x for x in topNouns if x in counts]
topNouns = sorted(list(set(topNouns)), key=lambda x:float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]]))

print(topNouns)
print(len(topNouns))

with open("/u/scr/mhahn/reinforce-logs-both/gpt2-pure-completion/"+__file__+".txt", "w") as outFile:
 for noun in topNouns[:15]+topNouns[-15:]:
   print(noun)
   for cont in nounsAndVerbs:
     print(noun, cont)
     print("\n".join([z[:z.index("\n")] if "\n" in z else z for z in sample("The "+noun+" that "+cont[0]+" who "+cont[1])]), file=outFile)
# nounsAndVerbs.append(["the senator",         "the diplomat",       "opposed",    "was winning",                   "really made him angry", "Did the XXXX make him angry?", "Y"])

