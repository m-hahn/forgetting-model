from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition="train", removeMarkup=True):
  path = WIKIPEDIA_HOME+"/english-"+partition+"-tagged.txt"
  chunk = []
  with open(path, "r") as inFile:
    for line in inFile:
      index = line.find("\t")
      if index == -1:
        if removeMarkup:
          continue
        else:
          index = len(line)-1
      word = line[:index]
      chunk.append(word.lower())
      if len(chunk) > 1000000:
      #   random.shuffle(chunk)
         yield chunk
         chunk = []
  yield chunk

def training(language):
  return load(language, "train")

def dev(language, removeMarkup=True):
  return load(language, "valid", removeMarkup=removeMarkup)

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)


