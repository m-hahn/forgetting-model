from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition="train", removeMarkup=True):
  if language == "italian":
    path = WIKIPEDIA_HOME+"/itwiki-"+partition+"-tagged.txt"
  elif language == "english":
    path = WIKIPEDIA_HOME+"/english-"+partition+"-tagged.txt"
  elif language == "german":
    path = WIKIPEDIA_HOME+""+language+"-"+partition+"-tagged.txt"
  else:
    path = WIKIPEDIA_HOME+"/WIKIPEDIA/"+language+"/"+language+"-"+partition+"-tagged.txt"
  chunk = []
  if language != "japanese":
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
  else:
     import gzip
     with gzip.open(path+".gz", "rb") as inFile:
      for line in inFile:
       line = line.decode("utf-8")
       for t in line.strip().replace("\ ", "").split(" "):
         if len(t) <= 1:
            continue
         try:
            word = t[:t.index("/")].lower()
         except ValueError:
           print(t)
           continue
#         print(t, word)
         if len(word) == 0:
            continue
         chunk.append(word)
         if len(chunk) > 1000000:
            yield chunk
            chunk = []
  yield chunk

def training(language):
  return load(language, "train")
#   with open(WIKIPEDIA_HOME+""+language+"-train.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
def dev(language, removeMarkup=True):
  return load(language, "valid", removeMarkup=removeMarkup)

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)

#   with open(WIKIPEDIA_HOME+""+language+"-valid.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
#

#     for line in data:
#        yield line


