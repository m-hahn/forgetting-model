
def getColumn(data, x):
    return [y[x] for y in data]

def readTSV(path):
   data = [x.split("\t") for x in open(path).read().strip().split("\n")]
   header = data[0]
   data = [dict(zip(header, x)) for x in data[1:]]
   for x in header:
      column = getColumn(data, x)
   return data
   

def names(table):
   return sorted(list(table[0]))


#
#model_plain = readTSV("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_ZERO_EYE-T.py_495510756_ZeroLoss")
#
#corpus = readTSV("/u/scr/mhahn/Dundee/DundeeMerged.csv")
#
#matching = readTSV("matchData_EYE.py.tsv")
#tokenized = readTSV("/u/scr/mhahn/Dundee/DundeeTreebankTokenized2.csv")
#
#
model_id=899225807

from collections import defaultdict
def mean(x):
  return sum(x)/len(x)
def averageAcrossRepetitions(table):
    values = defaultdict(list)
    for line in table:
       line_ = tuple(sorted(((x,y) for x,y in line.items() if x not in ["Surprisal", "SurprisalReweighted", "Repetition"])))
       values[line_].append(float(line["SurprisalReweighted"]))
    return sorted([dict(x + (("SurprisalReweighted", mean(values[x])),)) for x in values], key=lambda x:[int(q) for q in (x["Sentence"], x["Region"], x["Itemno"], x["WNUM"])])
       

model = averageAcrossRepetitions(readTSV("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_EYE2-T.py_" + str(model_id) + "_Model"))



print(model[:10])


#, sep=""), sep="\t") %>% group_by(Sentence, Region, TokenLower, Itemno, WNUM, SentenceID, ID, WORD, tokenInWord) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted))
# 






