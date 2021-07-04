# Based on:
#  char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos12_Long.py (loss model & code for language model)
# And autoencoder2_mlp_bidir_Erasure_SelectiveLoss_Reinforce2_Tuning_SuperLong_Both_Saving.py (autoencoder)
# And (for the plain LM): ../autoencoder/autoencoder2_mlp_bidir_AND_languagemodel_sample.py
print("Character aware!")
import os
# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys
import random
from collections import defaultdict
import argparse



nounsAndVerbsIncompatible = []
nounsAndVerbsCompatible = []

nounsAndVerbsCompatible.append(['the teacher', 'the principal', 'liked', 'annoyed the student', 'was'])
nounsAndVerbsIncompatible.append(['the teacher', 'the principal', 'liked', 'failed the student', 'was'])
nounsAndVerbsCompatible.append(['the doctor', 'the colleague', 'distrusted', 'bothered the patients', 'seemed'])
nounsAndVerbsIncompatible.append(['the doctor', 'the colleague', 'distrusted', 'cured the patients', 'seemed'])
nounsAndVerbsCompatible.append(['the bully', 'the children', 'hated', 'shocked the boy', 'was'])
nounsAndVerbsIncompatible.append(['the bully', 'the children', 'hated', 'harassed the boy', 'was'])
nounsAndVerbsCompatible.append(['the agent', 'the fbi', 'sent', 'confused the criminal', 'was'])
nounsAndVerbsIncompatible.append(['the agent', 'the fbi', 'sent', 'arrested the criminal', 'was'])
nounsAndVerbsCompatible.append(['the senator', 'the diplomat', 'supported', 'troubled the opponent', 'deserved'])
nounsAndVerbsIncompatible.append(['the senator', 'the diplomat', 'supported', 'defeated the opponent', 'deserved'])
nounsAndVerbsCompatible.append(['the fiancé', 'the author', 'met', 'startled the bride', 'did'])
nounsAndVerbsIncompatible.append(['the fiancé', 'the author', 'met', 'married the bride', 'did'])
nounsAndVerbsCompatible.append(['the businessman', 'the sponsor', 'backed', 'surprised the employee', 'came'])
nounsAndVerbsIncompatible.append(['the businessman', 'the sponsor', 'backed', 'fired the employee', 'came'])
nounsAndVerbsCompatible.append(['the thief', 'the detective', 'caught', 'enraged the woman', 'broke'])
nounsAndVerbsIncompatible.append(['the thief', 'the detective', 'caught', 'robbed the woman', 'broke'])
nounsAndVerbsCompatible.append(['the criminal', 'the stranger', 'distracted', 'startled the officer', 'seemed'])
nounsAndVerbsIncompatible.append(['the criminal', 'the stranger', 'distracted', 'abducted the officer', 'seemed'])
nounsAndVerbsCompatible.append(['the customer', 'the vendor', 'welcomed', 'terrified the clerk', 'was'])
nounsAndVerbsIncompatible.append(['the customer', 'the vendor', 'welcomed', 'contacted the clerk', 'was'])
nounsAndVerbsCompatible.append(['the president', 'the farmer', 'admired', 'impressed the commander', 'took'])
nounsAndVerbsIncompatible.append(['the president', 'the farmer', 'admired', 'appointed the commander', 'took'])
nounsAndVerbsCompatible.append(['the victim', 'the swimmer', 'rescued', 'surprised the criminal', 'appeared'])
nounsAndVerbsIncompatible.append(['the victim', 'the swimmer', 'rescued', 'sued the criminal', 'appeared'])
nounsAndVerbsCompatible.append(['the guest', 'the cousin', 'invited', 'pleased the uncle', 'drove'])
nounsAndVerbsIncompatible.append(['the guest', 'the cousin', 'invited', 'visited the uncle', 'drove'])
nounsAndVerbsCompatible.append(['the psychiatrist', 'the nurse', 'assisted', 'horrified the patient', 'became'])
nounsAndVerbsIncompatible.append(['the psychiatrist', 'the nurse', 'assisted', 'diagnosed the patient', 'became'])
nounsAndVerbsCompatible.append(['the driver', 'the guide', 'called', 'amazed the tourist', 'was'])
nounsAndVerbsIncompatible.append(['the driver', 'the guide', 'called', 'phoned the tourist', 'was'])
nounsAndVerbsCompatible.append(['the actor', 'the fans', 'loved', 'astonished the director', 'appeared'])
nounsAndVerbsIncompatible.append(['the actor', 'the fans', 'loved', 'greeted the director', 'appeared'])
nounsAndVerbsCompatible.append(['the banker', 'the analyst', 'bribed', 'excited the customer', 'proved'])
nounsAndVerbsIncompatible.append(['the banker', 'the analyst', 'bribed', 'trusted the customer', 'proved'])
nounsAndVerbsCompatible.append(['the judge', 'the attorney', 'hated', 'vindicated the defendant', 'was'])
nounsAndVerbsIncompatible.append(['the judge', 'the attorney', 'hated', 'acquitted the defendant', 'was'])
nounsAndVerbsCompatible.append(['the captain', 'the crew', 'trusted', 'motivated the sailor', 'was'])
nounsAndVerbsIncompatible.append(['the captain', 'the crew', 'trusted', 'commanded the sailor', 'was'])
nounsAndVerbsCompatible.append(['the manager', 'the boss', 'authorized', 'saddened the intern', 'seemed'])
nounsAndVerbsIncompatible.append(['the manager', 'the boss', 'authorized', 'hired the intern', 'seemed'])
nounsAndVerbsCompatible.append(['the plaintiff', 'the jury', 'interrogated', 'startled the witness', 'made'])
nounsAndVerbsIncompatible.append(['the plaintiff', 'the jury', 'interrogated', 'interrupted the witness', 'made'])
nounsAndVerbsCompatible.append(['the drinker', 'the thug', 'hit', 'stunned the bartender', 'sounded'])
nounsAndVerbsIncompatible.append(['the drinker', 'the thug', 'hit', 'tricked the bartender', 'sounded'])
nounsAndVerbsCompatible.append(['the pediatrician', 'the receptionist', 'supported', 'disturbed the parent', 'had'])
nounsAndVerbsIncompatible.append(['the pediatrician', 'the receptionist', 'supported', 'distrusted the parent', 'had'])
nounsAndVerbsCompatible.append(['the medic', 'the survivor', 'thanked', 'surprised the surgeon', 'turned'])
nounsAndVerbsIncompatible.append(['the medic', 'the survivor', 'thanked', 'greeted the surgeon', 'turned'])
nounsAndVerbsCompatible.append(['the lifeguard', 'the soldier', 'taught', 'encouraged the swimmer', 'took'])
nounsAndVerbsIncompatible.append(['the lifeguard', 'the soldier', 'taught', 'rescued the swimmer', 'took'])
nounsAndVerbsCompatible.append(['the fisherman', 'the gardener', 'helped', 'delighted the politician', 'was'])
nounsAndVerbsIncompatible.append(['the fisherman', 'the gardener', 'helped', 'admired the politician', 'was'])
nounsAndVerbsCompatible.append(['the janitor', 'the organizer', 'criticized', 'amused the audience', 'was'])
nounsAndVerbsIncompatible.append(['the janitor', 'the organizer', 'criticized', 'ignored the audience', 'was'])
nounsAndVerbsCompatible.append(['the investor', 'the scientist', 'detested', 'disappointed the entrepreneur', 'drove'])
nounsAndVerbsIncompatible.append(['the investor', 'the scientist', 'detested', 'deceived the entrepreneur', 'drove'])
nounsAndVerbsCompatible.append(['the firefighter', 'the neighbor', 'insulted', 'discouraged the houseowner', 'went'])
nounsAndVerbsIncompatible.append(['the firefighter', 'the neighbor', 'insulted', 'rescued the houseowner', 'went'])
nounsAndVerbsCompatible.append(['the vendor', 'the storeowner', 'recruited', 'satisfied the client', 'had'])
nounsAndVerbsIncompatible.append(['the vendor', 'the storeowner', 'recruited', 'welcomed the client', 'had'])
nounsAndVerbsCompatible.append(['the plumber', 'the apprentice', 'consulted', 'puzzled the woman', 'was'])
nounsAndVerbsIncompatible.append(['the plumber', 'the apprentice', 'consulted', 'assisted the woman', 'was'])
nounsAndVerbsCompatible.append(['the sponsor', 'the musician', 'entertained', 'captivated the onlookers', 'had'])
nounsAndVerbsIncompatible.append(['the sponsor', 'the musician', 'entertained', 'cheered the onlookers', 'had'])
nounsAndVerbsCompatible.append(['the carpenter', 'the craftsman', 'carried', 'confused the apprentice', 'was'])
nounsAndVerbsIncompatible.append(['the carpenter', 'the craftsman', 'carried', 'trained the apprentice', 'was'])
nounsAndVerbsCompatible.append(['the daughter', 'the sister', 'found', 'frightened the grandmother', 'seemed'])
nounsAndVerbsIncompatible.append(['the daughter', 'the sister', 'found', 'greeted the grandmother', 'seemed'])
nounsAndVerbsCompatible.append(['the tenant', 'the foreman', 'looked for', 'annoyed the shepherd', 'proved'])
nounsAndVerbsIncompatible.append(['the tenant', 'the foreman', 'looked for', 'questioned the shepherd', 'proved'])
nounsAndVerbsCompatible.append(['the musician', 'the father', 'missed', 'displeased the artist', 'had'])
nounsAndVerbsIncompatible.append(['the musician', 'the father', 'missed', 'injured the artist', 'had'])
nounsAndVerbsCompatible.append(['the pharmacist', 'the stranger', 'saw', 'distracted the customer', 'sounded'])
nounsAndVerbsIncompatible.append(['the pharmacist', 'the stranger', 'saw', 'questioned the customer', 'sounded'])
nounsAndVerbsCompatible.append(['the bureaucrat', 'the guard', 'shouted at', 'disturbed the newscaster', 'had'])
nounsAndVerbsIncompatible.append(['the bureaucrat', 'the guard', 'shouted at', 'instructed the newscaster', 'had'])
nounsAndVerbsCompatible.append(['the cousin', 'the brother', 'described', 'troubled the uncle', 'gave'])
nounsAndVerbsIncompatible.append(['the cousin', 'the brother', 'described', 'killed the uncle', 'gave'])
nounsAndVerbsCompatible.append(['the surgeon', 'the patient', 'thanked', 'shocked his colleagues', 'was'])
nounsAndVerbsIncompatible.append(['the surgeon', 'the patient', 'thanked', 'cured his colleagues', 'was'])




assert len(nounsAndVerbsCompatible) == len(nounsAndVerbsIncompatible)

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

topNouns.append("assessment")
topNouns.append("truth")
topNouns.append("declaration")
topNouns.append("complaint")
topNouns.append("admission")
topNouns.append("disclosure")
topNouns.append("confirmation")
topNouns.append("guess")
topNouns.append("remark")
topNouns.append("news")
topNouns.append("proof")
topNouns.append("message")
topNouns.append("announcement")
topNouns.append("statement")
topNouns.append("thought")
topNouns.append("allegation")
topNouns.append("indication")
topNouns.append("recognition")
topNouns.append("speculation")
topNouns.append("accusation")
topNouns.append("reminder")
topNouns.append("rumor")
topNouns.append("finding")
topNouns.append("idea")
topNouns.append("feeling")
topNouns.append("conjecture")
topNouns.append("perception")
topNouns.append("certainty")
topNouns.append("revelation")
topNouns.append("understanding")
topNouns.append("claim")
topNouns.append("view")
topNouns.append("observation")
topNouns.append("conviction")
topNouns.append("presumption")
topNouns.append("intuition")
topNouns.append("opinion")
topNouns.append("conclusion")
topNouns.append("notion")
topNouns.append("suggestion")
topNouns.append("sense")
topNouns.append("suspicion")
topNouns.append("assurance")
topNouns.append("insinuation")
topNouns.append("realization")
topNouns.append("assertion")
topNouns.append("impression")
topNouns.append("contention")
topNouns.append("assumption")
topNouns.append("belief")
topNouns.append("fact")

topNouns = sorted(list(set(topNouns)))

import random


with open("/u/scr/mhahn/stimuli_lm.txt_GRNN", "r") as inFile:
   sentences = [[w.split("\t") for w in q.split("\n")] for q in inFile.read().strip().split("<eos>")[1:]]
for i in range(len(sentences)):
   sentence = sentences[i][1:-1]
   if len(sentence) == 0:
     continue
   form = " ".join([x[2] for x in sentence])
   surprisal = float(sentence[-1][3])
   sentences[i] = (form, surprisal)

with open("/u/scr/mhahn/log-gulordava.tsv", "w") as outFile:
 print("\t".join(["Noun", "Item", "Region", "Condition", "Surprisal", "SurprisalReweighted"]), file=outFile)
 for noun in topNouns:
  for c in [True, False]:
   for stimulus in {True: nounsAndVerbsCompatible, False : nounsAndVerbsIncompatible}[c]:
    N2, N3, V3, V2, V1 = stimulus
    for condition in ["NoSC", "SC", "SCRC"]:
     noun = sentences[0][0].split(" ")[1]
     assert noun in topNouns or noun == "<unk>", noun

     if condition == "NoSC":
        s=(f"The {noun} {V1}")
     elif condition == "SC":
        s=(f"The {noun} that {N2} {V2} {V1}")
     else:
        s=(f"The {noun} that {N2} who {N3} {V3} {V2} {V1}")
#     print(s)
     if s != sentences[0][0]:
       print("WARNING", (s, sentences[0]))
     print("\t".join([noun, (stimulus[0]+"_"+stimulus[1]).replace("the ", ""), "V1_0", condition+"_"+("co" if c else "in"), str(sentences[0][1]), str(sentences[0][1])]), file=outFile)
     sentences = sentences[1:]

#Noun    Item    Region  Condition       Surprisal       SurprisalReweighted     ThatFraction    ThatFractionReweighted  SurprisalsWithThat      SurprisalsWithoutThat
#story   teacher_principal       V1_0    SCRC_co 2.292   2.292   100     100     2.292053699493408       nan
#story   teacher_principal       V1_0    SC_co   2.831   2.831   100     100     2.8314902782440186      nan
#story   teacher_principal       V1_0    SCRC_in 3.377   3.377   100     100     3.3770246505737305      nan

