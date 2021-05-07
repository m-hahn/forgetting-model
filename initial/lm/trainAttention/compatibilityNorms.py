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
nounsAndVerbsIncompatible.append(["the principal",       "the teacher",        "kissed",      "was fired",                     "was quoted in the newspaper", "was the XXXX quoted in the newspaper ?", "Y"])
nounsAndVerbsIncompatible.append(["the sculptor",        "the painter",        "admired",    "was n't talented",   "was completely untrue", "was the XXXX untrue ?", "Y"])
nounsAndVerbsIncompatible.append(["the consultant",      "the artist",         "hired",      "was a fraud",       "shocked everyone", "did the XXXX shock everyone ?", "Y"])
nounsAndVerbsIncompatible.append(["the runner",          "the psychiatrist",   "treated",    "was doping",        "was ridiculous", "was the XXXX ridiculous ?", "Y"])
nounsAndVerbsIncompatible.append(["the child",           "the medic",          "rescued",    "was unharmed",      "relieved everyone", "did the XXXX relieve everyone ?", "Y"])
nounsAndVerbsIncompatible.append(["the criminal",        "the officer",        "arrested",   "was guilty",        "was entirely bogus", "was the XXXX bogus ?", "Y"])
nounsAndVerbsIncompatible.append(["the student",         "the professor",      "hated",      "dropped out",       "made the professor happy", "did the XXXX make the professor happy ?", "Y"])
nounsAndVerbsIncompatible.append(["the mobster",         "the media",          "portrayed",  "had disappeared",    "turned out to be true", "did the XXXX turn out to be true ?", "Y"])
nounsAndVerbsIncompatible.append(["the actor",           "the star",        "loved",      "was missing",       "made her cry", "did the XXXX almost make her cry ?", "Y"])
nounsAndVerbsIncompatible.append(["the preacher",        "the parishioners",   "fired",      "stole money",        "proved to be true", "did the XXXX prove to be true ?", "Y"])
nounsAndVerbsIncompatible.append(["the violinist",       "the sponsors",       "backed",     "abused drugs",                       "is likely true", "was the XXXX likely true ?", "Y"])
nounsAndVerbsIncompatible.append(["the senator",         "the diplomat",       "opposed",    "was winning",                   "made him angry", "did the XXXX make him angry ?", "Y"])
nounsAndVerbsIncompatible.append(["the commander",       "the president",      "appointed",  "was corrupt",         "troubled people", "did the XXXX trouble people ?", "Y"])
nounsAndVerbsIncompatible.append(["the victim",         "the criminal",       "assaulted",  "was surviving",         "calmed everyone down", "did the XXXX calm everyone down ?", "Y"])
nounsAndVerbsIncompatible.append(["the politician",      "the banker",         "bribed",     "stole money",         "came as a shock to his supporters", "did the XXXX come as a shock ?", "Y"])
nounsAndVerbsIncompatible.append(["the surgeon",         "the patient",        "thanked",    "had no degree",         "was not a surprise", "was the XXXX unsurprising ?", "Y"])
nounsAndVerbsIncompatible.append(["the extremist",       "the agent",          "caught",     "got an award",         "was disconcerting", "was the XXXX disconcerting ?", "Y"])
nounsAndVerbsIncompatible.append(["the clerk",           "the customer",       "called",     "was a hero",         "seemed absurd", "did the XXXX seem absurd ?", "Y"])
nounsAndVerbsIncompatible.append(["the trader",          "the businessman",    "consulted",  "had insider information",         "was confirmed", "was the XXXX confirmed ?", "Y"])
nounsAndVerbsIncompatible.append(["the ceo",             "the employee",       "impressed",  "was retiring",         "was entirely correct", "was the XXXX correct ?", "Y"])

nounsAndVerbsCompatible = []
#nounsAndVerbsCompatible.append(["the principal",       "the teacher",        "kissed",      "appeared on tv",                     "was quoted in the newspaper", "was the XXXX quoted in the newspaper ?", "Y"])
#nounsAndVerbsCompatible.append(["the sculptor",        "the painter",        "admired",    "surprised the doctor",   "was completely untrue", "was the XXXX untrue ?", "Y"])
#nounsAndVerbsCompatible.append(["the consultant",      "the artist",         "hired",      "was confirmed",       "shocked everyone", "did the XXXX shock everyone ?", "Y"])
#nounsAndVerbsCompatible.append(["the runner",          "the psychiatrist",   "treated",    "was credible",        "was ridiculous", "was the XXXX ridiculous ?", "Y"])
#nounsAndVerbsCompatible.append(["the child",           "the medic",          "rescued",    "made people happy",      "relieved everyone", "did the XXXX relieve everyone ?", "Y"])
#nounsAndVerbsCompatible.append(["the criminal",        "the officer",        "arrested",   "was refuted",        "was entirely bogus", "was the XXXX bogus ?", "Y"])
#nounsAndVerbsCompatible.append(["the student",         "the professor",      "hated",      "shocked his colleagues",       "made the professor happy", "did the XXXX make the professor happy ?", "Y"])
#nounsAndVerbsCompatible.append(["the mobster",         "the media",          "portrayed",  "calmed everyone down",    "turned out to be true", "did the XXXX turn out to be true ?", "Y"])
#nounsAndVerbsCompatible.append(["the actor",           "the starlet",        "loved",      "was quoted in newspapers",       "made her cry", "did the XXXX almost make her cry ?", "Y"])
#nounsAndVerbsCompatible.append(["the preacher",        "the parishioners",   "fired",      "was foolish",        "proved to be true", "did the XXXX prove to be true ?", "Y"])
#nounsAndVerbsCompatible.append(["the violinist",       "the sponsors",       "backed",     "made her cry",                       "is likely true", "was the XXXX likely true ?", "Y"])
#nounsAndVerbsCompatible.append(["the senator",         "the diplomat",       "opposed",    "annoyed him",                   "really made him angry", "did the XXXX make him angry ?", "Y"])
#nounsAndVerbsCompatible.append(["the commander",       "the president",      "appointed",  "was dangerous",         "troubled people", "did the XXXX trouble people ?", "Y"])
#nounsAndVerbsCompatible.append(["the victim",         "the criminal",       "assaulted",  "remained hidden",         "calmed everyone down", "did the XXXX calm everyone down ?", "Y"])
#nounsAndVerbsCompatible.append(["the politician",      "the banker",         "bribed",     "was popular",         "came as a shock to his supporters", "did the XXXX come as a shock ?", "Y"])
#nounsAndVerbsCompatible.append(["the surgeon",         "the patient",        "thanked",    "was widely known",         "was not a surprise", "was the XXXX unsurprising ?", "Y"])
#nounsAndVerbsCompatible.append(["the extremist",       "the agent",          "caught",     "stunned everyone",         "was disconcerting", "was the XXXX disconcerting ?", "Y"])
#nounsAndVerbsCompatible.append(["the clerk",           "the customer",       "called",     "was idiotic",         "seemed absurd", "did the XXXX seem absurd ?", "Y"])
#nounsAndVerbsCompatible.append(["the trader",          "the businessman",    "consulted",  "sounded hopeful",         "was confirmed", "was the XXXX confirmed ?", "Y"])
#nounsAndVerbsCompatible.append(["the CEO",             "the employee",       "impressed",  "hurt him",         "was entirely correct", "was the XXXX correct ?", "Y"])
#
#
#
#
#
#nounsAndVerbsCompatible.append(["the clerk", "the customer", "called", "was sad", "seemed absurd ."])
#nounsAndVerbsIncompatible.append(["the clerk", "the customer", "called", "was heroic", "seemed absurd ."])
#nounsAndVerbsCompatible.append(["the CEO", "the employee", "impressed", "deserved attention", "was entirely correct ."])
#nounsAndVerbsIncompatible.append(["the CEO", "the employee", "impressed", "was retiring", "was entirely correct ."])
#nounsAndVerbsCompatible.append(["the driver", "the tourist", "consulted", "was crazy", "seemed hard to believe ."])
#nounsAndVerbsIncompatible.append(["the driver", "the tourist", "consulted", "was lying", "seemed hard to believe ."])
#nounsAndVerbsCompatible.append(["the bookseller", "the thief", "robbed", "was a total fraud", "shocked his family ."])
#nounsAndVerbsIncompatible.append(["the bookseller", "the thief", "robbed", "got a heart attack", "shocked his family ."])
#nounsAndVerbsCompatible.append(["the neighbor", "the woman", "distrusted", "startled the child", "was a lie ."])
#nounsAndVerbsIncompatible.append(["the neighbor", "the woman", "distrusted", "killed the dog", "was a lie ."])
#nounsAndVerbsCompatible.append(["the scientist", "the mayor", "trusted", "couldn't be trusted", "was only a malicious smear ."])
#nounsAndVerbsIncompatible.append(["the scientist", "the mayor", "trusted", "had faked data", "was only a malicious smear ."])
#nounsAndVerbsCompatible.append(["the lifesaver", "the swimmer", "called", "pleased the children", "impressed the whole city ."])
#nounsAndVerbsIncompatible.append(["the lifesaver", "the swimmer", "called", "saved the children", "impressed the whole city ."])
#nounsAndVerbsCompatible.append(["the entrepreneur", "the philanthropist", "funded", "exasperated the nurse", "came as a disappointment ."])
#nounsAndVerbsIncompatible.append(["the entrepreneur", "the philanthropist", "funded", "wasted the money", "came as a disappointment ."])
#nounsAndVerbsCompatible.append(["the trickster", "the woman", "recognized", "was finally acknowledged", "calmed people down ."])
#nounsAndVerbsIncompatible.append(["the trickster", "the woman", "recognized", "was finally caught", "calmed people down ."])
#nounsAndVerbsCompatible.append(["the student", "the bully", "intimidated", "drove everyone crazy", "devastated his parents ."])
#nounsAndVerbsIncompatible.append(["the student", "the bully", "intimidated", "plagiarized his homework", "devastated his parents ."])
#nounsAndVerbsCompatible.append(["the carpenter", "the craftsman", "carried", "confused the apprentice", "was acknowledged ."])
#nounsAndVerbsIncompatible.append(["the carpenter", "the craftsman", "carried", "hurt the apprentice", "was acknowledged ."])
#nounsAndVerbsCompatible.append(["the daughter", "the sister", "found", "frightened the grandmother", "seemed concerning ."])
#nounsAndVerbsIncompatible.append(["the daughter", "the sister", "found", "greeted the grandmother", "seemed concerning ."])
#nounsAndVerbsCompatible.append(["the tenant", "the foreman", "looked for", "annoyed the shepherd", "proved to be made up ."])
#nounsAndVerbsIncompatible.append(["the tenant", "the foreman", "looked for", "questioned the shepherd", "proved to be made up ."])
#nounsAndVerbsCompatible.append(["the musician", "the father", "missed", "displeased the artist", "confused the banker ."])
#nounsAndVerbsIncompatible.append(["the musician", "the father", "missed", "injured the artist", "confused the banker ."])
#nounsAndVerbsCompatible.append(["the pharmacist", "the stranger", "saw", "distracted the customer", "sounded surprising ."])
#nounsAndVerbsIncompatible.append(["the pharmacist", "the stranger", "saw", "questioned the customer", "sounded surprising ."])
#nounsAndVerbsCompatible.append(["the bureaucrat", "the guard", "shouted at", "disturbed the newscaster", "annoyed the neighbor ."])
#nounsAndVerbsIncompatible.append(["the bureaucrat", "the guard", "shouted at", "instructed the newscaster", "annoyed the neighbor ."])
#nounsAndVerbsCompatible.append(["the cousin", "the brother", "attacked", "troubled the uncle", "startled the mother ."])
#nounsAndVerbsIncompatible.append(["the cousin", "the brother", "attacked", "killed the uncle", "startled the mother ."])
#
VNnounsAndVerbsIncompatible = []
VNnounsAndVerbsCompatible = []

VNnounsAndVerbsIncompatible.append(["the teacher", "the principal", "liked", "failed the student", "was only a malicious smear ."])
VNnounsAndVerbsCompatible.append(["the teacher", "the principal", "liked", "annoyed the student", "was only a malicious smear ."])
VNnounsAndVerbsIncompatible.append(["the doctor", "the colleague", "distrusted", "cured the patients", "seemed hard to believe ."])
VNnounsAndVerbsCompatible.append(["the doctor", "the colleague", "distrusted", "bothered the patients", "seemed hard to believe ."])
VNnounsAndVerbsIncompatible.append(["the bully", "the children", "hated", "harassed the boy", "was entirely correct ."])
VNnounsAndVerbsCompatible.append(["the bully", "the children", "hated", "shocked the boy", "was entirely correct ."])
VNnounsAndVerbsIncompatible.append(["the agent", "the fbi", "sent", "arrested the criminal", "was acknowledged ."])
VNnounsAndVerbsCompatible.append(["the agent", "the fbi", "sent", "confused the criminal", "was acknowledged ."])
VNnounsAndVerbsIncompatible.append(["the senator", "the diplomat", "supported", "defeated the opponent", "deserved attention ."])
VNnounsAndVerbsCompatible.append(["the senator", "the diplomat", "supported", "troubled the opponent", "deserved attention ."])
VNnounsAndVerbsIncompatible.append(["the fiancé", "the author", "met", "married the bride", "did not surprise anyone ."])
VNnounsAndVerbsCompatible.append(["the fiancé", "the author", "met", "startled the bride", "did not surprise anyone ."])
VNnounsAndVerbsIncompatible.append(["the businessman", "the sponsor", "backed", "fired the employee", "came as a disappointment ."])
VNnounsAndVerbsCompatible.append(["the businessman", "the sponsor", "backed", "hurt the employee", "came as a disappointment ."])
VNnounsAndVerbsIncompatible.append(["the thief", "the detective", "caught", "robbed the woman", "shocked her family ."])
VNnounsAndVerbsCompatible.append(["the thief", "the detective", "caught", "enraged the woman", "shocked her family ."])
VNnounsAndVerbsIncompatible.append(["the criminal", "the stranger", "distracted", "killed the officer", "seemed concerning ."])
VNnounsAndVerbsCompatible.append(["the criminal", "the stranger", "distracted", "surprised the officer", "seemed concerning ."])
VNnounsAndVerbsIncompatible.append(["the customer", "the vendor", "welcomed", "called the clerk", "was very believable ."])
VNnounsAndVerbsCompatible.append(["the customer", "the vendor", "welcomed", "horrified the clerk", "was very believable ."])
VNnounsAndVerbsIncompatible.append(["the president", "the farmer", "admired", "appointed the commander", "was entirely bogus ."])
VNnounsAndVerbsCompatible.append(["the president", "the farmer", "admired", "impressed the commander", "was entirely bogus ."])
VNnounsAndVerbsIncompatible.append(["the victim", "the swimmer", "rescued", "sued the criminal", "appeared on tv ."])
VNnounsAndVerbsCompatible.append(["the victim", "the swimmer", "rescued", "surprised the criminal", "appeared on tv ."])
VNnounsAndVerbsIncompatible.append(["the guest", "the cousin", "invited", "visited the uncle", "calmed everyone down ."])
VNnounsAndVerbsCompatible.append(["the guest", "the cousin", "invited", "pleased the uncle", "calmed everyone down ."])
VNnounsAndVerbsIncompatible.append(["the psychiatrist", "the nurse", "assisted", "diagnosed the patient", "impressed the whole city ."])
VNnounsAndVerbsCompatible.append(["the psychiatrist", "the nurse", "assisted", "pleased the patient", "impressed the whole city ."])
VNnounsAndVerbsIncompatible.append(["the driver", "the guide", "called", "drove the tourist", "was absolutely true ."])
VNnounsAndVerbsCompatible.append(["the driver", "the guide", "called", "amazed the tourist", "was absolutely true ."])
VNnounsAndVerbsIncompatible.append(["the actor", "the fans", "loved", "greeted the director", "appeared to be true ."])
VNnounsAndVerbsCompatible.append(["the actor", "the fans", "loved", "astonished the director", "appeared to be true ."])
VNnounsAndVerbsIncompatible.append(["the banker", "the analyst", "deceived", "trusted the customer", "proved to be made up ."])
VNnounsAndVerbsCompatible.append(["the banker", "the analyst", "deceived", "excited the customer", "proved to be made up ."])
VNnounsAndVerbsIncompatible.append(["the judge", "the attorney", "hated", "convicted the defendant", "was a lie ."])
VNnounsAndVerbsCompatible.append(["the judge", "the attorney", "hated", "vindicated the defendant", "was a lie ."])
VNnounsAndVerbsIncompatible.append(["the captain", "the crew", "trusted", "commanded the sailor", "was nice to hear ."])
VNnounsAndVerbsCompatible.append(["the captain", "the crew", "trusted", "encouraged the sailor", "was nice to hear ."])
VNnounsAndVerbsIncompatible.append(["the manager", "the boss", "authorized", "hired the intern", "seemed absurd ."])
VNnounsAndVerbsCompatible.append(["the manager", "the boss", "authorized", "saddened the intern", "seemed absurd ."])
VNnounsAndVerbsIncompatible.append(["the plaintiff", "the jury", "interrogated", "attacked the witness", "made it into the news ."])
VNnounsAndVerbsCompatible.append(["the plaintiff", "the jury", "interrogated", "startled the witness", "made it into the news ."])
VNnounsAndVerbsIncompatible.append(["the guest", "the thug", "hit", "tricked the bartender", "sounded hilarious ."])
VNnounsAndVerbsCompatible.append(["the guest", "the thug", "hit", "stunned the bartender", "sounded hilarious ."])
VNnounsAndVerbsIncompatible.append(["the pediatrician", "the receptionist", "supported", "distrusted the parent", "troubled people ."])
VNnounsAndVerbsCompatible.append(["the pediatrician", "the receptionist", "supported", "disturbed the parent", "troubled people ."])
VNnounsAndVerbsIncompatible.append(["the medic", "the survivor", "thanked", "greeted the surgeon", "turned out to be untrue ."])
VNnounsAndVerbsCompatible.append(["the medic", "the survivor", "thanked", "annoyed the surgeon", "turned out to be untrue ."])
VNnounsAndVerbsIncompatible.append(["the lifeguard", "the soldier", "taught", "rescued the swimmer", "pleased the townspeople ."])
VNnounsAndVerbsCompatible.append(["the lifeguard", "the soldier", "taught", "encouraged the swimmer", "pleased the townspeople ."])
VNnounsAndVerbsIncompatible.append(["the fisherman", "the gardener", "helped", "admired the politician", "was interesting ."])
VNnounsAndVerbsCompatible.append(["the fisherman", "the gardener", "helped", "delighted the politician", "was interesting ."])
VNnounsAndVerbsIncompatible.append(["the janitor", "the organizer", "criticized", "ignored the audience", "was funny ."])
VNnounsAndVerbsCompatible.append(["the janitor", "the organizer", "criticized", "amused the audience", "was funny ."])
VNnounsAndVerbsIncompatible.append(["the investor", "the scientist", "hated", "deceived the entrepreneur", "taught everyone a lesson ."])
VNnounsAndVerbsCompatible.append(["the investor", "the scientist", "hated", "disappointed the entrepreneur", "taught everyone a lesson ."])
VNnounsAndVerbsIncompatible.append(["the firefighter", "the neighbor", "insulted", "rescued the resident", "struck john as implausible ."])
VNnounsAndVerbsCompatible.append(["the firefighter", "the neighbor", "insulted", "disappointed the resident", "struck john as implausible ."])
VNnounsAndVerbsIncompatible.append(["the vendor", "the salesman", "recruited", "welcomed the client", "excited the boss ."])
VNnounsAndVerbsCompatible.append(["the vendor", "the salesman", "recruited", "enchanted the client", "excited the boss ."])
VNnounsAndVerbsIncompatible.append(["the plumber", "the apprentice", "consulted", "assisted the woman", "was true ."])
VNnounsAndVerbsCompatible.append(["the plumber", "the apprentice", "consulted", "puzzled the woman", "was true ."])
VNnounsAndVerbsIncompatible.append(["the sponsor", "the musician", "entertained", "cheered the onlookers", "pleased everyone ."])
VNnounsAndVerbsCompatible.append(["the sponsor", "the musician", "entertained", "captivated the onlookers", "pleased everyone ."])


nounsAndVerbsCompatible = []
nounsAndVerbsCompatible.append(["the principal",       "the teacher",        "kissed",      "appeared on tv",                     "was quoted in the newspaper", "was the XXXX quoted in the newspaper ?", "Y"])
nounsAndVerbsCompatible.append(["the sculptor",        "the painter",        "admired",    "surprised the doctor",   "was completely untrue", "was the XXXX untrue ?", "Y"])
nounsAndVerbsCompatible.append(["the consultant",      "the artist",         "hired",      "was confirmed",       "shocked everyone", "did the XXXX shock everyone ?", "Y"])
nounsAndVerbsCompatible.append(["the runner",          "the psychiatrist",   "treated",    "was credible",        "was ridiculous", "was the XXXX ridiculous ?", "Y"])
nounsAndVerbsCompatible.append(["the child",           "the medic",          "rescued",    "made people happy",      "relieved everyone", "did the XXXX relieve everyone ?", "Y"])
nounsAndVerbsCompatible.append(["the criminal",        "the officer",        "arrested",   "was refuted",        "was entirely bogus", "was the XXXX bogus ?", "Y"])
nounsAndVerbsCompatible.append(["the student",         "the professor",      "hated",      "shocked his colleagues",       "made the professor happy", "did the XXXX make the professor happy ?", "Y"])
nounsAndVerbsCompatible.append(["the mobster",         "the media",          "portrayed",  "calmed everyone down",    "turned out to be true", "did the XXXX turn out to be true ?", "Y"])
nounsAndVerbsCompatible.append(["the actor",           "the star",        "loved",      "was quoted in newspapers",       "made her cry", "did the XXXX almost make her cry ?", "Y"])
nounsAndVerbsCompatible.append(["the preacher",        "the parishioners",   "fired",      "was foolish",        "proved to be true", "did the XXXX prove to be true ?", "Y"])
nounsAndVerbsCompatible.append(["the violinist",       "the sponsors",       "backed",     "made her cry",                       "is likely true", "was the XXXX likely true ?", "Y"])
nounsAndVerbsCompatible.append(["the senator",         "the diplomat",       "opposed",    "annoyed him",                   "made him angry", "did the XXXX make him angry ?", "Y"])
nounsAndVerbsCompatible.append(["the commander",       "the president",      "appointed",  "was dangerous",         "troubled people", "did the XXXX trouble people ?", "Y"])
nounsAndVerbsCompatible.append(["the victim",         "the criminal",       "assaulted",  "remained hidden",         "calmed everyone down", "did the XXXX calm everyone down ?", "Y"])
nounsAndVerbsCompatible.append(["the politician",      "the banker",         "bribed",     "was popular",         "came as a shock to his supporters", "did the XXXX come as a shock ?", "Y"])
nounsAndVerbsCompatible.append(["the surgeon",         "the patient",        "thanked",    "was widely known",         "was not a surprise", "was the XXXX unsurprising ?", "Y"])
nounsAndVerbsCompatible.append(["the extremist",       "the agent",          "caught",     "stunned everyone",         "was disconcerting", "was the XXXX disconcerting ?", "Y"])
nounsAndVerbsCompatible.append(["the clerk",           "the customer",       "called",     "was dumb",         "seemed absurd", "did the XXXX seem absurd ?", "Y"])
nounsAndVerbsCompatible.append(["the trader",          "the businessman",    "consulted",  "sounded hopeful",         "was confirmed", "was the XXXX confirmed ?", "Y"])
nounsAndVerbsCompatible.append(["the ceo",             "the employee",       "impressed",  "hurt him",         "was entirely correct", "was the XXXX correct ?", "Y"])





nounsAndVerbsCompatible.append(["the clerk", "the customer", "called", "was sad", "seemed absurd ."])
nounsAndVerbsIncompatible.append(["the clerk", "the customer", "called", "was heroic", "seemed absurd ."])
nounsAndVerbsCompatible.append(["the ceo", "the employee", "impressed", "deserved attention", "was entirely correct ."])
nounsAndVerbsIncompatible.append(["the ceo", "the employee", "impressed", "was retiring", "was entirely correct ."])
nounsAndVerbsCompatible.append(["the driver", "the tourist", "consulted", "was crazy", "seemed hard to believe ."])
nounsAndVerbsIncompatible.append(["the driver", "the tourist", "consulted", "was lying", "seemed hard to believe ."])
nounsAndVerbsCompatible.append(["the bookseller", "the thief", "robbed", "was a total fraud", "shocked his family ."])
nounsAndVerbsIncompatible.append(["the bookseller", "the thief", "robbed", "got a heart attack", "shocked his family ."])
nounsAndVerbsCompatible.append(["the neighbor", "the woman", "distrusted", "startled the child", "was a lie ."])
nounsAndVerbsIncompatible.append(["the neighbor", "the woman", "distrusted", "killed the dog", "was a lie ."])
nounsAndVerbsCompatible.append(["the scientist", "the mayor", "trusted", "could n't be trusted", "was only a malicious smear ."])
nounsAndVerbsIncompatible.append(["the scientist", "the mayor", "trusted", "had faked data", "was only a malicious smear ."])
nounsAndVerbsCompatible.append(["the lifeguard", "the swimmer", "called", "pleased the children", "impressed the whole city ."])
nounsAndVerbsIncompatible.append(["the lifeguard", "the swimmer", "called", "saved the children", "impressed the whole city ."])
nounsAndVerbsCompatible.append(["the entrepreneur", "the philanthropist", "funded", "hurt the nurse", "came as a disappointment ."])
nounsAndVerbsIncompatible.append(["the entrepreneur", "the philanthropist", "funded", "wasted the money", "came as a disappointment ."])
nounsAndVerbsCompatible.append(["the trickster", "the woman", "recognized", "was finally acknowledged", "calmed people down ."])
nounsAndVerbsIncompatible.append(["the trickster", "the woman", "recognized", "was finally caught", "calmed people down ."])
nounsAndVerbsCompatible.append(["the student", "the bully", "intimidated", "drove everyone crazy", "devastated his parents ."])
nounsAndVerbsIncompatible.append(["the student", "the bully", "intimidated", "faked his homework", "devastated his parents ."])
nounsAndVerbsCompatible.append(["the carpenter", "the craftsman", "carried", "confused the apprentice", "was acknowledged ."])
nounsAndVerbsIncompatible.append(["the carpenter", "the craftsman", "carried", "hurt the apprentice", "was acknowledged ."])
nounsAndVerbsCompatible.append(["the daughter", "the sister", "found", "frightened the grandmother", "seemed concerning ."])
nounsAndVerbsIncompatible.append(["the daughter", "the sister", "found", "greeted the grandmother", "seemed concerning ."])
nounsAndVerbsCompatible.append(["the tenant", "the foreman", "looked for", "annoyed the shepherd", "proved to be made up ."])
nounsAndVerbsIncompatible.append(["the tenant", "the foreman", "looked for", "questioned the shepherd", "proved to be made up ."])
nounsAndVerbsCompatible.append(["the musician", "the father", "missed", "displeased the artist", "confused the banker ."])
nounsAndVerbsIncompatible.append(["the musician", "the father", "missed", "injured the artist", "confused the banker ."])
nounsAndVerbsCompatible.append(["the pharmacist", "the stranger", "saw", "distracted the customer", "sounded surprising ."])
nounsAndVerbsIncompatible.append(["the pharmacist", "the stranger", "saw", "questioned the customer", "sounded surprising ."])
nounsAndVerbsCompatible.append(["the bureaucrat", "the guard", "shouted at", "disturbed the newscaster", "annoyed the neighbor ."])
nounsAndVerbsIncompatible.append(["the bureaucrat", "the guard", "shouted at", "instructed the newscaster", "annoyed the neighbor ."])
nounsAndVerbsCompatible.append(["the cousin", "the brother", "attacked", "troubled the uncle", "startled the mother ."])
nounsAndVerbsIncompatible.append(["the cousin", "the brother", "attacked", "killed the uncle", "startled the mother ."])


for x in VNnounsAndVerbsIncompatible:
   x.append("v")
for x in VNnounsAndVerbsCompatible:
   x.append("v")

for x in nounsAndVerbsIncompatible:
   x.append("o")
for x in nounsAndVerbsCompatible:
   x.append("o")
assert len(nounsAndVerbsCompatible) == len(nounsAndVerbsIncompatible)

nounsAndVerbsCompatible += VNnounsAndVerbsCompatible
nounsAndVerbsIncompatible += VNnounsAndVerbsIncompatible


assert len(nounsAndVerbsCompatible) == len(nounsAndVerbsIncompatible)

import math

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

topNouns = list(set(topNouns))



with open("../../../../forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", "r") as inFile:
   counts = [x.replace('"', '').split("\t") for x in inFile.read().strip().split("\n")]
   header = ["LineNum"] + counts[0]
   assert len(header) == len(counts[1])
   header = dict(list(zip(header, range(len(header)))))
   counts = {line[header["Noun"]] : line for line in counts[1:]}


print(len(topNouns))
print([x for x in topNouns if x not in counts])
topNouns = [x for x in topNouns if x in counts]

def thatBias(noun):
   return math.log(float(counts[noun][header["CountThat"]]))-math.log(float(counts[noun][header["CountBare"]]))

topNouns = sorted(list(set(topNouns)), key=lambda x:thatBias(x))

print(topNouns)
print(len(topNouns))
#quit()


# This is to ensure the tsv files are useful even when the script is stopped prematurely
random.shuffle(topNouns)

    
#plain_lm = PlainLanguageModel()
#plain_lmFileName = "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars.py"

#if args.load_from_plain_lm is not None:
#  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+plain_lmFileName+"_code_"+str(args.load_from_plain_lm)+".txt")
#  for i in range(len(checkpoint["components"])):
#      plain_lm.modules[i].load_state_dict(checkpoint["components"][i])
#  del checkpoint


# Helper Functions

def correlation(x, y):
   variance_x = (x.pow(2)).mean() - x.mean().pow(2)
   variance_y = (y.pow(2)).mean() - y.mean().pow(2)
   return ((x-x.mean())* (y-y.mean())).mean()/(variance_x*variance_y).sqrt()


def rindex(x, y):
   return max([i for i in range(len(x)) if x[i] == y])


def encodeContextCrop(inp, context):
     sentence = context.strip() + " " + inp.strip()
     print("ENCODING", sentence)
     numerified = [stoi_total[char] if char in stoi_total else 2 for char in sentence.split(" ")]
     print(len(numerified))
     numerified = numerified[-args.sequence_length-1:]
     numerified = torch.LongTensor([numerified for _ in range(args.batchSize)]).t().cuda()
     return numerified

def flatten(x):
   l = []
   for y in x:
     for z in y:
        l.append(z)
   return l


calibrationSentences = []
#calibrationSentences.append("she is a good runner but a bad swimmer")
#calibrationSentences.append("she are a good runners but a bad swimmers")
#calibrationSentences.append("the children ate a bowl of international trade for breakfast")
#calibrationSentences.append("the children ate a bowl of tasty cereals for breakfast")
#calibrationSentences.append("they know how to solve the problem")
#calibrationSentences.append("they knows how to solve the problem")
#
## word salad
#calibrationSentences.append("his say later with to use electrons")
#calibrationSentences.append("and a enlarged in transformed into")
#calibrationSentences.append("available a solvated in the cosmetics")
#calibrationSentences.append("it alkali a basis used metal been")
#calibrationSentences.append("has result were are pear masters")
#calibrationSentences.append("for are include course additional barrow")
#calibrationSentences.append("to school a and apple flavors cremations")
#
## acceptable sentences, mostly adapted from COLA, some words exchanged to ensure InVocab
#calibrationSentences.append("she went there yesterday and saw a movie")
#calibrationSentences.append("perseus saw the gorgon in his shield")
#calibrationSentences.append("what did you say ( that ) the poet had written ?")
#calibrationSentences.append("i saw people playing there on the beach")
#calibrationSentences.append("i did n't want any cake")
#calibrationSentences.append("that i should kiss pigs is my fondest dream")
#calibrationSentences.append("ron failed biology , unfortunately")
#calibrationSentences.append("the men chuckle")
#calibrationSentences.append("i expected there to be a problem")
#calibrationSentences.append("gilgamesh wanted to seduce ishtar , and seduce ishtar he did")
#calibrationSentences.append("harry collapsed")
#calibrationSentences.append("i asked who saw what")
#calibrationSentences.append("he has been happy")
#calibrationSentences.append("poseidon had run away , before the executioner murdered hera")
#calibrationSentences.append("merlin is a dangerous sorcerer")
#calibrationSentences.append("anson saw anson")
#calibrationSentences.append("i am to eat macaroni")
#calibrationSentences.append("poseidon had escaped , before the executioner arrived")
#calibrationSentences.append("owners love truffles")
#calibrationSentences.append("humans love to eat owners of pigs")
#calibrationSentences.append("those pigs love truffles")
#calibrationSentences.append("i 'd planned to have finished by now")
#calibrationSentences.append("has the potion not worked ?")
#calibrationSentences.append("what i love is toast and sun dried tomatoes")
#calibrationSentences.append("mary ran")
#calibrationSentences.append("he replied that he was happy")
#calibrationSentences.append("no one could remove the blood from the wall")
#calibrationSentences.append("julie maintained that the barman was sober")
#calibrationSentences.append("benjamin gave lee the cloak")
#calibrationSentences.append("aphrodite wanted hera to persuade athena to leave")
#calibrationSentences.append("gilgamesh is fighting the dragon")
#calibrationSentences.append("i claimed she was pregnant")
#calibrationSentences.append("for jenny , i intended to be present")
#calibrationSentences.append("gilgamesh missed aphrodite")
#calibrationSentences.append("she might be pregnant")
#calibrationSentences.append("anson demonized david at the club")
#calibrationSentences.append("jason asked whether the potion was ready")
#calibrationSentences.append("frieda closed the door")
#calibrationSentences.append("medea might have given jason a poisoned robe ( just treat a poisoned robe as an np")
#calibrationSentences.append("quickly kiss anson !")
#calibrationSentences.append("julie felt hot")
#calibrationSentences.append("agamemnon expected esther to seem to be happy")
#calibrationSentences.append("that the answer is obvious upset hermes")
#calibrationSentences.append("homer recited the poem about achilles ?")
#calibrationSentences.append("no vampire can survive sunrise")
#calibrationSentences.append("under the bed is the best place to hide")
#calibrationSentences.append("anson appeared")
#calibrationSentences.append("there seems to be a problem")
#calibrationSentences.append("i intoned that she was happy")
#calibrationSentences.append("medea saw who ?")
#calibrationSentences.append("no one expected that agamemnon would win")
#calibrationSentences.append("believing that the world is flat gives one some solace")
#calibrationSentences.append("kick them !")
#calibrationSentences.append("medea wondered if the potion was ready")
#calibrationSentences.append("who all did you meet when you were in derry ?")
#calibrationSentences.append("who did you hear an oration about ?")
#calibrationSentences.append("alison ran")
#calibrationSentences.append("romeo sent letters to juliet")
#calibrationSentences.append("richard 's gift of the helicopter to the hospital and of the bus to the school")
#calibrationSentences.append("nathan caused benjamin to see himself in the mirror")
#calibrationSentences.append("a. madeleine planned to catch the sardines and she did")
#calibrationSentences.append("i did not understand")
#calibrationSentences.append("gilgamesh loved ishtar and aphrodite did too")
#calibrationSentences.append("we believed him to be omnipotent")
#calibrationSentences.append("david ate mangoes and raffi should too")
#calibrationSentences.append("julie and fraser ate those delicious pies in julie 's back garden")
#calibrationSentences.append("the old pigs love truffles")
#calibrationSentences.append("the boys all should could go")
#calibrationSentences.append("aphrodite quickly freed the animals")
#calibrationSentences.append("paul had two affairs")
#calibrationSentences.append("what alison and david did was soak their feet in a bucket")
#calibrationSentences.append("anson demonized david almost constantly")
#calibrationSentences.append("anson 's hen nibbled his ear")
#calibrationSentences.append("before the executioner arrived , poseidon had escaped")
#calibrationSentences.append("gilgamesh did n't leave")
#calibrationSentences.append("genie intoned that she was tired")
#calibrationSentences.append("look at all these books which book would you like ?")
#calibrationSentences.append("i do n't remember what i said all ?")
#calibrationSentences.append("the pig grunts")
#calibrationSentences.append("people are stupid")
#calibrationSentences.append("what i arranged was for jenny to be present")
#calibrationSentences.append("i compared ginger to fred")
#calibrationSentences.append("which poet wrote which ode ?")
#calibrationSentences.append("how did julie ask if jenny left ?")
#calibrationSentences.append("dracula thought him to be the prince of darkness")
#calibrationSentences.append("i must eat macaroni")
#calibrationSentences.append("i asked who john would introduce to who")
#calibrationSentences.append("reading shakespeare satisfied me")
#calibrationSentences.append("humans love to eat owners")
#calibrationSentences.append("gilgamesh fears death and achilles does as well")
#calibrationSentences.append("how did julie say that jenny left ?")
#calibrationSentences.append("show me letters !")
#calibrationSentences.append("the readings of shakespeare satisfied me")
#calibrationSentences.append("anson demonized david every day")
#calibrationSentences.append("the students demonstrated this morning")
#calibrationSentences.append("we believed aphrodite to be omnipotent")
#calibrationSentences.append("emily caused benjamin to see himself in the mirror")
#calibrationSentences.append("nothing like that would i ever eat again")
#calibrationSentences.append("where has he put the cake ?")
#calibrationSentences.append("jason persuaded medea to desert her family")
#calibrationSentences.append("gilgamesh perhaps should be leaving")
#calibrationSentences.append("gilgamesh has n't kissed ishtar")
#calibrationSentences.append("it is easy to slay the gorgon")
#calibrationSentences.append("i had the strangest feeling that i knew you")
#calibrationSentences.append("what all did you get for christmas ?")
#
## unacceptable sentences from COLA
#calibrationSentences.append("they drank the pub")
#calibrationSentences.append("the professor talked us")
#calibrationSentences.append("we yelled ourselves")
#calibrationSentences.append("we yelled harry hoarse")
#calibrationSentences.append("harry coughed himself")
#calibrationSentences.append("harry coughed us into a fit")
#calibrationSentences.append("they caused him to become angry by making him")
#calibrationSentences.append("they caused him to become president by making him")
#calibrationSentences.append("they made him to exhaustion")
#calibrationSentences.append("the car honked down the road")
#calibrationSentences.append("the dog barked out of the room")
#calibrationSentences.append("the witch went into the forest by vanishing")
#calibrationSentences.append("the building is tall and tall")
#calibrationSentences.append("this building is taller and taller")
#calibrationSentences.append("this building got than that one")
#calibrationSentences.append("this building is than that one")
#calibrationSentences.append("bill floated into the cave for hours")
#calibrationSentences.append("bill pushed harry off the sofa for hours")
#calibrationSentences.append("bill cried sue to sleep")
#calibrationSentences.append("the elevator rumbled itself to the ground")
#calibrationSentences.append("she yelled hoarse")
#calibrationSentences.append("ted cried to sleep")
#calibrationSentences.append("the ball wriggled itself loose")
#calibrationSentences.append("the most you want , the least you eat")
#calibrationSentences.append("i demand that the more john eat , the more he pay")
#calibrationSentences.append("i demand that john pays more , the more he eat")
#calibrationSentences.append("you get angrier , the more we eat , do n't we")
#calibrationSentences.append("the harder it has rained , how much faster a flow that appears in the river ?")
#calibrationSentences.append("the harder it rains , how much faster that do you run ?")
#



calibrationSentences.append("The divorcee has come to love her life ever since she got divorced.") 
calibrationSentences.append("The mathematician at the banquet baffled the philosopher although she rarely needed anyone else's help.")
calibrationSentences.append("The showman travels to different cities every month.")
calibrationSentences.append("The roommate takes out the garbage every week.")
calibrationSentences.append("The dragon wounded the knight although he was far too crippled to protect the princess.")
calibrationSentences.append("The office-worker worked through the stack of files on his desk quickly.")
calibrationSentences.append("The firemen at the scene apprehended the arsonist because there was a great deal of evidence pointing to his guilt.")
calibrationSentences.append("During the season, the choir holds rehearsals in the church regularly.")
calibrationSentences.append("The speaker who the historian offended kicked a chair after the talk was over and everyone had left the room.")
calibrationSentences.append("The milkman punctually delivers the milk at the door every day.")
calibrationSentences.append("The quarterback dated the cheerleader although this hurt her reputation around school.")
calibrationSentences.append("The citizens of France eat oysters.")
calibrationSentences.append("The bully punched the kid after all the kids had to leave to go to class.")
calibrationSentences.append("After the argument, the husband ignored his wife.")
calibrationSentences.append("The engineer who the lawyer who was by the elevator scolded blamed the secretary but nobody listened to his complaints.")
calibrationSentences.append("The librarian put the book onto the shelf.")
calibrationSentences.append("The photographer processed the film on time.")
calibrationSentences.append("The spider that the boy who was in the yard captured scared the dog since it was larger than the average spider.")
calibrationSentences.append("The sportsman goes jogging in the park regularly.")
calibrationSentences.append("The customer who was on the phone contacted the operator because the new long-distance pricing plan was extremely inconvenient.")
calibrationSentences.append("The private tutor explained the assignment carefully.")
calibrationSentences.append("The audience who was at the club booed the singer before the owner of the bar could remove him from the stage.")
calibrationSentences.append("The defender is constantly scolding the keeper.")
calibrationSentences.append("The hippies who the police at the concert arrested complained to the officials while the last act was going on stage.")
calibrationSentences.append("The natives on the island captured the anthropologist because she had information that could help the tribe.")
calibrationSentences.append("The trainee knew that the task which the director had set for him was impossible to finish within a week.")
calibrationSentences.append("The administrator who the nurse from the clinic supervised scolded the medic while a patient was brought into the emergency room.")
calibrationSentences.append("The company was sure that its new product, which its researchers had developed, would soon be sold out.")
calibrationSentences.append("The astronaut that the journalists who were at the launch worshipped criticized the administrators after he discovered a potential leak in the fuel tank.")
calibrationSentences.append("The janitor who the doorman who was at the hotel chatted with bothered a guest but the manager decided not to fire him for it.")
calibrationSentences.append("The technician at the show repaired the robot while people were taking a break for coffee.")
calibrationSentences.append("The salesman feared that the printer which the customer bought was damaged.")
calibrationSentences.append("The students studied the surgeon whenever he performed an important operation.")
calibrationSentences.append("The locksmith can crack the safe easily.")
calibrationSentences.append("The woman who was in the apartment hired the plumber despite the fact that he couldn't fix the toilet.")
calibrationSentences.append("Yesterday the swimmer saw only a turtle at the beach.")
calibrationSentences.append("The surgeon who the detective who was on the case consulted questioned the coroner because the markings on the body were difficult to explain.")
calibrationSentences.append("The gangster who the detective at the club followed implicated the waitress because the police suspected he had murdered the shopkeeper.")
calibrationSentences.append("During the party everybody was dancing to rock music.")
calibrationSentences.append("The fans at the concert loved the guitarist because he played with so much energy.")
calibrationSentences.append("The intern comforted the patient because he was in great pain.")
calibrationSentences.append("The casino hired the daredevil because he was confident that everything would go according to plan.")
calibrationSentences.append("The beggar is often scrounging for cigarettes.")
calibrationSentences.append("The cartoonist who the readers supported pressured the dean because she thought that censorship was never appropriate.")
calibrationSentences.append("The prisoner who the guard attacked tackled the warden although he had no intention of trying to escape.")
calibrationSentences.append("The passer-by threw the cardboard box into the trash-can with great force.")
calibrationSentences.append("The biker who the police arrested ran a light since he was driving under the influence of alcohol.")
calibrationSentences.append("The scientists who were in the lab studied the alien while the blood sample was run through the computer.")
calibrationSentences.append("The student quickly finished his homework assignments.")
calibrationSentences.append("The environmentalist who the demonstrators at the rally supported calmed the crowd until security came and sent everyone home.")
calibrationSentences.append("The producer shoots a new movie every year.")
calibrationSentences.append("The rebels who were in the jungle captured the diplomat after they threatened to kill his family for not complying with their demands.")
calibrationSentences.append("Dinosaurs ate other reptiles during the stone age.")
calibrationSentences.append("The manager who the baker loathed spoke to the new pastry chef because he had instituted a new dress code for all employees.")
calibrationSentences.append("The teacher doubted that the test that had taken him a long time to design would be easy to answer.")
calibrationSentences.append("The cook who the servant in the kitchen hired offended the butler and then left the mansion early to see a movie at the local theater.")


#in and is the it
#time maintain metal commercially the was salt cut as two-year college flavorings solutions
#a alkali and and four-year into twice programme cation and \tduring addition ammonia burials
#often and used butyllithiums perfumery as this amide



#def getTotalSentenceSurprisalsCalibration(SANITY="Sanity", VERBS=2): # Surprisal for EOS after 2 or 3 verbs
#    assert SANITY in ["Sanity", "Model"]
#    assert VERBS in [1,2]
#    print(plain_lm) 
#    surprisalsPerNoun = {}
#    thatFractionsPerNoun = {}
#    numberOfSamples = 12
#    with torch.no_grad():
#     with open("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w") as outFile:
#      print("Sentence", "Region", "Word", "Surprisal", file=outFile)
#
#      for sentence in calibrationSentences:
#            print(sentence)
#            context = "later the nurse suggested they treat the patient with an antibiotic but in the end this did not happen . "
#            remainingInput = sentence.split(" ")
#            for i in range(len(remainingInput)):
#              numerified = encodeContextCrop(" ".join(remainingInput[:i+1]), context)
#              assert numerified.size()[0] == args.sequence_length+1, (numerified.size())
#              # Run the noise model
#              numeric, numeric_noised = forward(numerified, train=False, printHere=False, provideAttention=False, onlyProvideMemoryResult=True, NUMBER_OF_REPLICATES=numberOfSamples)
#              numeric_noised = torch.where(numeric == stoi["."]+3, numeric, numeric_noised)
#              numeric = numeric.unsqueeze(2).expand(-1, -1, 24).view(-1, numberOfSamples*24)
#              numeric_noised = numeric_noised.unsqueeze(2).expand(-1, -1, 24).contiguous().view(-1, numberOfSamples*24)
#              numeric_noised[args.sequence_length] = 0 # A simple hack for dealing with the issue that the last word 
#              # Get samples from the reconstruction posterior
#              result, resultNumeric, fractions, thatProbs = autoencoder.sampleReconstructions(numeric, numeric_noised, None, 2, numberOfBatches=numberOfSamples*24)
# #             print(resultNumeric.size())
#              resultNumeric = resultNumeric.transpose(0,1).contiguous()
#              nextWord = torch.LongTensor([stoi_total.get(remainingInput[i], stoi_total["OOV"]) for _ in range(numberOfSamples*24)]).unsqueeze(0).cuda()
#              resultNumeric = torch.cat([resultNumeric[:-1], nextWord], dim=0).contiguous()
#              # get next-word surprisals using the prior
#              totalSurprisal, _, samplesFromLM, predictionsPlainLM = plain_lm.forward(resultNumeric, train=False, computeSurprisals=False, returnLastSurprisal=True, numberOfBatches=numberOfSamples*24)
##              print(totalSurprisal.size())
#              totalSurprisal = totalSurprisal.view(numberOfSamples, 24)
#              surprisalOfNextWord = totalSurprisal.exp().mean(dim=1).log().mean()
#              print("SURPRISAL", sentence, i, remainingInput[i],float( surprisalOfNextWord))
#              print(sentence, i, remainingInput[i], float( surprisalOfNextWord), file=outFile)
#
##    with open("/u/scr/mhahn/reinforce-logs-both/full-logs-tsv/"+__file__+"_"+str(args.myID)+"_"+SANITY, "w") as outFile:
##      print("Noun", "Region", "Condition", "Surprisal", "ThatFraction", file=outFile)
##      for noun in topNouns:
##       for region in ["V3", "V2", "V1", "EOS"]:
##         for condition in ["u", "g"]:
##           print(noun, region, condition, surprisalsPerNoun[noun][condition][region], thatFractionsPerNoun[noun][condition][region], file=outFile)
##    for region in ["V3", "V2", "V1", "EOS"]:
##       print(SANITY, "CORR", region, correlation(torch.FloatTensor([(float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]])) for x in topNouns]), torch.FloatTensor([surprisalsPerNoun[x]["g"][region]-surprisalsPerNoun[x]["u"][region] for x in topNouns])), correlation(torch.FloatTensor([(float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]])) for x in topNouns]), torch.FloatTensor([thatFractionsPerNoun[x]["g"][region]-thatFractionsPerNoun[x]["u"][region] for x in topNouns])))
##    overallSurprisalForCompletion = torch.FloatTensor([sum([surprisalsPerNoun[noun]["u"][region] - surprisalsPerNoun[noun]["g"][region] for region in ["V2", "V1", "EOS"]]) for noun in topNouns])
##    print(SANITY, "CORR total", correlation(torch.FloatTensor([(float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]])) for x in topNouns]), overallSurprisalForCompletion), "note this is inverted!")
#

def divideDicts(y, z):
   r = {}
   for x in y:
     r[x] = y[x]/z[x]
   return r

import scoreWithGPT2Large as scoreWithGPT2
import torch
import time
startTimePredictions = time.time()
if True:
#    print(plain_lm) 
#    topNouns = ["fact", "report"]
    with open("/u/scr/mhahn/reinforce-logs-both-short/misc/"+__file__+".txt", "w") as outFile:
     print("\t".join(["Noun", "Item", "Subject", "Compatible", "Verb", "Surprisal"]), file=outFile)
     with torch.no_grad():
      for nounIndex, NOUN in enumerate(topNouns):
        print(NOUN, "Time:", time.time() - startTimePredictions, nounIndex/len(topNouns), file=sys.stderr)
        for sentenceID in range(len(nounsAndVerbsCompatible)):
          print(sentenceID)
          context = None
          for compatible in ["compatible", "incompatible"]:
           for subject in range(2):
              if subject == 0 and nounIndex > 0:
                 continue
              sentenceList = (nounsAndVerbsCompatible if compatible == "compatible" else nounsAndVerbsIncompatible)[sentenceID]
              if subject == 1:
                context = f"The {NOUN} {sentenceList[3]}"
              else:
                context = f"The {sentenceList[0].split(' ')[1]} {sentenceList[3]}"
              print(sentenceList, subject, context)

              totalSurprisal = scoreWithGPT2.scoreSentences([context])
              surprisals_past = [x["past"] for x in totalSurprisal]
              surprisals_nextWord = [x["next"] for x in totalSurprisal]
              print("\t".join([str(q) for q in [NOUN, (sentenceList[0]+"_"+sentenceList[1]).replace("the ", ""), subject, compatible, sentenceList[3].replace(" ", "_"), float(surprisals_past[0]) + float(surprisals_nextWord[0])]]), file=outFile)

