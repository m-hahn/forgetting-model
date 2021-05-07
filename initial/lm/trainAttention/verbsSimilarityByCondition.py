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


import torch
toEmbeddings = {}
with open("/u/scr/mhahn/glove/glove.6B.300d.txt", "r") as inFile:                                                                                                                                    
   counter = 0                                                                                                                                                                                       
   for line in inFile:                                                                                                                                                                               
      counter += 1                                                                                                                                                                                   
      if counter > 500000:                                                                                                                                                                            
          break                                                                                                                                                                                      
      if len(line) < 10:                                                                                                                                                                             
          continue                                                                                                                                                                                   
      line = line.split(" ")                                                                                                                                                                         
      if True:
          if len(line) != 301:                                                                                                                                                                       
             print("ERROR", line[:5])                                                                                                                                                                
             continue                                                                                                                                                                                
          toEmbeddings[line[0]] = torch.FloatTensor([float(x) for x in line[1:]])        

with open(f"output/{__file__}.tsv", "w") as outFile:
 print("Item", "Compatible", "MatrixVerb", "InnerVerb", "Cosine", file=outFile)
 for i in range(len(nounsAndVerbsCompatible)):
    verb0 = nounsAndVerbsCompatible[i][4].split(" ")[0]
    verb1 = nounsAndVerbsCompatible[i][3].split(" ")[0]
    verb2 = nounsAndVerbsIncompatible[i][3].split(" ")[0]
    emb0 = toEmbeddings[verb0]
    emb1 = toEmbeddings[verb1]
    emb2 = toEmbeddings[verb2]
    print(i, 1, verb0, verb1,float( (emb0*emb1).mean() / math.sqrt(float((emb0*emb0).mean() * (emb1*emb1).mean()))), file=outFile)
    print(i, 0, verb0, verb2,float( (emb0*emb2).mean() / math.sqrt(float((emb0*emb0).mean() * (emb2*emb2).mean()))), file=outFile)





