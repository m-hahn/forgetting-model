with open("/u/scr/mhahn/Dundee/DundeeTreebankTokenized.csv", "r") as inFile:
   dundee = [x.split("\t") for x in inFile.read().strip().split("\n")]
   header = dundee[0]
   headerList = header
   header = dict(zip(header, list(range(len(header)))))
   dundee = dundee[1:]


calibrationSentences = []

lastWNUM = -1
tokenInWord = 0
with open("/u/scr/mhahn/Dundee/DundeeTreebankTokenized2.csv", "w") as outFile:
    print("\t".join(headerList + ["TokenInWord", "Capitalized"]), file=outFile)
    for i in range(len(dundee)):
        line = dundee[i]
        Itemno, WNUM, SentenceID, ID, WORD, Token = line
        Itemno = (Itemno)
        if False: #i > 0 and Itemno == dundee[i-1][header["Itemno"]] and ID == dundee[i-1][header["ID"]]:
            continue
        else:
            if WNUM == lastWNUM:
              tokenInWord += 1
            else:
              tokenInWord = 1
            print("\t".join([Itemno, WNUM, SentenceID, ID, WORD, Token, str(tokenInWord), str(WORD != WORD.lower()).upper()]), file=outFile)
            lastWNUM = WNUM
    print(calibrationSentences[-1:])
    #quit()
    
    
