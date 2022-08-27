header = ["token","logprob","offset","line"]

stimuli = []
with open("stimuli.txt") as inFile:
    try:
      while True:
        item = [None]
        stimuli.append((next(inFile).strip().split(" "), "SC", "compatible", [[]], item))
        stimuli.append((next(inFile).strip().split(" "), "SCRC", "compatible", [[]], item))
        stimuli.append((next(inFile).strip().split(" "), "SC", "incompatible", [[]], item))
        stimuli.append((next(inFile).strip().split(" "), "SCRC", "incompatible", [[]], item))
        item[0] = stimuli[-1][0][4] + "_" + stimuli[-1][0][7]
    except StopIteration:
        pass

print(stimuli)
with open("stimuli_gpt3_20210422.csv", "r") as inFile:
    for line in inFile:
        if len(line) < 3:
            continue
        line = line.rstrip("\n").split(",")
        assert len(line) == 4
        if line[0].startswith(" "):
            stimuli[int(line[3])][3].append([])
        stimuli[int(line[3])][3][-1].append(line)


print(stimuli)

with open("gpt3.tsv", "w") as outFile:
  for stimulus in stimuli:
#    print(stimulus)
    assert len(stimulus[0]) == len(stimulus[3])
    for i in range(len(stimulus[0])):
        if stimulus[0][i].startswith("#"):
#            print(stimulus[0][i], stimulus[3][i])
            logprob = sum([float(x[1]) for x in stimulus[3][i]])
            print("\t".join([str(q) for q in [stimulus[4][0], stimulus[1],stimulus[2], logprob, "".join([x[0] for x in stimulus[3][i]]).strip()]]), file=outFile)
#    quit()



