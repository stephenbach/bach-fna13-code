#!/usr/bin/python
import math

# compute the cosine similarity between sparse document word count vectors

MIN_SIMILARITY = 0.1

trueLinks = []

for line in open('links.txt', 'r'):
	tokens = line.strip().split('\t')
	trueLinks.append((int(tokens[0]), int(tokens[1])))


fin = open('documentTFIDF.txt', 'r')

words = dict()
selfsim = dict()

for line in fin:
	tokens = line.strip().split('\t')

	page = int(tokens[0])

	vector = tokens[1].split(' ')

	words[page] = dict()

	sim = 0.0

	for value in vector:
		subtokens = value.split(':')
		word = int(subtokens[0])
		count = float(subtokens[1])

		words[page][word] = count
		sim += count * count
	selfsim[page] = sim

fin.close()

# compute true link similarities

recall = 0

trueSims = []
for (a, b) in trueLinks:
	sim = 0.0
	for word in words[a]:
		if word in words[b]:
			sim += words[a][word] * words[b][word]

	sim /= math.sqrt(selfsim[a] * selfsim[b])
	trueSims.append(sim)
	if sim > MIN_SIMILARITY:
		recall += 1
recall = float(recall) / float(len(trueLinks))

print "max similarity of true links: %f" % max(trueSims)
print "min similarity of true links: %f" % min(trueSims)
print "average similarity of true links: %f" % (sum(trueSims) / len(trueSims))
print "recall of pruning: %f" % (recall)


fout = open('candidates.txt', 'w')

for a in words:
	for b in words:
		sim = 0.0
		for word in words[a]:
			if word in words[b]:
				sim += words[a][word] * words[b][word]

		sim /= math.sqrt(selfsim[a] * selfsim[b])
		if sim > MIN_SIMILARITY:
			fout.write("%d\t%d\n" % (a, b))
fout.close()


