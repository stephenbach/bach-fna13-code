#!/usr/bin/python
import math

# compute the cosine similarity between sparse document word count vectors

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

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

recall = dict()
for threshold in thresholds:
	recall[threshold] = 0

trueSims = []
for (a, b) in trueLinks:
	sim = 0.0
	for word in words[a]:
		if word in words[b]:
			sim += words[a][word] * words[b][word]

	sim /= math.sqrt(selfsim[a] * selfsim[b])
	trueSims.append(sim)

	for threshold in thresholds:
		if sim > threshold:
			recall[threshold] += 1

print "max similarity of true links: %f" % max(trueSims)
print "min similarity of true links: %f" % min(trueSims)
print "average similarity of true links: %f" % (sum(trueSims) / len(trueSims))
for threshold in thresholds:
	print "recall of pruning at %f: %f" % (threshold, float(recall[threshold]) / float(len(trueLinks)))


fout = dict()
for threshold in thresholds:
	fout[threshold] = open('candidates.%1.1f.txt' % threshold, 'w')

total = len(words)
count = 1

wordPairs = words.items()

for i in range(len(wordPairs)):
	(a, wordsA) = wordPairs[i]
	for j in range(i, len(wordPairs)):
		(b, wordsB) = wordPairs[j]
		sim = 0.0
		for word in wordsA:
			if word in wordsB:
				sim += wordsA[word] * wordsB[word]

		sim /= math.sqrt(selfsim[a] * selfsim[b])

		for threshold in thresholds:
			if sim > threshold:
				fout[threshold].write("%d\t%d\n" % (a, b))
				if a != b:
					fout[threshold].write("%d\t%d\n" % (b, a))
	if count % 10 == 0:
		print "Finished pruning %d of %d total nodes" % (count, total)
	count += 1

for threshold in thresholds:
	fout[threshold].close()


