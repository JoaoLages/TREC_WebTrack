from utils.utils import read_qrels
from collections import Counter


def get_train_test_frequencies(files):

    frequencies = []
    for file in files:
        qrels = read_qrels(file)

        tags = [qrel[3] for qrel in qrels]

        tags_frequency = Counter(tags)

        # Divide by total len to get percentage
        for tag in tags_frequency:
            tags_frequency[tag] /= 1.*len(tags)

        frequencies.append(tags_frequency)

    return frequencies


qrel_files = ['qrels/2009qrels.adhoc.txt', 'qrels/2010qrels.adhoc.txt', 'qrels/2011qrels.adhoc.txt',
              'qrels/2012qrels.adhoc.txt', 'qrels/2013qrels.adhoc.txt', 'qrels/2014qrels.adhoc.txt']
frequencies = get_train_test_frequencies(qrel_files)

print("Adhoc Retrieval task: ")
for file, freq in zip(qrel_files, frequencies):
    print("%s frequencies: %s" % (file, freq))
