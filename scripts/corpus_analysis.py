from utils import read_qrels
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


# ADHOC TASK
frequencies = \
    get_train_test_frequencies(['qrels/2013qrels.adhoc.txt', 'qrels/2014qrels.adhoc.txt'])
print("Adhoc Retrieval task: ")
print("Train frequencies: %s" % frequencies[0])
print("Test frequencies: %s\n" % frequencies[1])

# Risk sensitive TASK
frequencies = \
    get_train_test_frequencies(['qrels/2013qrels.all.txt', 'qrels/2014qrels.all.txt'])
print("Risk sensitive task: ")
print("Train frequencies: %s" % frequencies[0])
print("Test frequencies: %s\n" % frequencies[1])

# Risk sensitive TASK - partioned
frequencies = \
    get_train_test_frequencies(['qrels/2013train.txt', 'qrels/2013dev.txt', 'qrels/2014qrels.all.txt'])
print("Risk sensitive task (3 partitions): ")
print("Train frequencies: %s" % frequencies[0])
print("Dev frequencies: %s" % frequencies[1])
print("Test frequencies: %s\n" % frequencies[2])

# Risk sensitive TASK - partioned (ALL)
frequencies = \
    get_train_test_frequencies(['qrels/all-train.txt', 'qrels/all-dev.txt', 'qrels/all-test.txt'])
print("Risk sensitive task (3 partitions - ALL): ")
print("Train frequencies: %s" % frequencies[0])
print("Dev frequencies: %s" % frequencies[1])
print("Test frequencies: %s\n" % frequencies[2])

# Risk sensitive TASK - subset
frequencies = \
    get_train_test_frequencies(['qrels/trainqrels.all.txt', 'qrels/devqrels.all.txt',
                                'qrels/testqrels.all.txt'])
print("Risk sensitive (example) task: ")
print("Train frequencies: %s" % frequencies[0])
print("Dev frequencies: %s" % frequencies[1])
print("Test frequencies: %s" % frequencies[2])
