from warc.warc import WARCFile
import os
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from utils.utils import write_file, read_qrels


EXTRACT09 = True
EXTRACT12 = False
RE_EXTRACT = True
MAX_TRIES = 20

NR_DISK_CLUEWEB12 = 3
disk2id_map = {
    1: {0, 1, 2, 3, 4},
    2: {5, 6, 7, 8, 9},
    3: {10, 11, 12, 13, 14},
    4: {15, 16, 17, 18, 19}
}
CLUEWEB12_DIR = '/media/joaolages/ClueWeb12_%sof4' % NR_DISK_CLUEWEB12
CLUEWEB12_QRELS = ['qrels/2013qrels.adhoc.txt', 'qrels/2014qrels.adhoc.txt']
CLUEWEB09_DIR = '/media/joaolages/CLUEWEB09_B/ClueWeb09_English_1'
CLUEWEB09_QRELS = ['qrels/2009qrels.adhoc.txt', 'qrels/2010qrels.adhoc.txt',
                   'qrels/2011qrels.adhoc.txt', 'qrels/2012qrels.adhoc.txt']
DUMP_FOLDER = 'DATA/corpora/dump_htmls'


def read_and_write_html_from_warc(q, file2ids, output_folder):

    while True:
        info = q.get()

        # Check end condition
        if info is None:
            break

        file, file_id = info

        # Check if file exists
        if not os.path.isfile(file):
            continue

        warc_trec_ids = set(file2ids[file_id])

        i = 0
        while warc_trec_ids and i != MAX_TRIES:
            fp = WARCFile(file, "rb")

            first_record = True
            for record in fp:
                if not first_record:
                    if record['warc-trec-id'] in warc_trec_ids:  # Found record
                        # Encode
                        try:
                            text = record.payload.encode('utf-8')
                        except UnicodeDecodeError:
                            text = record.payload

                        write_file(text, '%s/%s' % (output_folder, record['warc-trec-id']), encoding=None)
                        warc_trec_ids.remove(record['warc-trec-id'])
                else:
                    first_record = False
            i += 1
        if warc_trec_ids:
            write_file(warc_trec_ids, 'failed_extracted_files.txt', mode='a')


# Sanity checks
assert EXTRACT12 != EXTRACT09, 'One dataset at a time'

# Get qrels
qrels = []
if EXTRACT09:
    for f in CLUEWEB09_QRELS:
        qrels += read_qrels(f)
if EXTRACT12:
    for f in CLUEWEB12_QRELS:
        qrels += read_qrels(f)

# Gather ids in dict to grab more than 1 html per document at a time
file2ids = defaultdict(list)
for qrel in qrels:
    trec_id = qrel[2]
    # File does not exist yet
    if not os.path.isfile("%s/%s" % (DUMP_FOLDER, trec_id)) or RE_EXTRACT:
        file_id = '-'.join(trec_id.split('-')[1:3])  # file identifier
        if EXTRACT12:
            if int(file_id[0:2]) in disk2id_map[NR_DISK_CLUEWEB12]:
                file2ids[file_id].append(trec_id)
        else:
            file2ids[file_id].append(trec_id)

# Initialize Pool
q = mp.Queue(maxsize=mp.cpu_count())
pool = mp.Pool(
    mp.cpu_count(),
    initializer=read_and_write_html_from_warc,
    initargs=(q, file2ids, DUMP_FOLDER)
)
for file_id in tqdm(file2ids, total=len(file2ids)):

    if EXTRACT12:
        file_name = '%s.warc.gz' % file_id
        path = '%s/ClueWeb12_%s/%s/%s' % (CLUEWEB12_DIR, file_id[0:2],
                                          file_id.split('-')[0], file_name)
    else:
        file_name = '%s.warc.gz' % file_id.split('-')[1]
        path = '%s/%s/%s' % (CLUEWEB09_DIR, file_id.split('-')[0], file_name)

    q.put((path, file_id))  # blocks until q below its max size

# Tell workers we're done
for _ in range(mp.cpu_count()):
    q.put(None)

# Close pool
pool.close()
pool.join()
