from warc.warc import *
import os
import threading
from collections import defaultdict
from tqdm import tqdm
from utils.utils import write_file, read_qrels

THREAD_LIMIT = 100
MAX_TRIES = 20
NR_DISK_CLUEWEB12 = 3
disk2id_map = {
    1: {00, 01, 02, 03, 04},
    2: {05, 06, 07, 8, 9},
    3: {10, 11, 12, 13, 14},
    4: {15, 16, 17, 18, 19}
}
CLUEWEB12_DIR = '/media/joaolages/ClueWeb12_%sof4' % NR_DISK_CLUEWEB12
DUMP_FOLDER = 'corpora/dump'


def read_and_write_html_from_warc(file, warc_trec_ids, output_folder):

    i = 0
    while warc_trec_ids and i!=MAX_TRIES:
        fp = WARCFile(file, "rb")

        first_record = True
        for record in fp:
            if not first_record:
                if record['warc-trec-id'] in warc_trec_ids:  # Found record
                    write_file(record.payload, '%s/%s' % (output_folder, record['warc-trec-id']), mode='wb')
                    warc_trec_ids.remove(record['warc-trec-id'])
            else:
                first_record = False
        i+=1
    if warc_trec_ids:
        write_file(warc_trec_ids, 'failed_extracted_files.txt', mode='a')

#if train:
    # Train
qrels = read_qrels('2013qrels.all.txt')
#else:
    # Test
qrels += read_qrels('2014qrels.all.txt')

# Gather ids in dict to grab more than 1 html per document at a time
file2ids = defaultdict(list)
for qrel in qrels:
    trec_id = qrel[2]
    # File does not exist yet
    if not os.path.isfile("%s/%s" % (DUMP_FOLDER, trec_id)):
        file_id = '-'.join(trec_id.split('-')[1:3])  # file identifier
        if int(file_id[0:2]) in disk2id_map[NR_DISK_CLUEWEB12]:
            file2ids[file_id].append(trec_id)

for file_id in tqdm(file2ids, total=len(file2ids)):
    file_name = '%s.warc.gz' % file_id
    path = '%s/ClueWeb12_%s/%s/%s' % (CLUEWEB12_DIR, file_id[0:2],
                                      file_id.split('-')[0], file_name)

    while threading.active_count() >= THREAD_LIMIT:
        pass
    threading.Thread(
        target=read_and_write_html_from_warc, args=(path, set(file2ids[file_id]), DUMP_FOLDER)
    ).start()
