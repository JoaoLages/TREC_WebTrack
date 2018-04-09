import requests
import json
import os
import sys
import argparse
from utils import DIFFBOT_TOKEN
from utils.utils import read_file, read_qrels, write_file, preprocess_text
from boilerpipe.extract import Extractor
from tqdm import tqdm
import multiprocessing as mp
import time


def run_process(q_recv, q_send, in_folder, out_folder, failed_extractions_file, max_tries, use_diffbot):
    """
    Tries 'max_tries' times to extract text using
    At the end, if using diffbot, tries one last time with boilerpipe
    """

    texts, trec_ids = [], []

    def retrieve_texts_from_html(html, use_diffbot=False):
        """ Use the Diffbot API/Boilerpipe to retrieve texts from HTML """

        if use_diffbot:
            dummy_url = 'https://www.diffbot.com/dev/analytics/'
            url_api = "https://api.diffbot.com/v3/article?token=%s" \
                      "&discussion=false&url=%s" % (DIFFBOT_TOKEN, dummy_url)
            headers = {'Content-type': 'text/html'}
            content = json.loads(requests.post(url_api, data=html, headers=headers).text)

            text = content["objects"][0]["text"]
            title = content["objects"][0]["title"]

            text = '\n'.join([title, text])
        else:
            text = Extractor(extractor='ArticleExtractor', html=html).getText()

        return text

    while True:
        trec_id = q_recv.get()

        # Check end condition
        if trec_id is None:
            break

        # Check if file exists
        if not os.path.isfile("%s/%s" % (in_folder, trec_id)):
            continue

        # Read HTML
        html = read_file("%s/%s" % (in_folder, trec_id), encoding='latin1')

        i = 0
        while i != max_tries:
            try:
                texts.append(retrieve_texts_from_html(html, use_diffbot=use_diffbot))
                trec_ids.append(trec_id)
                break
            except Exception as e:  # Extraction failed
                # print(e)
                i += 1

        if i == max_tries:
            write_file("%s\n" % trec_id, failed_extractions_file, 'a')

    q_send.put((texts, trec_ids))


def retrieve_texts_from_html(html, use_diffbot=False):
    """ Use the Diffbot API/Boilerpipe to retrieve texts from HTML """

    if use_diffbot:
        dummy_url = 'https://www.diffbot.com/dev/analytics/'
        url_api = "https://api.diffbot.com/v3/article?token=%s" \
                  "&discussion=false&url=%s" % (DIFFBOT_TOKEN, dummy_url)
        headers = {'Content-type': 'text/html'}
        content = json.loads(requests.post(url_api, data=html, headers=headers).text)

        text = content["objects"][0]["text"]
        title = content["objects"][0]["title"]

        text = '\n'.join([title, text])
    else:
        text = Extractor(extractor='ArticleExtractor', html=html).getText()

    return text


def argument_parser(sys_argv):
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(
        prog='Train models',
    )
    parser.add_argument(
        '--max-tries',
        help="Max tries to extract each text from html",
        default=20,
        type=int
    )
    parser.add_argument(
        '--qrels-files',
        help="Qrels files",
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--trec-id-files',
        help="TREC id files",
        type=str,
        nargs='+'
    )
    parser.add_argument(
        '--in-folder',
        help="folder with HTML texts",
        default="DATA/corpora/dump_htmls/",
        type=str
    )
    parser.add_argument(
        '--out-folder',
        help="folder to write extracted texts",
        default="DATA/corpora/texts/",
        type=str
    )
    parser.add_argument(
        '--failed-extractions-file',
        help="File to write TREC ids of the failed extraction",
        default="failed_extractions.txt",
        type=str
    )
    parser.add_argument(
        '--re-extract',
        help="Flag used to extract files again, even if they already exist",
        default=False,
        type=bool
    )
    parser.add_argument(
        '--use-diffbot',
        help="Use Diffbot or Boilerpipe",
        default=True,
        type=bool
    )
    args = parser.parse_args(sys_argv)

    # SANITY CHECKS
    assert args.qrels_files or args.trec_id_files, \
        "Need to provide either --trec-id-files or --qrels-files"

    if args.re_extract:
        print('Re-extracting texts from HTMLs ', end='')
    else:
        print('Extracting texts from HTMLs ', end='')
    if args.use_diffbot:
        print('with Diffbot ', end='')
    else:
        print('with Boilerpipe ', end='')
    print('(%d max tries for each HTML)'
          % args.max_tries)

    return args


if __name__ == '__main__':
    # Argument handling
    args = argument_parser(sys.argv[1:])

    trec_ids = []

    # Get trec_ids from qrels
    if args.qrels_files:
        qrels = []
        for file in args.qrels_files:
            qrels += read_qrels(file)

        # Get only the TREC ids that were not extracted yet
        if not args.re_extract:
            trec_ids += [q[2] for q in qrels
                         if not os.path.isfile("%s/%s" % (args.out_folder, q[2]))]
        else:
            trec_ids += [q[2] for q in qrels]

    # Get trec_ids from trec_id_files
    if args.trec_id_files:
        ids = []
        for file in args.trec_id_files:
            ids += read_qrels(file)

        # Get only the TREC ids that were not extracted yet
        if not args.re_extract:
            trec_ids += [id[0] for id in ids
                         if not os.path.isfile("%s/%s" % (args.out_folder, id[0]))]
        else:
            trec_ids += [id[0] for id in ids]

    # Get unique ids and extract texts
    trec_ids = list(set(trec_ids))
    q_process_recv = mp.Queue(maxsize=mp.cpu_count())
    q_process_send = mp.Queue(maxsize=mp.cpu_count())
    pool = mp.Pool(
        mp.cpu_count(),
        initializer=run_process,
        initargs=(q_process_recv, q_process_send, args.in_folder, args.out_folder, args.failed_extractions_file, args.max_tries, args.use_diffbot)
    )
    for trec_id in tqdm(trec_ids, desc='Transforming HTMLs'):
        q_process_recv.put(trec_id)  # blocks until q below its max size

    # Tell workers we're done
    for _ in range(mp.cpu_count()):
        q_process_recv.put(None)

    # Receive info
    pbar = tqdm(total=len(trec_ids), ncols=100, leave=True, desc='Writing texts')
    time.sleep(0.1)  # It's how tqdm works...
    for _ in range(mp.cpu_count()):
        for text, trec_id in zip(*q_process_send.get()):
            write_file(text, '%s/%s' % (args.out_folder, trec_id))
            pbar.update(1)

    # Close pool
    pool.close()
    pool.join()

    # for trec_id in tqdm(trec_ids, total=len(trec_ids)):
    #     # Check if file exists
    #     if not os.path.isfile("%s/%s" % (args.in_folder, trec_id)):
    #         continue
    #     # Read HTML
    #     html = read_file("%s/%s" % (args.in_folder, trec_id), encoding='latin1')
    #
    #     i = 0
    #     while i != args.max_tries:
    #         try:
    #             text = retrieve_texts_from_html(html, use_diffbot=args.use_diffbot)
    #             write_file(text, '%s/%s' % (args.out_folder, trec_id))
    #             break
    #         except Exception as e:  # Extraction failed
    #             print(e)
    #             i += 1
    #
    #     # FIXME: Failed extraction
    #     if i == args.max_tries:
    #         write_file("%s\n" % trec_id, args.failed_extractions_file, 'a')

