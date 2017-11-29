import requests
import json
import os
import sys
import argparse
import threading
import time
from utils import DIFFBOT_TOKEN
from utils.utils import read_file, read_qrels, write_file, preprocess_text
from boilerpipe.extract import Extractor
from tqdm import tqdm


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


def run_thread(html, folder, trec_id, failed_extractions_file, max_tries=5, use_diffbot=False):
    """
    Tries 'max_tries' times to extract text using
    At the end, if using diffbot, tries one last time with boilerpipe
    """
    i = 0
    while i != max_tries:
        try:
            text = retrieve_texts_from_html(html, use_diffbot=use_diffbot)
            write_file(text, '%s/%s' % (folder, trec_id))
            return
        except:  # Extraction failed
            i += 1

    if use_diffbot:
        # Try one last time with Boilerpipe
        try:
            text = retrieve_texts_from_html(html, use_diffbot=False)
            write_file(text, '%s/%s' % (folder, trec_id))
            return
        except:
            pass

    # FIXME: Failed extraction
    write_file("%s\n" % trec_id, failed_extractions_file, 'a')


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
        '--thread-limit',
        help="Max number of threads",
        default=250,
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
    print('(%d max tries for each HTML and %d max threads running)'
          % (args.max_tries, args.thread_limit))

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

    # Get unique ids
    trec_ids = list(set(trec_ids))

    for trec_id in tqdm(trec_ids, total=len(trec_ids)):
        # Read HTML
        html = read_file("%s/%s" % (args.in_folder, trec_id), mode='rb', encoding=None)

        # Wait loop
        while threading.active_count() >= args.thread_limit:
            time.sleep(2)  # Sleep for 2 seconds
            pass

        # Call thread
        threading.Thread(
            target=run_thread,
            args=(html, args.out_folder, trec_id, args.failed_extractions_file, args.max_tries, args.use_diffbot)
        ).start()
