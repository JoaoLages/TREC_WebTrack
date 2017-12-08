import time
import numpy as np
from math import ceil
from itertools import chain
from collections import defaultdict
from tqdm import tqdm
from utils.utils import gdeval, read_qrels
import tempfile
import subprocess


AVAILABLE_METRICS = {
    'F1_SCORE',
    'ACCURACY',
    'NDCG20',
    'ERR20'
}


def ndcg20_err20(pred, info_dict, model_config, use_invrank=False):
    # Assertion
    assert 'qids' in info_dict and 'cwids' in info_dict

    assert len(info_dict['qids']) == len(info_dict['cwids']) == len(pred), \
        "Length mismatch"

    # Build qid_cwid_pred
    qid_cwid_pred = defaultdict(dict)
    for qid, cwid, p in zip(info_dict['qids'], info_dict['cwids'], pred):
        qid_cwid_pred[qid][cwid] = p

    with tempfile.NamedTemporaryFile(mode='w', delete=True) as tmpf:
        for qid in sorted(info_dict['qids']):
            rank = 1
            for cwid in sorted(qid_cwid_pred[qid], key=lambda x: -qid_cwid_pred[qid][x]):
                if use_invrank:
                    score = 1/rank
                else:
                    score = qid_cwid_pred[qid][cwid]
                tmpf.write('%d Q0 %s %d %.10e %s\n' % (qid, cwid, rank, score, model_config['name']))
                rank += 1
        tmpf.flush()
        val_res = subprocess.check_output([gdeval, '-k', '20', model_config['qrel_file'], tmpf.name]).decode('utf-8')
    amean_line = val_res.splitlines()[-1]
    cols = amean_line.split(',')
    ndcg20, err20 = float(cols[-2]), float(cols[-1])
    return ndcg20, err20


def f1_score(gold, pred):
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)

    # Initialize
    tp = 0.
    fp = 0.
    fn = 0.

    for g, p in zip(gold, pred):
        if g == p:
            if p == 1:
                tp += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1

    if tp == 0:
        return 0.

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return 2.*precision*recall/(precision+recall)


def get_metric_scores(metrics, meta_data, predicted_probs, gold_tags, model_config):

    metric_scores, metric_names = [], []

    info_dict = defaultdict(list)
    for m in meta_data:
        for key in ['qids', 'cwids']:
            try:
                info_dict[key].extend(m[key])
            except KeyError:
                raise Exception("NDCG20/ERR20 needs 'qids' and 'cwids' in meta-data")

    if 'rerank_files' in model_config:
        # Get all query-document pairs in QREL file
        qid_cwid2pred = {(x, y): z for x, y, z in zip(info_dict['qids'], info_dict['cwids'], predicted_probs)}

        # Build rerank_dict like info_dict and new predicted_probs
        rerank_dict = defaultdict(lambda: defaultdict(list))
        new_probs = defaultdict(list)
        for key in model_config['rerank_files']:
            # Get qids and cwids in common
            lines = read_qrels(model_config['rerank_files'][key])
            for line in lines:
                if (int(line[0]), line[2]) in qid_cwid2pred:
                    # In common, append prediction
                    rerank_dict[key]['qids'].append(int(line[0]))
                    rerank_dict[key]['cwids'].append(line[2])
                    new_probs[key].append(qid_cwid2pred[(int(line[0]), line[2])])
                else:
                    # Not in common, just add -inf to predictions
                    rerank_dict[key]['qids'].append(int(line[0]))
                    rerank_dict[key]['cwids'].append(line[2])
                    new_probs[key].append(float('-inf'))

            # Get NDCG and ERR
            ndcg20, err20 = ndcg20_err20(new_probs[key], rerank_dict[key], model_config, use_invrank=True)

            # Add scores
            metric_scores.append(ndcg20)
            metric_names.append('NDCG20-%s' % key)
            metric_scores.append(err20)
            metric_names.append('ERR20-%s' % key)

    if 'ACCURACY' in metrics:
        correct = sum([1. for p, g in zip(predicted_probs, gold_tags)
                       if round(p) == g])
        metric_scores.append(correct / len(gold_tags))
        metric_names.append('ACCURACY')

    if 'F1_SCORE' in metrics:
        # FIXME, expected tags not probs
        metric_scores.append(f1_score(gold_tags, predicted_probs))
        metric_names.append('F1_SCORE')

    if 'NDCG20' in metrics or 'ERR20' in metrics:
        ndcg20, err20 = ndcg20_err20(predicted_probs, info_dict, model_config)

    if 'NDCG20' in metrics:
        metric_scores.append(ndcg20)
        metric_names.append('NDCG20')

    if 'ERR20' in metrics:
        metric_scores.append(err20)
        metric_names.append('ERR20')

    if not metric_names:
        raise Exception("No metric to evaluate model")
    for m in metrics:
        if m not in metric_names:
            print("%s is an invalid metric" % m)

    return metric_scores, metric_names


def color(x, color_select):

    colors = {
        'green': '\033[1;32m',
        'yellow': '\033[1;33m',
        'red': '\033[1;31m',
        'end': '\033[0;0m'
    }

    if isinstance(x, float):
        return '%s%2.3f%s' % (colors[color_select], x, colors['end'])
    elif isinstance(x, int):
        return '%s%d%s' % (colors[color_select], x, colors['end'])
    else:
        return '%s%s%s' % (colors[color_select], x, colors['end'])


class TrainLogger():

    def __init__(self, config=None):
        self.config = config
        self.state = None
        self.metrics = {m: [] for m in [config['monitoring_metric']] + config['metrics']}

        self.best_monitoring_metric = 0.
        self.epoch = 0
        self.best_epoch = 0
        self.n_samples = None  # update_nrsamples will have to be called
        self.b_size = config['batch_size']
        self.pbar = None
        self.init_time = time.time()

    def reset_logger(self, n_samples):
        self.n_samples = n_samples
        if self.b_size:
            self.n_batches = ceil(self.n_samples / self.b_size)
        else:
            self.n_batches = 1
        self.batch_idx = 0

    def update_on_batch(self, objective):
        if objective:
            self.batch_idx += 1

            if self.batch_idx == 1:
                self.pbar = tqdm(total=self.n_samples, ncols=100, leave=True)
                time.sleep(0.1)  # It's how tqdm works...

            self.pbar.set_description(
                'Epoch %i | Batch %i/%i' %
                (self.epoch + 1, self.batch_idx, self.n_batches)
            )
            self.pbar.set_postfix(loss=objective['loss'])
            self.pbar.update(self.b_size)

    def update_on_epoch(self, predictions, gold, meta_data, model_config):

        # Reset save state
        self.state = None

        if self.pbar:
            self.pbar.close()
            self.batch_idx = 0

        def flatten_list(items):
            if isinstance(items[0], list):
                return list(chain.from_iterable(items))
            else:
                return items

        # Flatten
        predicted_probs = np.array([
            y
            for batch in predictions
            for y in flatten_list(batch['probs'])
        ]).astype(int)

        if 'binary_tags' in model_config:
            gold_tags = np.array([
                x
                for batch in gold
                for x in flatten_list(batch['binary_tags'])
            ]).astype(int)
        else:
            gold_tags = np.array([
                x for batch in gold for x in flatten_list(batch['tags'])
            ]).astype(int)

        metric_scores, metric_names = \
            get_metric_scores(self.metrics, meta_data, predicted_probs, gold_tags, model_config)

        # Update state
        self.state = None
        for m_score, m_name in zip(metric_scores, metric_names):
            self.metrics[m_name].append(m_score)
        self.epoch += 1

        color_select = 'red'
        if self.metrics[self.config['monitoring_metric']][-1] > \
                self.best_monitoring_metric:
            self.state = 'save'
            self.best_monitoring_metric = self.metrics[self.config['monitoring_metric']][-1]
            self.best_epoch = self.epoch
            color_select = 'green'

        epoch_time = (time.time() - self.init_time)

        # Inform user
        print("Epoch %d |" % self.epoch, end='')
        for m_name in metric_names:
            print(
                "%s %s |" %
                (m_name, color(self.metrics[m_name][-1], color_select)),
                end=''
            )
        print(
            "Best %s: %s at epoch %d |" % (
                self.config['monitoring_metric'],
                color(self.best_monitoring_metric, 'yellow'),
                self.best_epoch
            ),
            end=''
        ),
        print("Time elapsed: %d seconds" % epoch_time)

        self.init_time = time.time()
