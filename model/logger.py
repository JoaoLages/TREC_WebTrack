from math import ceil
from itertools import chain, product
from collections import defaultdict
from tqdm import tqdm
from utils.utils import gdeval, read_qrels
from plotly.offline import plot
import plotly.graph_objs as go
import tempfile
import subprocess
import time
import numpy as np


AVAILABLE_METRICS = {
    'F1_SCORE',
    'ACCURACY',
    'NDCG20',
    'ERR20'
}

judgments_to_label = {'Nav': 4, 'HRel': 2, 'Rel': 1, 'NRel': 0, 'Junk': -2}
labels_to_judgement = {4: 'Nav', 2: 'HRel', 1: 'Rel', 0: 'NRel', -2: 'Junk'}
trec_year_to_judgments = {
    'wt09': {2: 'HRel', 1: 'Rel', 0: 'NRel'},
    'wt10': {4: 'Nav', 3: 'HRel', 2: 'HRel', 1: 'Rel', 0: 'NRel', -2: 'Junk'},
    'wt11': {3: 'Nav', 2: 'HRel', 1: 'Rel', 0: 'NRel', -2: 'Junk'},
    'wt12': {4: 'Nav', 3: 'HRel', 2: 'HRel', 1: 'Rel', 0: 'NRel', -2: 'Junk'},
    'wt13': {4: 'Nav', 3: 'HRel', 2: 'HRel', 1: 'Rel', 0: 'NRel', -2: 'Junk'},
    'wt14': {4: 'Nav', 3: 'HRel', 2: 'HRel', 1: 'Rel', 0: 'NRel', -2: 'Junk'}
}


def trec_qid_to_year(qid):
    if int(qid) <= 50:
        return 'wt09'
    elif int(qid) <= 100:
        return 'wt10'
    elif int(qid) <= 150:
        return 'wt11'
    elif int(qid) <= 200:
        return 'wt12'
    elif int(qid) <= 250:
        return 'wt13'
    else:
        return 'wt14'


def create_docpairs(pred, gold, info_dict):
    # Assertion
    assert 'qids' in info_dict and 'cwids' in info_dict

    assert len(info_dict['qids']) == len(info_dict['cwids']) == len(pred) == len(gold), \
        "Length mismatch"

    # Build qid_cwid_labelinvrank
    qid_cwid_labelinvrank = defaultdict(dict)
    rank = 1
    for qid, cwid, g, _ in sorted(zip(info_dict['qids'], info_dict['cwids'], gold, pred),
                                  key=lambda x: x[3]):
        qid_cwid_labelinvrank[qid][cwid] = (g, 1/rank)  # (label, invrank)
        rank += 1

    pkey_docpairs = defaultdict(list)
    for qid in qid_cwid_labelinvrank:
        label_cwids = defaultdict(list)
        year = trec_qid_to_year(qid)

        # Transform label and save cwids in label_cwids
        for cwid in qid_cwid_labelinvrank[qid]:
            label = qid_cwid_labelinvrank[qid][cwid][0]
            jud = trec_year_to_judgments[year][label]
            label = judgments_to_label[jud]
            label_cwids[label].append(cwid)

        labels = list(label_cwids.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ll, lh = min(labels[i], labels[j]), max(labels[i], labels[j])
                dls, dhs = label_cwids[ll], label_cwids[lh]
                pairkey = '%s-%s' % (labels_to_judgement[lh], labels_to_judgement[ll])
                for dl, dh in product(dls, dhs):
                    pkey_docpairs[pairkey].append((qid, dl, dh))

    pkey_qidcount = defaultdict(dict)
    pkey_qid_acc = defaultdict(dict)
    for pkey in pkey_docpairs:
        qid_dl_dh = pkey_docpairs[pkey]
        for qid, dl, dh in qid_dl_dh:
            if qid not in pkey_qidcount[pkey]:
                pkey_qidcount[pkey][qid] = [0, 0]  # [correct, total]
            if qid_cwid_labelinvrank[qid][dl][1] < qid_cwid_labelinvrank[qid][dh][1]:
                pkey_qidcount[pkey][qid][0] += 1
            pkey_qidcount[pkey][qid][1] += 1

    for pkey in pkey_qidcount:
        accs = list()
        total_all = 0
        for qid in pkey_qidcount[pkey]:
            correct, total = pkey_qidcount[pkey][qid]
            total_all += total
            acc = correct / total
            pkey_qid_acc[pkey][qid] = acc
            accs.append(acc)
        pkey_qid_acc[pkey][0] = np.mean(accs)
        pkey_qid_acc[pkey][-1] = total_all

    return pkey_qid_acc


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


def get_metric_scores(metrics, meta_data, predicted_probs, gold_tags, model_config, keep_non_overlaps=False):

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
                    if keep_non_overlaps:
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
        self.loss_history = []
        self.current_loss = None

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
            self.current_loss = objective['loss']
            self.pbar.set_postfix(loss=self.current_loss)
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

        # Get metric scores
        metric_scores, metric_names = \
            get_metric_scores(self.metrics, meta_data, predicted_probs, gold_tags, model_config)

        # Save metrics and loss
        for m_score, m_name in zip(metric_scores, metric_names):
            self.metrics[m_name].append(m_score)
        self.loss_history.append(self.current_loss)
        self.epoch += 1

        color_select = 'red'
        if self.metrics[self.config['monitoring_metric']][-1] > self.best_monitoring_metric:
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

    def plot_curve(self, path):
        plots = []
        # Add metrics
        for metric in self.metrics:
            plots.append(
                go.Scatter(
                    x=list(range(1, len(self.metrics[metric])+1)),
                    y=self.metrics[metric],
                    mode='lines+markers',
                    name='%s (%s at epoch %s)' %
                         (metric, max(self.metrics[metric]), np.argmax(np.asarray(self.metrics[metric])) + 1)
                )
            )
        # Add loss
        plots.append(
            go.Scatter(
                x=list(range(1, len(self.loss_history) + 1)),
                y=self.loss_history,
                mode='lines+markers',
                name='Loss (%s at epoch %s)' %
                     (min(self.loss_history), np.argmin(np.asarray(self.loss_history)) + 1)
            )
        )
        # Save graph in html file
        filename = "Loss_and_metrics_evolution"
        layout = go.Layout(title='Loss and metrics evolution')
        _ = plot(go.Figure(data=plots, layout=layout), image='png', image_filename=filename,
                 filename="%s/%s.html" % (path, filename), show_link=False, auto_open=False)
