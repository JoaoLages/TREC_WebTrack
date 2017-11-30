import os
import sys
import argparse
import numpy as np
from data.data_iterator import Data
from model.model_interface import ModelInterface
from utils.utils import parse_config, edit_config
from itertools import chain
from model.logger import AVAILABLE_METRICS, get_metric_scores


def evaluate(predicted, gold, metrics, meta_data, model_config):
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

    metric_scores, metric_names = get_metric_scores(metrics, meta_data, predicted_probs, gold_tags, model_config)

    for m_score, m_name in zip(metric_scores, metric_names):
        print("%s: %2.4f" % (m_name, m_score))


def argument_parser(sys_argv):
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(
        prog='Test models'
    )

    parser.add_argument(
        '--data-config',
        help="configuration file path",
        required=True,
        type=str
    )
    parser.add_argument(
        '--model-folder',
        help="Folder where the model is available",
        required=True,
        type=str
    )
    parser.add_argument(
        '--results-folder',
        help="where to store probabilities of each class",
        type=str
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        help="Metrics solicited",
        type=str
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        help="Datasets to be evaluated",
        default=None,
        type=str
    )
    parser.add_argument(
        '--overload',
        help="Pairs of parameters to overload",
        nargs='+',
        type=str
    )
    args = parser.parse_args(sys_argv)

    if args.metrics:
        assert all(metric in AVAILABLE_METRICS for metric in args.metrics), \
            "Supported metrics %s" % (" ".join(AVAILABLE_METRICS))

    if 'NDCG20' in args.metrics or 'ERR20' in args.metrics:
        qrel_file_flag = True
    else:
        qrel_file_flag = False

    config = parse_config(args.data_config)
    if args.overload:
        config = edit_config(config, args.overload)

    # Remove train from data config
    config['datasets'] = {
        dset: config['datasets'][dset]
        for dset in config['datasets']
        if dset != 'train'
    }

    # Force test to run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    return config, args, qrel_file_flag


if __name__ == '__main__':

    # Argument handling
    data_config, args, qrel_file_flag = argument_parser(sys.argv[1:])

    # Load model
    model = ModelInterface(model_folder=args.model_folder)

    # Get parameters
    model_parameters = model.get('p')

    # Pass sim_matrix_config, query_idf_config and num_negative to data_config
    data_config['sim_matrix_config'] = model_parameters['sim_matrix_config']
    data_config['query_idf_config'] = model_parameters['query_idf_config']
    data_config['num_negative'] = model_parameters['num_negative']

    # Load data
    data = Data(config=data_config)

    if args.datasets:
        dsets = args.datasets
    else:
        dsets = ['dev', 'test']

    for dset in dsets:
        if qrel_file_flag:
            # For TREC qrel file and rerank files
            qrel_file, rerank_files = [], {}
            for file in data_config['datasets']['%s' % dset]:
                if not isinstance(file, dict):
                    qrel_file.append(file)
                else:
                    for key in file:
                        rerank_files[key] = file[key]

            assert len(qrel_file) == 1, "Only provide one QREL file for %s" % dset
            model.config['qrel_file'] = qrel_file[0]
            model.config['rerank_files'] = rerank_files

            # Delete rerank files
            data_config['datasets'][dset] = qrel_file

        # Train
        set_features = data.batches(
            dset,
            batch_size=model.config['batch_size'],
            features_model=model.get_features
        )

        # Predict
        predictions = []
        gold = []
        meta_data = []
        for batch in set_features:
            predictions.append(model.predict(batch['input']))
            gold.append(batch['output'])
            if 'meta-data' in batch['input']:
                meta_data.append(batch['input']['meta-data'])

        # Evaluate
        if args.metrics:
            print("%s%s%s" % ('\033[1;33m', " ".join(data_config['datasets'][dset]), '\033[0;0m'))
            evaluate(predictions, gold, args.metrics, meta_data, model.config)

        # Save results
        if args.results_folder:
            file_path = "%s/%s.probs" % (args.results_folder, dset)
            with open(file_path, 'w') as fid:
                for res in predictions:
                    for x in res['probs']:
                        fid.write('%1.5f\n' % x)
            print("Wrote %s" % file_path)
