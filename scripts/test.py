import os
import sys
import argparse
import numpy as np
from data.data_iterator import Data
from model.model_interface import ModelInterface
from utils.utils import parse_config, edit_config
from model.logger import AVAILABLE_METRICS, evaluate


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
        default=None,
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
    parser.add_argument('--cnn-out', action='store_true')
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

    # Remove train and dev from data config
    config['datasets'] = {
        dset: config['datasets'][dset]
        for dset in config['datasets']
        if dset not in ['train', 'dev']
    }

    # Force test to run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    return config, args, qrel_file_flag


if __name__ == '__main__':

    # Argument handling
    data_config, args, qrel_file_flag = argument_parser(sys.argv[1:])

    total_models = ["%s/%s" % (args.model_folder, x) for x in os.listdir(args.model_folder) if '.h5' in x]

    if args.cnn_out:
        assert len(total_models) == 1

    # Init model and total_pred
    model = ModelInterface(model_folder=total_models[0])
    total_pred = None
    for i, model_file in enumerate(total_models):
        # Load model
        model.load(model_file)

        # Load data
        if i == 0:
            # Get parameters
            model_parameters = model.get('p')

            # Pass some keys of model_config to data_config
            data_config['sim_matrix_config'] = model_parameters['sim_matrix_config']
            data_config['query_idf_config'] = model_parameters['query_idf_config']
            data_config['num_negative'] = model_parameters['num_negative']
            data_config['use_description'] = model_parameters['use_description']
            data_config['use_topic'] = model_parameters['use_topic']
            data_config['custom_loss'] = model_parameters['custom_loss']

            # Load data
            data = Data(config=data_config)

            if args.datasets:
                dsets = args.datasets
            else:
                dsets = ['test']

        for dset in dsets:
            if i == 0:
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

                    # Delete rerank files
                    data_config['datasets'][dset] = qrel_file

                # Get batches
                set_features = data.batches(
                    [dset],
                    batch_size=model.config['batch_size'],
                    features_model=model.get_features
                )

            # Predict
            predictions, cnn_outs = [], []
            if i == 0:
                gold = []
                meta_data = []
            for batch in set_features:
                predictions.extend(model.predict(batch['input']))
                if args.cnn_out:
                    cnn_outs.append(model.get_kmax_input(batch['input']))
                if i == 0:
                    gold.append(batch['output'])
                    if 'meta-data' in batch['input']:
                        meta_data.append(batch['input']['meta-data'])

            if total_pred is not None:
                total_pred += np.array(predictions) / len(total_models)
            else:
                total_pred = np.array(predictions) / len(total_models)

    # Add additional info to model config
    if qrel_file_flag:
        model.config['qrel_file'] = qrel_file[0]
        model.config['rerank_files'] = rerank_files

    # Evaluate
    if args.metrics:
        print("%s%s%s" % ('\033[1;33m', " ".join(data_config['datasets'][dset]), '\033[0;0m'))
        metric_scores, metric_names = evaluate(total_pred, gold, args.metrics, meta_data, model.config)
        for m_score, m_name in zip(metric_scores, metric_names):
            print("%s: %2.4f" % (m_name, m_score))

    # Save midlayer outputs
    if args.cnn_out:
        import pickle
        with open('cnn_outputs.pkl', 'wb') as f:
            pickle.dump([cnn_outs, meta_data], f)

    # Save results
    if args.results_folder:
        file_path = "%s/%s.probs" % (args.results_folder, dset)
        with open(file_path, 'w') as fid:
            for x in predictions:
                fid.write('%1.5f\n' % x)
        print("Wrote %s" % file_path)
