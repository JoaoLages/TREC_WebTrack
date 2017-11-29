import sys
import argparse
from data.data_iterator import Data
from model.model_interface import ModelInterface
from model.logger import TrainLogger, AVAILABLE_METRICS
from utils.utils import parse_config, edit_config
from math import ceil
import os


def argument_parser(sys_argv):
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(
        prog='Train models',
    )
    parser.add_argument(
        '--data-config',
        help="Data configuration file path",
        required=True,
        type=str
    )
    parser.add_argument(
        '--model-config',
        help="Model configuration file path",
        required=True,
        type=str
    )
    parser.add_argument(
        '--model-folder',
        help="Will overload model-config's variable",
        type=str
    )
    parser.add_argument(
        '--metrics',
        help="Metrics to calculate while training model",
        default=['ACCURACY'],
        nargs='+',
        type=str
    )
    parser.add_argument(
        '--overload',
        help="Pairs of parameters to overload",
        nargs='+',
        type=str
    )
    args = parser.parse_args(sys_argv)

    data_config = parse_config(args.data_config)
    model_config = parse_config(args.model_config)

    # Retain only train and dev
    data_config['datasets'] = {
        'train': data_config['datasets']['train'],
        'dev': data_config['datasets']['dev']
    }

    # Pass sim_matrix_config, query_idf_config and num_negative to data_config
    data_config['sim_matrix_config'] = model_config['sim_matrix_config']
    data_config['query_idf_config'] = model_config['query_idf_config']
    data_config['num_negative'] = model_config['num_negative']

    if 'NDCG20' in args.metrics or 'ERR20' in args.metrics:
        # For TREC qrel file
        assert len(data_config['datasets']['dev']) == 1, \
            "Only provide one QREL file for dev"
        model_config['qrel_file'] = data_config['datasets']['dev'][0]

    if args.model_folder:
        model_config['model_folder'] = args.model_folder

    for metric in args.metrics + [model_config['metric']]:
        assert metric in AVAILABLE_METRICS, \
            "Unavailable metric %s" % metric

    config = {
        'data': data_config,
        'model': model_config,
        'monitoring_metric': model_config['metric'],
        'metrics': args.metrics
    }

    if args.overload:
        config = edit_config(config, args.overload)
        if 'gpu_device' in config['model']:
            # Bruteforced for Keras/TF
            if not isinstance(config['model']['gpu_device'], tuple):
                config['model']['gpu_device'] = [config['model']['gpu_device']]
            os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % ','.join(str(x) for x in config['model']['gpu_device'])

    return config


if __name__ == '__main__':

    # Argument handling
    config = argument_parser(sys.argv[1:])

    # Load data
    data = Data(config=config['data'])

    # Load model
    model = ModelInterface(config=config['model'])

    # Initialize features
    if not model.initialized:
        model.initialize_features(data)

    # Get data iterators over features
    if config['model']['all_features']:
        # Train
        train_data = data.batches(
            'train',
            batch_size=data.size('train')
        )
        # Dev
        dev_data = data.batches(
            'dev',
            batch_size=data.size('dev')
        )
        # All data
        all_data = {
            'input': {
                'train': train_data[0]['input'],
                'test': dev_data[0]['input']
            },
            'output': {
                'train': train_data[0]['output'],
                'test': dev_data[0]['output']
            }
        }

        train_features, dev_features, nr_samples = \
            model.get_features(**all_data)
        logger_config = {
            'nr_samples': nr_samples,
            'batch_size': config['model']['batch_size'],
            'monitoring_metric': config['monitoring_metric'],
            'metrics': config['metrics']
        }

    else:
        # Train
        train_features = data.batches(
            'train',
            batch_size=config['model']['batch_size'],
            features_model=model.get_features
        )
        # Dev
        dev_features = data.batches(
            'dev',
            batch_size=config['model']['batch_size'],
            features_model=model.get_features
        )
        logger_config = {
            'nr_samples': train_features.nr_samples,
            'batch_size': config['model']['batch_size'],
            'monitoring_metric': config['monitoring_metric'],
            'metrics': config['metrics']
        }

    # Start trainer
    train_logger = TrainLogger(logger_config)
    for epoch_n in range(config['model']['epochs']):

        # Train
        if config['model']['all_features']:
            indices = list(range(logger_config['nr_samples']))
            for i in range(ceil(logger_config['nr_samples'] / config['model']['batch_size'])):
                batch_indices = indices[i * config['model']['batch_size']: (i + 1) * config['model']['batch_size']]
                # Construct batch
                batch = {
                    'input': {},
                    'output': {}
                }
                for key in train_features['input']:
                    batch['input'][key] = [train_features['input'][key][i]
                                           for i in batch_indices]
                for key in train_features['output']:
                    batch['output'][key] = [train_features['output'][key][i]
                                            for i in batch_indices]

                objective = model.update(**batch)
                train_logger.update_on_batch(objective)
        else:
            for batch in train_features:
                objective = model.update(**batch)
                train_logger.update_on_batch(objective)

        # Validation
        predictions = []
        gold = []
        meta_data = []
        if config['model']['all_features']:
            predictions.append(model.predict(dev_features['input']))
            gold.append(dev_features['output'])
        else:
            for batch in dev_features:
                predictions.append(model.predict(batch['input']))
                gold.append(batch['output'])
                if 'meta-data' in batch['input']:
                    meta_data.append(batch['input']['meta-data'])

        train_logger.update_on_epoch(predictions, gold, meta_data, config['model'])
        if train_logger.state == 'save':
            model.save()

    print("Model saved under %s" % config['model']['model_folder'])
