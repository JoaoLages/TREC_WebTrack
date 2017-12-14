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
        help="Path to save model's outputs",
        required=True,
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
    parser.add_argument(
        '--round-robin',
        help="If true, does every train combination (every train folder gets to be validation once)",
        default=False,
        action='store_true'
    )
    args = parser.parse_args(sys_argv)

    data_config = parse_config(args.data_config)
    model_config = parse_config(args.model_config)

    if args.round_robin:
        assert 'dev' not in data_config['datasets'], \
            "When using --round-robin, dev can't be specified, put all files under 'train'"
        assert len(data_config['datasets']['train']) >= 2, \
            "Please provide more than 1 file for train when using --round-robin"

        # Get train combinations (leave 1 out for dev)
        train_combinations = []
        aux_dict = {}
        for i, dev_file in enumerate(data_config['datasets']['train']):
            train_combinations.append(('train_%d' % (i+1), 'dev_%d' % (i+1)))
            aux_dict['train_%d' % (i+1)] = data_config['datasets']['train'][:i]+data_config['datasets']['train'][i+1:]
            aux_dict['dev_%d' % (i+1)] = [dev_file]

            # For TREC qrel file
            if 'NDCG20' in args.metrics or 'ERR20' in args.metrics:
                model_config['qrel_file_%d' % i] = dev_file

        # Replace with aux_dict
        data_config['datasets'] = aux_dict
    else:
        # Retain only train and dev
        data_config['datasets'] = {
            'train': data_config['datasets']['train'],
            'dev': data_config['datasets']['dev']
        }
        train_combinations = [('train', 'dev')]

        # For TREC qrel file
        if 'NDCG20' in args.metrics or 'ERR20' in args.metrics:
            assert len(data_config['datasets']['dev']) == 1, \
                "Only provide one QREL file for dev"
            model_config['qrel_file_0'] = data_config['datasets']['dev'][0]

    # Pass some keys of model_config to data_config
    data_config['sim_matrix_config'] = model_config['sim_matrix_config']
    data_config['query_idf_config'] = model_config['query_idf_config']
    data_config['num_negative'] = model_config['num_negative']
    data_config['use_description'] = model_config['use_description']
    data_config['use_topic'] = model_config['use_topic']

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

    return config, train_combinations


if __name__ == '__main__':

    # Argument handling
    config, train_combinations = argument_parser(sys.argv[1:])

    # Load data
    data = Data(config=config['data'])

    # Start train logger
    logger_config = {
        'batch_size': config['model']['batch_size'],
        'monitoring_metric': config['monitoring_metric'],
        'metrics': config['metrics']
    }
    train_logger = TrainLogger(logger_config)

    # Iterate through train combinations
    for i, (train, dev) in enumerate(train_combinations):
        # Load model
        model = ModelInterface(config=config['model'])

        # Initialize features
        if not model.initialized:
            model.initialize_features(data)

        # Train
        train_features = data.batches(
            train,
            batch_size=config['model']['batch_size'],
            features_model=model.get_features,
            shuffle_seed=config['data']['shuffle_seed']
        )
        # Dev
        dev_features = data.batches(
            dev,
            batch_size=config['model']['batch_size'],
            features_model=model.get_features,
            shuffle_seed=config['data']['shuffle_seed']
        )

        # Reset train logger
        if 'nr_samples' in config['model']:
            nr_samples = config['model']['nr_samples']
        else:
            # Use all the data every epoch
            nr_samples = train_features.nr_samples
        train_logger.reset_logger(nr_samples)

        # QREL file
        config['model']['qrel_file'] = config['model']['qrel_file_%d' % i]

        # Start epoch training
        for epoch_n in range(config['model']['epochs']):
            # Shuffle train
            train_features.shuffle()

            # Train
            for count, batch in enumerate(train_features):
                if count == nr_samples/config['model']['batch_size']:
                    break
                objective = model.update(**batch)
                train_logger.update_on_batch(objective)

            # Validation
            predictions = []
            gold = []
            meta_data = []
            for batch in dev_features:
                predictions.append(model.predict(batch['input']))
                gold.append(batch['output'])
                if 'meta-data' in batch['input']:
                    meta_data.append(batch['input']['meta-data'])

            train_logger.update_on_epoch(predictions, gold, meta_data, config['model'])
            if train_logger.state == 'save':
                model.save()

    # Save model evolution
    train_logger.plot_curve(config['model']['model_folder'])

    print("Model saved under %s" % config['model']['model_folder'])
