import sys
import argparse
from data.data_iterator import Data
from model.model_interface import ModelInterface
from model.logger import TrainLogger, AVAILABLE_METRICS
from utils.utils import parse_config, edit_config
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
        default=['ERR20', 'NDCG20'],
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
            train_combinations.append(
                (['train_%d' % (x+1) for x in range(len(data_config['datasets']['train'])) if x != i], ['dev_%d' % (i+1)])
            )
            aux_dict['train_%d' % (i+1)] = [dev_file]
            aux_dict['dev_%d' % (i+1)] = [dev_file]

            # For TREC qrel file
            model_config['val_qrel_file_%d' % i] = dev_file

            # For retraining
            if model_config['retrain']:
                model_config['train_qrel_files_%d' % i] = \
                    [d for x, d in enumerate(data_config['datasets']['train']) if x != i]

        # Replace with aux_dict
        data_config['datasets'] = aux_dict
    else:
        # Retain only train and dev
        data_config['datasets'] = {
            'train': data_config['datasets']['train'],
            'dev': data_config['datasets']['dev']
        }
        train_combinations = [(['train'], ['dev'])]

        # For TREC qrel file
        assert len(data_config['datasets']['dev']) == 1, \
            "Only provide one QREL file for dev"
        model_config['val_qrel_file_0'] = data_config['datasets']['dev'][0]

        if model_config['retrain']:
            model_config['train_qrel_files_0'] = data_config['datasets']['train']

    # Pass some keys of model_config to data_config
    data_config['sim_matrix_config'] = model_config['sim_matrix_config']
    data_config['query_idf_config'] = model_config['query_idf_config']
    data_config['num_negative'] = model_config['num_negative']
    data_config['use_description'] = model_config['use_description']
    data_config['use_topic'] = model_config['use_topic']
    data_config['custom_loss'] = model_config['custom_loss']
    if model_config['retrain']:
        data_config['retrain_mode'] = model_config['retrain_mode']

    # if model_config['sim_matrix_config']['use_static_matrices'] and model_config['top_k'] != 0:
    #     raise Exception("'use_embedding_layer' is set to True but 'top_k' != 0 and 'use_static_matrices' set to True, which makes embeddings useless")
        
    if 'embeddings_path' in data_config:
        model_config['embeddings_path'] = data_config['embeddings_path']
    model_config['model_folder'] = args.model_folder

    for metric in args.metrics + [model_config['metric']]:
        assert metric in AVAILABLE_METRICS, \
            "Unavailable metric %s" % metric

    config = {
        'data': data_config,
        'model': model_config,
        'monitoring_metric': model_config['metric'],
        'metrics': args.metrics,
        'num_gpus': 1
    }

    if args.overload:
        config = edit_config(config, args.overload)
        if 'gpu_device' in config['model']:
            # Bruteforced for Keras/TF
            if not isinstance(config['model']['gpu_device'], tuple):
                config['model']['gpu_device'] = [config['model']['gpu_device']]
            os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % ','.join(str(x) for x in config['model']['gpu_device'])
        config['num_gpus'] = len(config['model']['gpu_device'])

    return config, train_combinations


if __name__ == '__main__':

    # Argument handling
    config, train_combinations = argument_parser(sys.argv[1:])

    # Load data
    data = Data(config=config['data'])

    # Start train logger
    config['model']['batch_size'] *= config['num_gpus'] 
    logger_config = {
        'batch_size': config['model']['batch_size'],
        'monitoring_metric': config['monitoring_metric'],
        'metrics': config['metrics'],
        'n_train': len(train_combinations)
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
        train_logger.reset_logger(nr_samples, hard_reset=True)

        # QREL file
        config['model']['qrel_file'] = config['model']['val_qrel_file_%d' % i]

        # Start epoch training
        for epoch_n in range(config['model']['epochs']):
            # Shuffle train
            train_features.shuffle()

            # 1st Train
            for count, batch in enumerate(train_features):
                if count == nr_samples/config['model']['batch_size']:
                    break
                objective = model.update(**batch)
                train_logger.update_on_batch(objective)

            # 1st Validation
            predictions = []
            gold = []
            meta_data = []
            for batch in dev_features:
                predictions.extend(model.predict(batch['input']))
                gold.append(batch['output'])
                if 'meta-data' in batch['input']:
                    meta_data.append(batch['input']['meta-data'])

            train_logger.update_on_epoch(predictions, gold, meta_data, config['model'])
            if train_logger.state == 'save':
                model.save(str(i))

            if config['model']['retrain']:
                # Get retrain data
                retrain_features = data.retrain_batches(
                    train_features, model.predict,
                    config['model']['train_qrel_files_%d' % i],
                    batch_size=config['model']['batch_size'],
                    features_model=model.get_features
                )

                # Shuffle retrain
                retrain_features.shuffle()

                # Reset train logger
                if 'nr_samples' in config['model']:
                    retrain_nr_samples = config['model']['nr_samples']
                else:
                    # Use all the data every epoch
                    retrain_nr_samples = retrain_features.nr_samples

                # 2nd Train
                train_logger.reset_logger(retrain_nr_samples)  # Reset logger to normal train mode
                for count, batch in enumerate(retrain_features):
                    if count == retrain_nr_samples / config['model']['batch_size']:
                        break
                    objective = model.update(**batch, class_weight={1:config['model']['retrain_weight']})
                    train_logger.update_on_batch(objective)

                # 2nd Validation
                predictions = []
                gold = []
                meta_data = []
                for batch in dev_features:
                    predictions.extend(model.predict(batch['input']))
                    gold.append(batch['output'])
                    if 'meta-data' in batch['input']:
                        meta_data.append(batch['input']['meta-data'])

                train_logger.update_on_epoch(predictions, gold, meta_data, config['model'])
                if train_logger.state == 'save':
                    model.save(str(i))

                # Reset logger to normal train mode
                train_logger.reset_logger(nr_samples)

    # Save model evolution
    train_logger.plot_curve(config['model']['model_folder'])

    print("Model saved under %s" % config['model']['model_folder'])
