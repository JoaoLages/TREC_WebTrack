import pickle as pkl
from model.template import Model as TemplateModel
from data.text import initialization
import os
import yaml
import numpy as np


AVAILABLE_MODELS = {
    'REPACRR': {
        'model': None,
        'file_termination': 'h5'
    }
}


def init_config(config, data=None, model_folder=None):

    if not model_folder:
        print("Constructing Vocabulary")

        encoding, unique_tags, unique_tags_binary = initialization(
            data,
            config['embeddings']
        )

        pkl.dump(encoding, open("%s/vocabulary" % config['model_folder'], "wb"))
        pkl.dump(unique_tags, open("%s/unique_tags" % config['model_folder'], "wb"))

        print("Finished constructing Vocabulary")
    else:
        print("Loading Vocabulary")

        vocabulary_file = '%s/vocabulary' % model_folder
        assert os.path.isfile(vocabulary_file), \
            "vocabulary file does not exist in %s" % model_folder

        encoding = pkl.load(open(vocabulary_file, "rb"))

        unique_tags_file = '%s/unique_tags' % model_folder
        assert os.path.isfile(unique_tags_file), \
            "unique_tags file does not exist in %s" % model_folder

        unique_tags = pkl.load(open(unique_tags_file, "rb"))

        unique_tags_binary_file = '%s/unique_tags_binary' % model_folder
        assert os.path.isfile(unique_tags_binary_file), \
            "unique_tags_binary file does not exist in %s" % model_folder

        unique_tags_binary = pkl.load(open(unique_tags_binary_file, "rb"))

    model_config = {
        # Provide needed information to extract raw input
        'input': {
            'query': {
                'vocabulary': encoding['vocabulary'],
                'special_tokens': encoding['special_tokens'],
            },
            'subtopic': {
                'vocabulary': encoding['vocabulary'],
                'special_tokens': encoding['special_tokens'],
            },
            'article': {
                'vocabulary': encoding['vocabulary'],
                'special_tokens': encoding['special_tokens'],
            }
        },
        'output': {
            'tags': {
                'vocabulary': list(unique_tags),
                'special_tokens': {}
            },
            'binary_tags': {
                'vocabulary': list(unique_tags_binary),
                'special_tokens': {}
            }
        },
        'embeddings': encoding['emb']
    }

    return model_config


class ModelInterface(TemplateModel):

    def __init__(self, config=None, model_folder=None):
        super(ModelInterface, self).__init__(
            config=config,
            model_folder=model_folder
        )
        if config:
            self.model = None

        self.metric = None
        self.initialized = False

    def initialize_features(self, data):
        if self.config['name'] in AVAILABLE_MODELS:
            # Save config
            os.system("mkdir -p {}".format(self.config['model_folder']))
            yaml.dump(self.config,
                      open("%s/config.yml" % self.config['model_folder'], "w"),
                      default_flow_style=False)

            # Construct vocabulary
            if self.config['construct_vocab']:
                model_config = init_config(self.config, data)
                self.config.update(model_config)
        else:
            raise ValueError("Invalid Model Name")

        # Import model
        if AVAILABLE_MODELS[self.config['name']]['model'] is None:
            assert self.config['name'] in AVAILABLE_MODELS, \
                "Model named %s not available" % self.config['name']

            if self.config['name'] == 'REPACRR':
                from model.models.repacrr import REPACRR
                AVAILABLE_MODELS[self.config['name']]['model'] = REPACRR

        self.initialized = True
        self.model = AVAILABLE_MODELS[self.config['name']]['model'](self.config)
        self.metric = self.model.metric

    def update(self, input=None, output=None):
        if input is not None:
            loss = self.model.update(input, output)
            return {'loss': loss}

    def predict(self, input, output=None):
        probs = self.model.predict(input=input, output=output)

        if isinstance(probs, np.ndarray):
            probs = probs.tolist()

        if isinstance(probs[0], list):
            return {
                'predicted_tags': [
                    int(x >= 0.5) for prob in probs for x in prob
                ],
                'probs': [
                    x for prob in probs for x in prob
                ]
            }
        else:
            return {
                # case
                'predicted_tags': [int(x >= 0.5) for x in probs],
                'probs': probs
            }

    def get_features(self, input=None, output=None):
        return self.model.get_features(input, output)

    def save(self):
        self.model.save()

    def load(self, model_folder):
        config_file = '%s/config.yml' % model_folder
        assert os.path.isfile(config_file), "config file does not exist in %s" % model_folder

        self.config = yaml.load(open(config_file, 'r'))

        if self.config['name'] in AVAILABLE_MODELS:
            if self.config['construct_vocab']:
                model_config = init_config(self.config, model_folder=model_folder)
                self.config.update(model_config)
        else:
            raise ValueError("Invalid Model Name")

        # Import model
        if AVAILABLE_MODELS[self.config['name']]['model'] is None:
            assert self.config['name'] in AVAILABLE_MODELS, \
                "Model named %s not available" % self.config['name']

            if self.config['name'] == 'REPACRR':
                from model.models.repacrr import REPACRR
                AVAILABLE_MODELS[self.config['name']]['model'] = REPACRR

        self.model = AVAILABLE_MODELS[self.config['name']]['model'](self.config)

        weights_file = '%s/%s.%s' % \
                       (model_folder,
                        self.config['name'].lower(),
                        AVAILABLE_MODELS[self.config['name']]['file_termination'])

        self.model.load(weights_file)

    def set(self, **kwargs):
        pass

    def get(self, name):
        return getattr(self.model, name)

    def on_exit(self):
        self.model.on_exit()
