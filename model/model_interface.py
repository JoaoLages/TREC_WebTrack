from model.template import Model as TemplateModel
import os
import yaml
import numpy as np


AVAILABLE_MODELS = {
    'REPACRR': {
        'model': None,
        'file_termination': 'h5'
    }
}


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

    def update(self, input=None, output=None, class_weight=None):
        if input is not None:
            loss = self.model.update(input, output, class_weight)
            return {'loss': loss}

    def predict(self, input, output=None):
        probs = self.model.predict(input=input, output=output)

        if isinstance(probs, np.ndarray):
            probs = probs.tolist()

        if isinstance(probs[0], list):
            return [x for prob in probs for x in prob]
        else:
            return probs

    def get_kmax_input(self, input, output=None):
        return self.model.get_kmax_input(input, output)

    def get_features(self, input=None, output=None):
        return self.model.get_features(input, output)

    def save(self, sub_name=None):
        self.model.save(sub_name)

    def load(self, model_folder):
        if not os.path.isdir(model_folder):
            config_file = "%s/config.yml" % os.path.dirname(model_folder)
        else:
            config_file = '%s/config.yml' % model_folder
        assert os.path.isfile(config_file), "config file does not exist in %s" % model_folder

        self.config = yaml.load(open(config_file, 'r'))

        # Import model
        if AVAILABLE_MODELS[self.config['name']]['model'] is None:
            assert self.config['name'] in AVAILABLE_MODELS, \
                "Model named %s not available" % self.config['name']

            if self.config['name'] == 'REPACRR':
                from model.models.repacrr import REPACRR
                AVAILABLE_MODELS[self.config['name']]['model'] = REPACRR

        self.model = AVAILABLE_MODELS[self.config['name']]['model'](self.config)

        if os.path.isdir(model_folder):
            weights_file = '%s/%s.%s' % \
                           (model_folder,
                            self.config['name'].lower(),
                            AVAILABLE_MODELS[self.config['name']]['file_termination'])
        else:
            weights_file = model_folder

        self.model.load(weights_file)

    def set(self, **kwargs):
        pass

    def get(self, name):
        return getattr(self.model, name)

    def on_exit(self):
        self.model.on_exit()
