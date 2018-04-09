
class Model(object):

    def __init__(self, model_folder=None, config=None):
        assert bool(model_folder) != bool(config), \
            "Provide either model_folder or config"
        if model_folder:
            self.load(model_folder)
        else:
            self.config = config
        self.initialized = False

    def initialize_features(self, *args):
        self.initialized = True
        raise NotImplementedError("Need to implement initialize_features method")

    def get_features(self, input=None, output=None):
        """
        Default feature extraction is do nothing
        """
        return {'input': input, 'output': output}

    def predict(self, *args):
        raise NotImplementedError("Need to implement predict method")

    def update(self, *args):
        # This needs to return at least {'cost' : 0}
        raise NotImplementedError("Need to implement update method")

    def set(self, **kwargs):
        raise NotImplementedError("Need to implement set method")

    def get(self, name):
        raise NotImplementedError("Need to implement get method")

    def save(self):
        raise NotImplementedError("Need to implement save method")

    def load(self, model_folder):
        raise NotImplementedError("Need to implement load method")
