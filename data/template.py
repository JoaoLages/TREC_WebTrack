"""
Simple sinusoid data-set
"""


class Data(object):
    """
    Template
    """
    def __init__(self, config):
        # Data-sets
        self.datasets = {}
        # Config
        self.config = config
        # Number of samples
        self.nr_samples = {}

    def size(self, set_name):
        raise NotImplementedError("You need to implement batches method")

    def batches(self, set_name):
        raise NotImplementedError("You need to implement batches method")

    def nr_batches(self, set_name):
        raise NotImplementedError("You need to implement nr_batches method")


class DataIterator():
    """
    Basic data iterator
    """
    def __init__(self, data, nr_samples):
        self.data = data
        self.nr_samples = nr_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
