# encoding:utf-8


class Config(object):

    def __init__(self):
        self.Batch_Size = 256
        self.epoch = 10


if __name__ == '__main__':
    config = Config()