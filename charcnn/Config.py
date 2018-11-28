# encoding:utf-8


class Config(object):

    def __init__(self):
        self.Batch_Size = 256
        self.epoch = 10
        self.conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]
        self.fully_layers = [1024, 1024]


if __name__ == '__main__':
    config = Config()