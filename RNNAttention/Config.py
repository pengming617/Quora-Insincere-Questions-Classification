# encoding:utf-8
import os


class Config(object):

    def __init__(self):
        self.batch_size = 256
        self.epoch = 10


if __name__ == '__main__':
    config = Config()