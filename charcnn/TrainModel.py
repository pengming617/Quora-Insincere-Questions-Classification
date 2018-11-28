# encoding:utf-8
import tensorflow as tf
from charcnn import Char_CNN
from charcnn import Processing
import charcnn.Config as Config
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import numpy as np
import os


# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
root_path = os.getcwd()
project_root_path = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), ".")))
config = Config.Config()


class TrainModel(object):
    '''
        训练模型
        保存模型
    '''
    def trainModel(self):

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)

            with tf.name_scope("readfile"):
                processing = Processing.Processing()
                articles, tags = processing.loadPracticeFile("data/train_all.txt")
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.2, random_state=0)

            with sess.as_default():
                charcnn = Char_CNN.CharCNN(
                    conv_layers=config.conv_layers,
                    fully_layers=config.fully_layers,
                    sen_max_length=len(X_train[0]),
                    alphabet_size=69,
                    class_nums=2)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                best_f1 = 0.0

                for time in range(config.epoch):
                    batch_size = config.Batch_Size
                    for trainX_batch, trainY_batch in self.get_batches(X_train, y_train, batch_size):
                        feed_dict = {
                            charcnn.input_x: np.array(trainX_batch),
                            charcnn.input_y: np.array(trainY_batch),
                            charcnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                        }
                        _, loss, train_accuracy = sess.run([charcnn.train_op, charcnn.loss, charcnn.accuracy], feed_dict)

                    print("训练集：第"+str((time+1))+"次迭代的损失为："+str(loss)+";准确率为："+str(train_accuracy))

                    all_dev = []
                    for devX_batch, devY_batch in self.get_batches(X_val, y_val, batch_size):
                        feed_dict = {
                            charcnn.input_x: np.array(devX_batch),
                            charcnn.input_y: np.array(devY_batch),
                            charcnn.dropout_keep_prob: 1.0
                        }
                        dev_loss, dev_predictions = sess.run([charcnn.loss, charcnn.predictions], feed_dict)
                        all_dev.extend(dev_predictions.tolist())

                    # f1值
                    y_true = []
                    for x in y_val:
                        if x[0] == 1:
                            y_true.append(0)
                        else:
                            y_true.append(1)
                    dev_f1 = f1_score(np.array(y_true), np.array(all_dev))
                    dev_recall = recall_score(np.array(y_true), np.array(all_dev))
                    dev_acc = accuracy_score(np.array(y_true), np.array(all_dev))
                    print("验证集：f1:{},recall:{},acc:{}\n".format(dev_f1, dev_recall, dev_acc))
                    if dev_f1 > best_f1:
                        best_f1 = dev_f1
                        saver.save(sess, "model/CharCNNModel.ckpt")
                        print("saved\n")

    def get_batches(self, X, Y, batch_size):
        if int(len(X) % batch_size) == 0:
            batches = int(len(X) / batch_size)
        else:
            batches = int(len(X) / batch_size) + 1
        for x in range(batches):
            if x != batches - 1:
                trainX_batch = X[x * batch_size:(x + 1) * batch_size]
                trainY_batch = Y[x * batch_size:(x + 1) * batch_size]
            else:
                trainX_batch = X[x * batch_size:len(X)]
                trainY_batch = Y[x * batch_size:len(Y)]
            yield trainX_batch, trainY_batch


if __name__ == '__main__':
    train = TrainModel()
    train.trainModel()