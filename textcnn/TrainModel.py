# encoding:utf-8
import tensorflow as tf
from textcnn import TextCNN
from textcnn import Processing
import textcnn.Config as Config
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import numpy as np
import os


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
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
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)

            with tf.name_scope("readfile"):
                processing = Processing.Processing()
                articles, tags = processing.loadPracticeFile("data/train_all.txt")
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.2, random_state=0)
            # 加载词典
            vocab = learn.preprocessing.VocabularyProcessor.restore('model/vocab.pickle')

            with sess.as_default():
                textcnn = TextCNN.TextCNN(
                    max_length=len(self.data_embedding_new[0]),
                    num_classes=len(y_train[0]),
                    vocab_size=len(vocab.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                # 对var_list中的变量计算loss的梯度 返回一个以元组(gradient, variable)组成的列表
                grads_and_vars = optimizer.compute_gradients(textcnn.loss)
                # 将计算出的梯度应用到变量上
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                best_f1 = 0.0

                for time in range(config.epoch):
                    batch_size = config.Batch_Size
                    for trainX_batch, trainY_batch in self.get_batches(X_train, y_train, batch_size):
                        feed_dict = {
                            textcnn.input_x: np.array(trainX_batch),
                            textcnn.input_y: np.array(trainY_batch),
                            textcnn.drop_keep_prob: FLAGS.dropout_keep_prob
                        }
                        _, loss, train_accuracy = sess.run([train_op, textcnn.loss, textcnn.accuracy], feed_dict)

                    print("第"+str((time+1))+"次迭代的损失为："+str(loss)+";准确率为："+str(train_accuracy))

                    all_dev = []
                    for devX_batch, devY_batch in self.get_batches(X_val, y_val, batch_size):
                        feed_dict = {
                            textcnn.input_x: np.array(devX_batch),
                            textcnn.input_y: np.array(devY_batch),
                            textcnn.drop_keep_prob: 1.0
                        }
                        dev_loss, dev_predictions = sess.run([textcnn.loss, textcnn.predictions], feed_dict)
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
                    print("f1:{},recall:{},acc:{}".format(dev_f1, dev_recall, dev_acc))
                    if dev_f1 > best_f1:
                        best_f1 = dev_f1
                        saver.save(sess, "model/TextCNNModel.ckpt")
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