# encoding:utf-8
import tensorflow as tf
from RNNAttention.Processing import Processing
from RNNAttention.Config import Config
from RNNAttention.RNN_Attention import RNN_Attention
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from tensorflow.contrib import learn

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("hidden_num", 100, "the number of hidden units")
tf.flags.DEFINE_integer("attn_size", 100, "the number of attn_size units")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L1 regularization lambda (default: 0.0)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
config = Config()


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
                processing = Processing()
                articles, tags = processing.loadPracticeFile("data/train_all.txt")
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.2, random_state=0)

            # 加载词典
            vocab = learn.preprocessing.VocabularyProcessor.restore('model/vocab.pickle')

            with sess.as_default():
                rnn_att = RNN_Attention(max_length=len(self.data_embedding_new[0]),
                                        num_classes=len(self.tags_new[0]),
                                        vocab_size=len(vocab.vocabulary_),
                                        embedding_size=FLAGS.embedding_dim,
                                        hidden_num=FLAGS.hidden_num,
                                        attn_size=FLAGS.attn_size)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                best_f1 = 0.0

                for time in range(config.epoch):
                    batch_size = config.batch_size
                    for trainX_batch, trainY_batch in self.get_batches(X_train, y_train, batch_size):
                        feed_dict = {
                            rnn_att.input_x: np.array(trainX_batch),
                            rnn_att.input_y: np.array(trainY_batch),
                            rnn_att.drop_out_prob: FLAGS.dropout_keep_prob,
                            # textRNN.mask_x: np.transpose(np.array(trainX_batch)),
                            rnn_att.seq_length: np.array(self.get_length(trainX_batch))
                        }
                        _, cost, accuracy = sess.run([rnn_att.train_op, rnn_att.cost, rnn_att.accuracy], feed_dict)

                    print("第"+str((time+1))+"次迭代的损失为："+str(cost)+";准确率为："+str(accuracy))

                    all_dev_score = []
                    for devX_batch, devY_batch in self.get_batches(X_val, y_val, batch_size):
                        feed_dict = {
                            rnn_att.input_x: np.array(devX_batch),
                            rnn_att.input_y: np.array(devY_batch),
                            rnn_att.drop_out_prob: 1.0,
                            # textRNN.mask_x: np.transpose(np.array(dev_x)),
                            rnn_att.seq_length: np.array(self.get_length(devX_batch))
                        }
                        dev_loss, dev_score = sess.run([rnn_att.cost, rnn_att.score], feed_dict)
                        all_dev_score.extend(dev_score.tolist())

                    all_dev = []
                    for x in all_dev_score:
                        if x[1] > 0.35:
                            all_dev.append(1)
                        else:
                            all_dev.append(0)
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
                        saver.save(sess, "model/RNNAttentionModel.ckpt")
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

    def get_length(self, trainX_batch):
        # sentence length
        lengths = []
        for sample in trainX_batch:
            count = 0
            for index in sample:
                if index != 0:
                    count += 1
                else:
                    break
            lengths.append(count)
        return lengths


if __name__ == '__main__':
    train = TrainModel()
    train.trainModel()