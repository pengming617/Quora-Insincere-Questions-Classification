from sklearn.model_selection import train_test_split
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from tqdm import tqdm
import numpy as np
import math
import os
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import tensorflow as tf

if not os.path.exists('output'):
    os.makedirs('output')
    print("mkdir success")
else:
    print('output exist')
train_df = pd.read_csv("data/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.1)

embeddings_index = {}
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

batch_size = 256
SEQ_LEN = 70
# Model Hyperparameters
dropout_keep_prob = 0.5
hidden_num = 128
attn_size = 128
epoch = 5


def preprocess(doc):
    wnl = WordNetLemmatizer()  # 词形还原
    for c in string.punctuation:
        # 去标点
        doc = doc.replace(c, ' ')
    for c in string.digits:
        # 去数字
        doc = doc.replace(c, '')
    doc = nltk.word_tokenize(doc)
    # 只保留长度不小于3的单词,去除停用词,验证是否为英文单词(利用wordnet)
    newdoc = []
    for word in doc:
        # if len(word) >= 3 and word not in stop_words and wordnet.synsets(word):
        if len(word) >= 3 and wordnet.synsets(word):
            word = wnl.lemmatize(word)
            newdoc.append(word)
    return newdoc, len(newdoc)


def text_to_array(text):
    empyt_emb = np.zeros(300)
    # print(empyt_emb)
    words, real_length = preprocess(text)
    embeds = [embeddings_index.get(x, empyt_emb) for x in words]
    embeds += [empyt_emb] * (SEQ_LEN - len(embeds))
    # print(len(embeds))
    return np.array(embeds), real_length


def batch_gen_df(x_df):
    n_batches = math.ceil(len(x_df) / batch_size)
    x_df = x_df.sample(frac=1.)  # Shuffle the data.
    for i in range(n_batches):
        texts = x_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = []
        text_length = []
        for text in texts:
            embeds, length = text_to_array(text)
            text_arr.append(embeds)
            text_length.append(length)
        target = x_df["target"][i*batch_size:(i+1)*batch_size]
        yield np.array(text_arr), np.array(target), np.array(text_length)


def batch_gen_test(x_df):
    n_batches = math.ceil(len(x_df) / batch_size)
    for i in range(n_batches):
        texts = x_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = []
        text_length = []
        for text in texts:
            embeds, length = text_to_array(text)
            text_arr.append(embeds)
            text_length.append(length)
        yield np.array(text_arr), np.array(text_length)


class RNN_Attention(object):

    def __init__(self, max_length, num_classes, hidden_num, attn_size):
        with tf.device('/gpu:1'):
            # placeholders for input output and dropout
            self.input_x = tf.placeholder(tf.float32, shape=[None, max_length, 300], name='input_x')
            self.input_y = tf.placeholder(tf.int32, shape=[None], name='input_y')
            self.drop_out_prob = tf.placeholder(tf.float32, name='drop_out_keep')
            self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')  # 语句的真实长度

            self.embedding_chars = self.input_x

            # build model
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_num)
            fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.drop_out_prob)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_num)
            bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.drop_out_prob)

            (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, self.embedding_chars,
                                                                       sequence_length=self.seq_length,
                                                                       dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)

            # attention Trainable parameters
            w_omega = tf.Variable(tf.random_normal([2 * hidden_num, attn_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([attn_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([attn_size], stddev=0.1))

            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.einsum("ijk,kl->ijl", outputs, w_omega) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.einsum("ijk,kl->ijl", v, tf.expand_dims(u_omega,-1))  # (B,T) shape
            vu = tf.squeeze(vu, -1)
            vu = self.mask(vu, self.seq_length, max_length)

            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            self.final_output = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)

            # outputs shape: (batch_size, sequence_length, 2*hidden_num)
            fc_w = tf.Variable(tf.truncated_normal([2 * hidden_num, num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([num_classes]), name='fc_b')

            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.score = tf.nn.softmax(self.logits, name='score')
            self.predictions = tf.argmax(self.score, 1, name="predictions")

            self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.cost = tf.reduce_sum(self.cost)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

            optimizer = tf.train.AdamOptimizer(0.001)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.input_y, tf.cast(self.predictions, tf.int32)), tf.float32))

    def mask(self, inputs, seq_len, max_len):
        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32)
        return inputs - (1 - mask) * 1e12


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        rnn_att = RNN_Attention(max_length=SEQ_LEN,
                                num_classes=2,
                                hidden_num=hidden_num,
                                attn_size=attn_size)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_f1 = 0.0

        for time in range(epoch):
            all_loss = 0.0
            accuracys = []
            for trainX_batch, trainY_batch, text_length in tqdm(batch_gen_df(train_df)):
                try:
                    feed_dict = {
                        rnn_att.input_x: trainX_batch,
                        rnn_att.input_y: trainY_batch,
                        rnn_att.drop_out_prob: dropout_keep_prob,
                        rnn_att.seq_length: text_length
                    }
                    _, cost, accuracy = sess.run([rnn_att.train_op, rnn_att.cost, rnn_att.accuracy], feed_dict)
                    all_loss += cost
                    accuracys.append(accuracy)
                except:
                    pass

            print("第" + str((time + 1)) + "次迭代的损失为：" + str(all_loss) + ";准确率为：" + str(np.mean(accuracys)))

            all_dev_score = []
            y_dev = []
            for valX_batch, valY_batch, text_length in tqdm(batch_gen_df(val_df)):
                try:
                    feed_dict = {
                        rnn_att.input_x: valX_batch,
                        rnn_att.input_y: valY_batch,
                        rnn_att.drop_out_prob: 1.0,
                        rnn_att.seq_length: text_length
                    }
                    dev_loss, dev_score = sess.run([rnn_att.cost, rnn_att.score], feed_dict)
                    all_dev_score.extend(dev_score.tolist())
                    y_dev.extend(valY_batch)
                except:
                    pass

            all_dev = []
            for x in all_dev_score:
                if x[1] > 0.35:
                    all_dev.append(1)
                else:
                    all_dev.append(0)
            # f1值
            dev_f1 = f1_score(np.array(y_dev), np.array(all_dev))
            dev_recall = recall_score(np.array(y_dev), np.array(all_dev))
            dev_acc = accuracy_score(np.array(y_dev), np.array(all_dev))
            print("验证集：f1:{},recall:{},acc:{}\n".format(dev_f1, dev_recall, dev_acc))
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                saver.save(sess, "output/RNNAttentionModel.ckpt")
                print("saved\n")

        print("model train over")

        # infer
        test_df = pd.read_csv("data/test.csv")
        all_preds = []
        for x, length in tqdm(batch_gen_test(test_df)):
            feed_dict = {
                rnn_att.input_x: x,
                rnn_att.drop_out_prob: 1.0,
                rnn_att.seq_length: length
            }
            y, s = sess.run([rnn_att.predictions, rnn_att.score], feed_dict)
            for x in range(len(y)):
                if s[x][1] > 0.35:
                    all_preds.append('1')
                else:
                    all_preds.append('0')
        submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": all_preds})
        submit_df.to_csv("submission.csv", index=False)
        print('submission success')