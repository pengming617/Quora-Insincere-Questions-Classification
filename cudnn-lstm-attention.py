from sklearn.model_selection import train_test_split
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import numpy as np
import math
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.python.ops import array_ops


train_df = pd.read_csv("data/train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.1)

embeddings_index = {}
f = open('data/glove.840B.300d.txt')
# f = open('../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt')
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
    newdoc = []
    for word in doc:
        if len(word) >= 2:
            word = wnl.lemmatize(word.lower())
            newdoc.append(word)
    if len(newdoc) > SEQ_LEN:
        return newdoc[0:SEQ_LEN], SEQ_LEN
    else:
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

class BaseRNN(object):
    pass


helper_doc = """\n[artf]> Implement Multi-layers BiDirectional CudnnRNN

CudnnRNN is more effective than vanilla RNN on GPU

Args:
    RNN: 
        - num_units: int | required
        - batch_size: int | required
        - input_size: int/placeholder | required
        - num_layers: int
            layers of rnn 
        - dropout: float/placeholder
            dropout rate
        - kernel: enum lstm(default) / gru

    __call__:
        - inputs: required
        - seq_len: default None
        - batch_first: bool
            True if the shape is (batch_size, seq_len, embed_size)
            False if the shape is (seq_len, batch_size, embed_size)
        - scope: str
        - reuse:

Usage:
    rnn = BiCudnnRNN(num_units, batch_size, input_size,
              num_layers=1, dropout=0.0, kernel='lstm', scope='rnn')
    (value1, ...) = rnn(inputs, 
                        seq_len=None, 
                        batch_first=True,
                        scope="bidirection_cudnn_rnn",
                        reuse=None)


Input:
    inputs: (batch_size, seq_len, embed_size)

Output:
    lstm: return 3 values that are output, c, h
        - output:  (batch_size, seq_len, 2 * num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, 2 * num_units)
            last context state
        - h: (batch_size, 2 * num_units)
            last hidden state

    gru: return 2 values that are output, c
        - output:  (batch_size, seq_len, 2 * num_layers * num_units)
            the product of the concatenation of each layer
        - c: (batch_size, 2 * num_units)
            last context state
"""


class BiCudnnRNN(BaseRNN):
    @staticmethod
    def helper():
        print(helper_doc)

    def __init__(self, num_units, batch_size, input_size, dropout,
                 num_layers=1, kernel='lstm'):

        if kernel == 'gru':
            self.rnn_cell = tf.contrib.cudnn_rnn.CudnnGRU
        else:
            self.rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM

        self.kernel = kernel
        self.num_layers = num_layers
        self.rnns = []
        self.inits = []
        self.dropout_mask = []

        # init layers
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else 2 * num_units

            init_rw_c = tf.tile(tf.get_variable('init_rw_c_{}'.format(layer),
                                                shape=[1, 1, num_units],
                                                initializer=tf.zeros_initializer()),
                                [1, batch_size, 1])

            init_rw_h = tf.tile(tf.get_variable('init_rw_h_{}'.format(layer),
                                                shape=[1, 1, num_units],
                                                initializer=tf.zeros_initializer()),
                                [1, batch_size, 1])

            init_bw_c = tf.tile(tf.get_variable('init_bw_c_{}'.format(layer),
                                                shape=[1, 1, num_units],
                                                initializer=tf.zeros_initializer()),
                                [1, batch_size, 1])

            init_bw_h = tf.tile(tf.get_variable('init_bw_h_{}'.format(layer),
                                                shape=[1, 1, num_units],
                                                initializer=tf.zeros_initializer()),
                                [1, batch_size, 1])

            rnn_fw = self.rnn_cell(1, num_units)
            rnn_bw = self.rnn_cell(1, num_units)

            mask_fw = tf.nn.dropout(tf.ones([1, batch_size, in_size], dtype=tf.float32), dropout)
            mask_bw = tf.nn.dropout(tf.ones([1, batch_size, in_size], dtype=tf.float32), dropout)

            self.inits.append(((init_rw_c, init_rw_h), (init_bw_c, init_bw_h)))
            self.rnns.append((rnn_fw, rnn_bw))
            self.dropout_mask.append((mask_fw, mask_bw))

    def __call__(self, inputs,
                 seq_len=None,
                 batch_first=True,
                 scope='bidirection_cudnn_rnn',
                 reuse=None):
        if batch_first:
            # transpose to batch second
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs = [inputs]

        with tf.variable_scope(scope, reuse=reuse):
            for layer in range(self.num_layers):
                (init_rw_c, init_rw_h), (init_bw_c, init_bw_h) = self.inits[layer]
                rnn_fw, rnn_bw = self.rnns[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]

                # forward
                with tf.variable_scope("fw_{}".format(layer)):
                    if self.kernel == 'lstm':
                        initial_state = (init_rw_c, init_rw_h)
                    else:
                        initial_state = (init_rw_c,)

                    out_fw, state_fw = rnn_fw(
                        outputs[-1] * mask_fw, initial_state=initial_state)

                # backword
                with tf.variable_scope("bw_{}".format(layer)):
                    if self.kernel == 'lstm':
                        initial_state = (init_rw_c, init_rw_h)
                    else:
                        initial_state = (init_rw_c,)
                    if seq_len is not None:
                        inputs_bw = tf.reverse_sequence(
                            outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    else:
                        inputs_bw = array_ops.reverse(outputs[-1] * mask_bw, axis=[0])

                    out_bw, state_bw = rnn_bw(inputs_bw, initial_state=initial_state)
                    if seq_len is not None:
                        out_bw = tf.reverse_sequence(
                            out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                    else:
                        out_bw = array_ops.reverse(out_bw, axis=[0])

                outputs.append(tf.concat([out_fw, out_bw], axis=2))

            res = tf.concat(outputs[1:], axis=2)

            if batch_first:
                # transpose back
                res = tf.transpose(res, [1, 0, 2])

            if self.kernel == 'lstm':
                C = tf.concat([state_fw[0][0], state_bw[0][0]], axis=1)
                H = tf.concat([state_fw[1][0], state_bw[1][0]], axis=1)
                return res, C, H
            C = tf.concat([state_fw[0][0], state_bw[0][0]], axis=1)
            return res, C


class RNN_Attention(object):

    def __init__(self, max_length, num_classes, hidden_num, attn_size, num_layers):
        # placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.float32, shape=[None, max_length, 300], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None], name='input_y')
        self.drop_out_prob = tf.placeholder(tf.float32, name='drop_out_keep')
        self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')  # 语句的真实长度

        self.embedding_chars = self.input_x

        # build model
        batch_size = array_ops.shape(self.embedding_chars)[0]
        bicudnn_lstm = BiCudnnRNN(num_units=hidden_num,
                                  num_layers=num_layers,
                                  batch_size=batch_size,
                                  input_size=300,
                                  dropout=self.drop_out_prob)
        outputs, C, H = bicudnn_lstm(self.embedding_chars, seq_len=self.seq_length)

        # attention Trainable parameters
        w_omega = tf.Variable(tf.random_normal([2 * hidden_num * num_layers, attn_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attn_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attn_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.einsum("ijk,kl->ijl", outputs, w_omega) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.einsum("ijk,kl->ijl", v, tf.expand_dims(u_omega, -1))  # (B,T) shape
        vu = tf.squeeze(vu, -1)
        vu = self.mask(vu, self.seq_length, max_length)

        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        self.final_output = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)

        # outputs shape: (batch_size, sequence_length, 2*hidden_num)
        fc_w = tf.Variable(tf.truncated_normal([2 * hidden_num * num_layers, num_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([num_classes]), name='fc_b')

        self.logits = tf.matmul(self.final_output, fc_w) + fc_b
        self.score = tf.nn.softmax(self.logits, name='score')
        self.predictions = tf.argmax(self.score, 1, name="predictions")

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        self.cost = tf.reduce_sum(self.cost)
        #         self.focal_loss = self.focal_loss(self.score, tf.cast(tf.one_hot(self.input_y, 2), tf.float32))
        #         self.cost = 0.8 * self.focal_loss + 0.2 * self.cost

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.input_y, tf.cast(self.predictions, tf.int32)), tf.float32))

    def mask(self, inputs, seq_len, max_len):
        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32)
        return inputs - (1 - mask) * 1e12

    def focal_loss(self, prediction_tensor, target_tensor, weights=None, alpha=0.75, gamma=2):
        # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        sigmoid_p = prediction_tensor

        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

        my_entry_cross = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0))

        return tf.reduce_mean(my_entry_cross)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        rnn_att = RNN_Attention(max_length=SEQ_LEN,
                                num_classes=2,
                                hidden_num=hidden_num,
                                attn_size=attn_size,
                                num_layers=2)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_f1 = 0.0

        for time in range(epoch):
            all_loss = 0.0
            accuracys = []
            for trainX_batch, trainY_batch, text_length in tqdm(batch_gen_df(train_df)):
                feed_dict = {
                    rnn_att.input_x: trainX_batch,
                    rnn_att.input_y: trainY_batch,
                    rnn_att.drop_out_prob: dropout_keep_prob,
                    rnn_att.seq_length: text_length
                }
                _, cost, accuracy = sess.run([rnn_att.train_op, rnn_att.cost, rnn_att.accuracy], feed_dict)
                all_loss += cost
                accuracys.append(accuracy)

            # print("第"+str((time+1))+"次迭代的损失为："+str(all_loss)+";准确率为："+str(np.mean(accuracys)))

            all_dev_score = []
            y_dev = []
            for valX_batch, valY_batch, text_length in tqdm(batch_gen_df(val_df)):
                feed_dict = {
                    rnn_att.input_x: valX_batch,
                    rnn_att.input_y: valY_batch,
                    rnn_att.drop_out_prob: 1.0,
                    rnn_att.seq_length: text_length
                }
                dev_loss, dev_score = sess.run([rnn_att.cost, rnn_att.score], feed_dict)
                all_dev_score.extend(dev_score.tolist())
                y_dev.extend(valY_batch)
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

        print("model train over----------f1:" + str(best_f1))

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