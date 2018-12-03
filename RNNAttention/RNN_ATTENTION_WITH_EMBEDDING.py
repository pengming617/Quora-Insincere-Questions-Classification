import tensorflow as tf


class RNN_Attention(object):

    def __init__(self, max_length, num_classes, hidden_num, attn_size):
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

        # attention
        attention_size = attn_size
        outputs = tf.transpose(outputs, [1, 0, 2])
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2 * hidden_num, attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            for t in range(max_length):
                u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in range(max_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            # masked
            attn_zconcat = self.mask(attn_zconcat, self.seq_length, max_length)
            self.alpha = tf.nn.softmax(attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.expand_dims(self.alpha, -1)

        self.final_output = tf.reduce_sum(tf.transpose(outputs, [1, 0, 2]) * alpha_trans, 1)

        # outputs shape: (batch_size, sequence_length, 2*hidden_num)
        fc_w = tf.Variable(tf.truncated_normal([2 * hidden_num, num_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([num_classes]), name='fc_b')

        self.logits = tf.matmul(self.final_output, fc_w) + fc_b
        self.score = tf.nn.softmax(self.logits, name='score')
        self.predictions = tf.argmax(self.score, 1, name="predictions")

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.input_y, tf.cast(self.predictions, tf.int32)), tf.float32))

    def mask(self, inputs, seq_len, max_len):
        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32)
        return inputs - (1 - mask) * 1e12


rnn_att = RNN_Attention(max_length=70,
                                num_classes=2,
                                hidden_num=100,
                                attn_size=100)
print("success")