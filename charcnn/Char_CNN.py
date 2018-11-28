import tensorflow as tf
from tensorflow.python.ops import array_ops
import math


class CharCNN(object):

    def __init__(self, conv_layers, fully_layers, sen_max_length, alphabet_size, class_nums):

        self.conv_layers = conv_layers
        self.fully_layers = fully_layers
        self.sen_max_length = sen_max_length
        self.alphabet_size = alphabet_size
        self.class_nums = class_nums

        with tf.name_scope("Input-Layer"):
            self.input_x = tf.placeholder(tf.int32, [None, sen_max_length], name='input_x')
            self.input_y = tf.placeholder(tf.int32, [None, class_nums], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("Embedding-Layer"), tf.device('/cpu:0'):
            # 加入全零的向量 表示空字符
            embedding = tf.concat([tf.zeros([1, alphabet_size]),
                                   tf.one_hot(tf.range(alphabet_size), alphabet_size, 1.0, 0.0)], 0, name='embedding')

            x = tf.nn.embedding_lookup(embedding, self.input_x)
            x = tf.expand_dims(x, -1)

        # convolution and pool cl 的格式为 [256,7,3] 其中256为filter的size
        # 7为filter_height 3为max_pool该维的size 为None则不进行
        for cl in conv_layers:
            filter_shape = [cl[1], x.get_shape()[2].value, 1, cl[0]]
            # Convolution layer
            stdv = 1 / math.sqrt(cl[0] * cl[1])
            W = tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv), dtype='float32', name='W')
            b = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name='b')
            conv = tf.nn.conv2d(x, W, [1, 1, 1, 1], "VALID", name='conv')
            x = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            if not cl[-1] is None:
                with tf.name_scope("max_pooling"):
                    pool = tf.nn.max_pool(x, ksize=[1, cl[-1], 1, 1],
                                          strides=[1, cl[-1], 1, 1], padding='VALID')
                    x = tf.transpose(pool, [0, 1, 3, 2])
            else:
                x = tf.transpose(x, [0, 1, 3, 2])

        with tf.name_scope("ReshapeLayer"):
            # Reshape layer
            vec_dim = x.get_shape()[1].value * x.get_shape()[2].value
            x = tf.reshape(x, [-1, vec_dim])

        with tf.name_scope("fully_connect"):
            weights = [vec_dim] + list(fully_layers)
            for k in range(len(fully_layers)):
                stdv = 1 / math.sqrt(weights[k])
                W = tf.Variable(tf.random_uniform(shape=[weights[k], fully_layers[k]], minval=-stdv, maxval=stdv),
                                dtype='float32', name='W')
                b = tf.Variable(tf.random_uniform(shape=[fully_layers[k]], minval=-stdv, maxval=stdv),
                                dtype='float32', name='b')
                x = tf.nn.xw_plus_b(x, W, b)

                with tf.name_scope("DropoutLayer"):
                    # Add dropout
                    x = tf.nn.dropout(x, self.dropout_keep_prob)

        with tf.name_scope("output_layer"):
            stdv = 1 / math.sqrt(fully_layers[-1])
            # Output layer
            W = tf.Variable(tf.random_uniform([fully_layers[-1], class_nums], minval=-stdv, maxval=stdv),
                            dtype='float32', name='W')
            b = tf.Variable(tf.random_uniform(shape=[class_nums], minval=-stdv, maxval=stdv), name='b')

            self.logit = tf.nn.xw_plus_b(x, W, b, name="scores")
            self.score = tf.nn.softmax(self.logit)
            self.predictions = tf.argmax(self.logit, 1)

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logit)
            loss = tf.reduce_mean(losses)
            self.focal_loss = self.focal_loss(self.score, tf.cast(self.input_y, tf.float32))
            self.loss = 0.7 * self.focal_loss + 0.3 * loss

        with tf.name_scope("Accuracy"):
            # Accuracy
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            # add summary
            loss_summary = tf.summary.scalar("cost", self.loss)
            # add summary
            accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
            self.summary = tf.summary.merge([loss_summary, accuracy_summary])

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            # 对var_list中的变量计算loss的梯度 返回一个以元组(gradient, variable)组成的列表
            grads_and_vars = optimizer.compute_gradients(self.loss)
            # 将计算出的梯度应用到变量上
            self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def focal_loss(self, prediction_tensor, target_tensor, weights=None, alpha=1.0, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         weights: A float tensor of shape [batch_size, num_anchors]
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        sigmoid_p = prediction_tensor

        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

        my_entry_cross = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0))

        return tf.reduce_mean(my_entry_cross)