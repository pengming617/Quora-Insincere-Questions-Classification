import tensorflow.contrib.learn as learn
import tensorflow as tf
import numpy as np
import RNNAttention.Processing as Processing

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
process = Processing.Processing()

dicts = {0: '0', 1: '1'}


class Infer(object):
    """
        ues CNN model to predict classification.
    """
    def __init__(self):
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore('model/vocab.pickle')
        self.checkpoint_file = tf.train.latest_checkpoint('model')
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                self.max_length = self.input_x.shape[1]
                self.seq_length = graph.get_operation_by_name("seq_length").outputs[0]
                self.drop_keep_prob = graph.get_operation_by_name("drop_out_keep").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("predictions").outputs[0]
                self.scores = graph.get_operation_by_name("score").outputs[0]

    def infer(self, sentences):
        # transfer to vector
        sentence_word = []
        length = []
        for sentence in sentences:
            sentence = process.preprocess(sentence)
            words = sentence.split(" ")
            sen_len = len(words)
            if sen_len > self.max_length:
                newwords = words[0:self.max_length]
                sentence_word.append(' '.join(newwords))
                length.append(self.max_length)
            else:
                sentence_word.append(sentence)
                length.append(sen_len)
        sentences_vectors = np.array(list(self.vocab_processor.fit_transform(sentence_word)))

        feed_dict = {
            self.input_x: sentences_vectors,
            self.drop_keep_prob: 1.0,
            self.seq_length: np.array(length)
        }
        y, s = self.sess.run([self.predictions, self.scores], feed_dict)

        labels = []
        for x in range(len(y)):
            if s[x][1] > 0.35:
                labels.append('1')
            else:
                labels.append('0')
        return labels, s








