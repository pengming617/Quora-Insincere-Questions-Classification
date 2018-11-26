import tensorflow.contrib.learn as learn
import os
import tensorflow as tf
import numpy as np
import textcnn.Processing as Processing

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
process = Processing.Processing()

dicts = {}
with open("model/labels.txt", 'r', encoding="utf-8") as f:
    for line in f.readlines():
        tag_type = line.replace("\n", "").split(":")
        dicts[int(tag_type[0])] = tag_type[1]


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
                self.drop_keep_prob = graph.get_operation_by_name("drop_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def infer(self, sentences):
        # transfer to vector
        sentence_word = []
        for sentence in sentences:
            sentence = process.preprocess(sentence)
            # sentence_word.append(' '.join(jieba.cut(sentence)))
            sentence_word.append(sentence)
        sentences_vectors = np.array(list(self.vocab_processor.fit_transform(sentence_word)))
        # softmax
        score = tf.nn.softmax(self.scores, 1)

        feed_dict = {
            self.input_x: sentences_vectors,
            self.drop_keep_prob: 1.0
        }
        y, s = self.sess.run([self.predictions, score], feed_dict)

        # 将数字转换为对应的意图
        labels = [dicts[x] for x in y]
        s = [np.max(x) for x in s]
        return labels, s

    # test the model
    def test_model(self):
        test = open('./model/dl_model/textcnn/test.txt', 'w')
        with open("./corpus_data/意图识别数据_all.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.replace("\n", "").replace("\t\t", "\t").split("\t")
                intent, scores = self.infer([data[1]])
                if intent[0] == data[2]:
                    test.writelines(
                        data[1] + "\t" + data[2] + "\t" + data[2] + "\t" + "True" + str(scores[0]) + "\n")
                else:
                    test.writelines(data[1] + "\t" + data[2] + "\t" + intent[0] + "\t" + "False"
                                    + str(scores[0]) + "\n")









