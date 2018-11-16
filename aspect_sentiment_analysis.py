import json
import string
import re

import codecs
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils as ut

# Read data
batch_size = 128
seq_max_len = 32
nb_sentiment_label = 2
embedding_size = 300
nb_linear_inside = 256
nb_lstm_inside = 256
layers = 1
TRAINING_ITERATIONS = 15000
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0005
negative_weight = 2.0
positive_weight = 1.0

label_dict = {
    'aspositive': 1,
    'asnegative': 2
}

data_dir = '../data'
flag_word2vec = True
flag_train = True
flag_test = True

train_data, train_mask, train_binary_mask, train_label, train_seq_len, train_sentiment_for_word, word_dict, \
word_dict_rev, embedding, aspect_list = ut.load_data(
    label_dict,
    seq_max_len,
    negative_weight,
    positive_weight,
)


def put_aspect(text, pol, aspects):
    aspect_text = ''
    first_flag = True
    aspect_flag = False
    for sent in text.splitlines():
        translator = str.maketrans('', '', string.punctuation)
        sent = sent.translate(translator)
        sent = sent.replace('.', ' ')
        sent = re.sub('\s+', ' ', sent).strip()
        sent = sent.replace('\u200c', ' ').replace('\r', '').replace('ي', 'ی').replace('ك', 'ک')
        for w in sent.split():
            for key, value in aspects.items():
                if w in value:
                    if pol == 2:
                        aspect_flag = True
                        w = str(key) + '{asnegative}'
                    if pol == 1:
                        aspect_flag = True
                        w = str(key) + '{aspositive}'
                    break
            if first_flag:
                aspect_text += str(w)
                first_flag = False
            else:
                aspect_text += ' ' + str(w)
    return aspect_text, aspect_flag


# full_path = path to crawled data (crawled_data.cd)
def create_filtered_data(full_path):
    # open crawled data
    with open(full_path, 'r', encoding='utf8') as f:
        products = []
        for row in f.readlines():
            raw_data = json.loads(row)
            comments = raw_data.get('cmts', None)
            rate = raw_data.get('r', None)
            cat = raw_data.get('c', None)
            if comments is None or len(comments) == 0 or cat is None or rate is None:
                continue
            valid_comments = []
            for comment in comments:
                pol = comment.get('pol', None)
                if pol is not None and pol != 0:
                    valid_comments.append(comment)
            if len(valid_comments) == 0:
                continue
            raw_data['cmts'] = valid_comments
            products.append(raw_data)
        # gather all comments
        all_comments = []
        for product in products:
            comments = product.get('cmts', [])
            for comment_dict in comments:
                pol = comment_dict.get('pol', None)
                if pol is None:
                    print('err')
                if pol == -1:
                    pol = 2
                text = comment_dict.get('txt', '')
                if text is None:
                    text = ''
                text, flag = put_aspect(text, pol, ut.aspects)
                if flag and len(text.split()) < seq_max_len:
                    all_comments.append(text)
                elif flag:
                    seen = False
                    f_word = True
                    temp_str = ''
                    count_word = 0
                    for w in text.split():
                        if count_word < seq_max_len:
                            if f_word:
                                temp_str += str(w)
                                f_word = False
                            else:
                                temp_str += ' ' + str(w)

                            if '{as' in w:
                                seen = True

                            count_word += 1
                        elif seen:
                            all_comments.append(temp_str)
                            count_word = 0
                            seen = False
                            f_word = True
                            temp_str = ''
                        else:
                            continue
                else:
                    continue

        f = open('data/all_comments.txt', 'w')
        for cmt in all_comments:
            f.write("%s\n" % cmt)
        return all_comments


if __name__ == '__main__':
    create_filtered_data('data/crawled_data.cd')

    nb_sample_train = len(train_data)

    # Modeling

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        tf_X_train = tf.placeholder(tf.float32, shape=[None, seq_max_len, embedding_size])
        tf_X_train_mask = tf.placeholder(tf.float32, shape=[None, seq_max_len])
        tf_X_binary_mask = tf.placeholder(tf.float32, shape=[None, seq_max_len])
        tf_y_train = tf.placeholder(tf.int64, shape=[None, seq_max_len])
        keep_prob = tf.placeholder(tf.float32)

        ln_w = tf.Variable(
            tf.truncated_normal([embedding_size, nb_linear_inside], stddev=1.0 / math.sqrt(embedding_size)))
        ln_b = tf.Variable(tf.zeros([nb_linear_inside]))

        sent_w = tf.Variable(tf.truncated_normal([nb_lstm_inside, nb_sentiment_label],
                                                 stddev=1.0 / math.sqrt(2 * nb_lstm_inside)))
        sent_b = tf.Variable(tf.zeros([nb_sentiment_label]))

        y_labels = tf.one_hot(tf_y_train, nb_sentiment_label,
                              on_value=1.0,
                              off_value=0.0,
                              axis=-1)

        X_train = tf.transpose(tf_X_train, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        X_train = tf.reshape(X_train, [-1, embedding_size])
        X_train = tf.add(tf.matmul(X_train, ln_w), ln_b)
        X_train = tf.nn.relu(X_train)
        X_train = tf.split(axis=0, num_or_size_splits=seq_max_len, value=X_train)

        # bidirection lstm
        # Creating the forward and backwards cells
        # X_train = tf.stack(X_train)
        # print(X_train.get_shape())
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(nb_lstm_inside, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(nb_lstm_inside, forget_bias=1.0)
        # Pass lstm_fw_cell / lstm_bw_cell directly to tf.nn.bidrectional_rnn
        # if only a single layer is needed
        lstm_fw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell] * layers)
        lstm_bw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell] * layers)
        # Get lstm cell output
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_multicell,
                                                                lstm_bw_multicell,
                                                                X_train, dtype='float32')
        # output_fw, output_bw = outputs
        # outputs = tf.multiply(output_fw, output_bw)
        # outputs = tf.concat(outputs, 2)
        output_fw, output_bw = tf.split(outputs, [nb_lstm_inside, nb_lstm_inside], 2)
        sentiment = tf.reshape(tf.add(output_fw, output_bw), [-1, nb_lstm_inside])
        # sentiment = tf.multiply(sentiment, tf_X_train_mask)
        # sentiment = tf.reduce_mean(sentiment, reduction_indices=1)
        # sentiment = outputs[-1]
        sentiment = tf.nn.dropout(sentiment, keep_prob)
        sentiment = tf.add(tf.matmul(sentiment, sent_w), sent_b)
        sentiment = tf.split(axis=0, num_or_size_splits=seq_max_len, value=sentiment)

        # change back dimension to [batch_size, n_step, n_input]
        sentiment = tf.stack(sentiment)
        sentiment = tf.transpose(sentiment, [1, 0, 2])
        sentiment = tf.multiply(sentiment, tf.expand_dims(tf_X_binary_mask, 2))

        cross_entropy = tf.reduce_mean(
            tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=sentiment, labels=y_labels), tf_X_train_mask))
        prediction = tf.argmax(tf.nn.softmax(sentiment), 2)
        correct_prediction = tf.reduce_sum(
            tf.multiply(tf.cast(tf.equal(prediction, tf_y_train), tf.float32), tf_X_binary_mask))
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.65, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

        saver = tf.train.Saver()

    # Training

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        x_test = list()
        for i in range(len(train_data)):
            sentence = list()
            for word_id in train_data[i]:
                sentence.append(embedding[word_id])
            x_test.append(sentence)

        x_train = list()
        for i in range(len(train_data)):
            sentence = list()
            for word_id in train_data[i]:
                sentence.append(embedding[word_id])
            x_train.append(sentence)

        if flag_train:
            loss_list = list()
            accuracy_list = list()

            for it in range(TRAINING_ITERATIONS):
                # generate batch (x_train, y_train, seq_lengths_train)
                if it * batch_size % nb_sample_train + batch_size < nb_sample_train:
                    index = it * batch_size % nb_sample_train
                else:
                    index = nb_sample_train - batch_size

                _, correct_prediction_train, cost_train = sess.run([optimizer, correct_prediction, cross_entropy],
                                                                   feed_dict={tf_X_train: np.asarray(
                                                                       x_train[index: index + batch_size]),
                                                                       tf_X_train_mask: np.asarray(
                                                                           train_mask[index: index + batch_size]),
                                                                       tf_X_binary_mask: np.asarray(
                                                                           train_binary_mask[
                                                                           index: index + batch_size]),
                                                                       tf_y_train: np.asarray(
                                                                           train_label[index: index + batch_size]),
                                                                       keep_prob: 1.0})

                print('training_accuracy => %.3f, cost value => %.5f for step %d, learning_rate => %.5f' % \
                      (float(correct_prediction_train) / np.sum(
                          np.asarray(train_binary_mask[index: index + batch_size])),
                       cost_train, it, learning_rate.eval()))

                loss_list.append(cost_train)
                accuracy_list.append(
                    float(correct_prediction_train) / np.sum(np.asarray(train_binary_mask[index: index + batch_size])))

                if it % 50 == 0:
                    correct_prediction_test, cost_test = sess.run([correct_prediction, cross_entropy],
                                                                  feed_dict={tf_X_train: np.asarray(x_test),
                                                                             tf_X_train_mask: np.asarray(train_mask),
                                                                             tf_X_binary_mask: np.asarray(
                                                                                 train_binary_mask),
                                                                             tf_y_train: np.asarray(train_label),
                                                                             keep_prob: 1.0})

                    print('test accuracy => %.3f , cost value  => %.5f' % (
                        float(correct_prediction_test) / np.sum(train_binary_mask), cost_test))

                    plt.plot(accuracy_list)
                    axes = plt.gca()
                    axes.set_ylim([0, 1.2])
                    plt.title('batch train accuracy')
                    plt.ylabel('accuracy')
                    plt.xlabel('step')
                    plt.savefig('accuracy.png')
                    plt.close()

                    plt.plot(loss_list)
                    plt.title('batch train loss')
                    plt.ylabel('loss')
                    plt.xlabel('step')
                    plt.savefig('loss.png')
                    plt.close()

            saver.save(sess, '../ckpt/se-apect-v0.ckpt')
        else:
            saver.restore(sess, '../ckpt/se-apect-v0.ckpt')

        correct_prediction_test, prediction_test = sess.run([correct_prediction, prediction],
                                                            feed_dict={tf_X_train: np.asarray(x_test),
                                                                       tf_X_train_mask: np.asarray(train_mask),
                                                                       tf_X_binary_mask: np.asarray(train_binary_mask),
                                                                       tf_y_train: np.asarray(train_label),
                                                                       keep_prob: 1.0})

        print('test accuracy => %.3f' % (float(correct_prediction_test) / np.sum(train_binary_mask)))
        f_result = codecs.open('../result/result', 'w', 'utf-8')
        f_result.write('#------------------------------------------------------------------------------------------#\n')
        f_result.write('#\t author: BinhDT\n')
        f_result.write(
            '#\t test accuracy %.2f\n' % (float(correct_prediction_test) * 100 / np.sum(np.asarray(train_mask) > 0.)))
        f_result.write(
            '#\t 1:positive, 2:negative\n')
        f_result.write('#------------------------------------------------------------------------------------------#\n')

        for i in range(len(train_data)):
            data_sample = ''
            for j in range(len(train_data[i])):
                if word_dict_rev[train_data[i][j]] == '<unk>':
                    continue
                elif train_mask[i][j] > 0.:
                    data_sample = data_sample + word_dict_rev[train_data[i][j]] + \
                                  '(label ' + str(train_label[i][j]) + \
                                  '|predict ' + str(prediction_test[i][j]) + ') '
                else:
                    data_sample = data_sample + word_dict_rev[train_data[i][j]] + ' '
            f_result.write('%s\n' % data_sample.replace('<padding>', ''))

        f_result.close()
