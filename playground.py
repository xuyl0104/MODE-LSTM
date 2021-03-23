import nltk
import os
import keras
import tensorflow as tf
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from main_yelp import model
from odelstm import *
import reader
from text_process_yelp import build_dataset, load_data, load_vocab

'''
dataset parameters
'''
embedding_dim = 300  # the dimension of word embedding
char_embedding_dim = 50  # the dimension of char embedding
s_maxlen = 20  # the maximun length of sentence
w_maxlen = 18  # the maximun length of word
num_class = 2
char_size = 44

'''
model parameters
'''
emb_dropout = 0.2
char_emb_dropout = 0.2
input_dropout = 0.0
recurrent_dropout = 0.0
output_dropout = 0.5
scales = [5, 10, 15]  # the window size of each scale
scale_nums = [2, 2, 2]  # the number of small hidden states for each ODE-LSTM
units = 50  # the dimension of each small hidden state
l2 = 0.001
lamda = 0.01  # the balance factor of penalization loss
output_size = 150  # the size of MLP
filters = 50  # output dim of char embedding
width = 3  # kernel size of conv

data_root = 'data/yelp/embeddings/'
vector_path = data_root + 'glove.filtered.npz'
train_path = data_root + 'train_word2idx.json'
dev_path = data_root + 'dev_word2idx.json'
test_path = data_root + 'test_word2idx.json'
gen_path = data_root + 'gen_word2idx.json'


def yelp_data_prep():
    all_train = './data/yelp/yelp_all_train.txt'
    all_test = './data/yelp/yelp_all_test.txt'
    all_dev = './data/yelp/yelp_all_dev.txt'

    train_pos = './data/yelp/yelp_train_pos.txt'
    train_neg = './data/yelp/yelp_train_neg.txt'
    test_pos = './data/yelp/yelp_test_pos.txt'
    test_neg = './data/yelp/yelp_test_neg.txt'
    dev_pos = './data/yelp/yelp_dev_pos.txt'
    dev_neg = './data/yelp/yelp_dev_neg.txt'

    # make files
    if os.path.exists(all_train):
        os.remove(all_train)
    if os.path.exists(all_test):
        os.remove(all_test)
    if os.path.exists(all_dev):
        os.remove(all_dev)

    # train data
    with open(train_pos) as trainpos:
        for line in trainpos:
            with open(all_train, 'a') as alltrain:
                alltrain.write("1 " + line)

    with open(train_neg) as trainneg:
        for line in trainneg:
            with open(all_train, 'a') as alltrain:
                alltrain.write("0 " + line)

    # test data
    with open(test_pos) as testpos:
        for line in testpos:
            with open(all_test, 'a') as alltest:
                alltest.write("1 " + line)

    with open(test_neg) as testneg:
        for line in testneg:
            with open(all_test, 'a') as alltest:
                alltest.write("0 " + line)

    # dev data
    with open(dev_pos) as devpos:
        for line in devpos:
            with open(all_dev, 'a') as alldev:
                alldev.write("1 " + line)

    with open(dev_neg) as devneg:
        for line in devneg:
            with open(all_dev, 'a') as alldev:
                alldev.write("0 " + line)


def yelp_combine_pos_neg(pos_file, neg_file, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(pos_file) as trainpos:
        for line in trainpos:
            with open(output_file, 'a') as alltrain:
                alltrain.write("1 " + line)

    with open(neg_file) as trainneg:
        for line in trainneg:
            with open(output_file, 'a') as alltrain:
                alltrain.write("0 " + line)


loss, modelstm_model = model(scales, scale_nums, units, emb_dropout, char_emb_dropout, output_dropout)
modelstm_model.load_weights('./save/yelp_bs256/mode-lstm')

# test_sentences, test_words, test_labels = reader.load_data_wc(test_path, shuffle=True)
# test_sen, test_word, test_label = reader.prepare_data_wc(test_sentences, test_words, test_labels, s_maxlen=s_maxlen, w_maxlen=w_maxlen)
# one_hot_label_test = keras.utils.to_categorical(test_label, num_class)
# loss, acc = modelstm_model.evaluate([test_sen, test_word], one_hot_label_test)
# print('loss : {}, test acc : {}'.format(loss, acc))

generated_pos_file = './data/yelp_generated/gen_pos.txt'
generated_neg_file = './data/yelp_generated/gen_neg.txt'
gen_set_file = './data/yelp_generated/gen.txt'
yelp_combine_pos_neg(generated_pos_file, generated_neg_file, gen_set_file)
genset, _ = load_data(gen_set_file)
word_vocab, _ = load_vocab(os.path.join(data_root, 'words.vocab'))
char_vocab, _ = load_vocab(os.path.join(data_root, 'chars.vocab'))
build_dataset(genset, os.path.join(data_root, 'gen_word2idx.json'), word_vocab, char_vocab,num_labels=num_class,one_hot=False)

gen_sentences, gen_words, gen_labels = reader.load_data_wc(gen_path, shuffle=True)
gen_sen, gen_word, gen_label = reader.prepare_data_wc(gen_sentences, gen_words, gen_labels, s_maxlen=s_maxlen, w_maxlen=w_maxlen)
one_hot_label_gen = keras.utils.to_categorical(gen_label, num_class)
loss, acc = modelstm_model.evaluate([gen_sen, gen_word], one_hot_label_gen)
print('loss : {}, test acc : {}'.format(loss, acc))
