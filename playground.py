import nltk
import os

# sentence1 = "1 this is a new model"
# sentence2 = "2 this is not a good model and we need to find another one"
#
# tokens1 = nltk.word_tokenize(sentence1)
# tokens2 = nltk.word_tokenize(sentence2)
#
# print tokens1
# print tokens2
#
# shared = list(set(tokens1) & set(tokens2))
# print shared

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
