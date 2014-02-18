__author__ = 'harsh'

import re
import sys
import random
import operator

from nltk import clean_html, tokenize, PunktWordTokenizer
from collections import Counter


def preprocess(file_contents, add_sent_markers=True):
    raw = clean_html(file_contents)
    raw = re.sub('\d:\d |\d,\d,|IsTruthFul,IsPositive,review', "", raw)
    sentence_list = tokenize.sent_tokenize(raw)
    print sentence_list
    if add_sent_markers:
        sentence_list = [('<s> ' + sentence + ' </s>') for sentence in sentence_list]
    word_lists = [PunktWordTokenizer().tokenize(sentence) for sentence in sentence_list]
    word_list = [item.lower() for sublist in word_lists for item in sublist]
    return word_list


def generate_bigrams_list(word_list):
    return zip(word_list, word_list[1:])


def uni_bi_grams(file_contents):
    word_list = preprocess(file_contents)
    unigrams_dict = Counter(word_list)
    bigrams_dict = Counter(generate_bigrams_list(word_list))
    unigrams_len = sum(unigrams_dict.itervalues())

    unigrams_probability_dict = {}
    for key, value in unigrams_dict.iteritems():
        unigrams_probability_dict[key] = round(value/float(unigrams_len), 6)

    bigrams_probability_dict = {}
    for key, value in bigrams_dict.iteritems():
        bigrams_probability_dict[key] = round(value/float(unigrams_dict[key[0]]), 4)

    return unigrams_probability_dict, bigrams_probability_dict


def unigram_random_sentence_generator(unigrams_prob_dict, sent_length=20):
    sentence = ''
    padding = 0.05
    max_unigrams_prob = max(unigrams_prob_dict.iteritems(), key=operator.itemgetter(1))
    while len(sentence.split()) < sent_length:
        rand_prob = random.uniform(0.0, max_unigrams_prob[1] + padding)
        closest_key, closest_val = min(unigrams_prob_dict.iteritems(), key=lambda (k, v): abs(v - rand_prob))
        if abs(rand_prob - closest_val) > 0.01:
            continue
        if closest_key == '</s>':
            break
        sentence = sentence + ' ' + closest_key

    return re.sub('<[^<]+?>', '', sentence)


def bigram_random_sentence_generator(bigrams_prob_dict, sent_length=30):
    sentence = ''
    mru_word = '<s>'
    while len(sentence.split()) < sent_length and '</s>' not in sentence:
        rand_prob = random.uniform(0.0, 1.0)
        short_bigram_dict = dict((k, v) for k, v in bigrams_prob_dict.iteritems() if k[0] == mru_word)
        closest_key, closest_val = min(short_bigram_dict.iteritems(), key=lambda (k, v): abs(v - rand_prob))
        if '<s>' not in sentence:
            if closest_key[0] == '<s>':
                sentence += closest_key[0] + " " + closest_key[1]
            else:
                continue
        else:
            mru_word = sentence.split()[-1]
            if closest_key[0] == mru_word:
                sentence += " " + closest_key[1]
            else:
                continue

    return re.sub('<[^<]+?>', "", sentence)


def main():
    if len(sys.argv) < 2:
        print "Please enter the name of a corpus file as a command line argument."
        sys.exit()

    try:
        file_obj = open(sys.argv[1])
    except IOError:
        print "ERR: %s not found in current directory", sys.argv[1]
        sys.exit()

    file_contents = file_obj.read()
    unigrams, bigrams = uni_bi_grams(file_contents)
    '''print "Unigrams .........."
    print unigrams
    print "Bigrams................"
    print bigrams'''
    print 'Unigram Random Sentence Generator...'
    unigrams_sent = unigram_random_sentence_generator(unigrams)
    print unigrams_sent
    print 'Bigram Random Sentence Generator...'
    bigrams_sent = bigram_random_sentence_generator(bigrams)
    print bigrams_sent
    return True

main()
