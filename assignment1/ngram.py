__author__ = 'harsh'

import re
import sys
import random
import operator
import nltk
from nltk import clean_html, tokenize, PunktWordTokenizer
from collections import Counter, defaultdict


def preprocess(file_contents, add_sent_markers=True):
    raw = clean_html(file_contents)
    raw = re.sub(r'\d+:\d+|\d+,\d+,|IsTruthFul,IsPositive,review', "", raw)
    sentence_list = tokenize.sent_tokenize(raw)
    if add_sent_markers:
        sentence_list = [('<s> ' + sentence + ' </s>') for sentence in sentence_list]
    word_lists = [PunktWordTokenizer().tokenize(sentence) for sentence in sentence_list]
    word_list = [item for sublist in word_lists for item in sublist]
    return word_list


def generate_bigrams_list(word_list):
    return zip(word_list, word_list[1:])

def create_dict(file_contents):
    word_list = preprocess(file_contents)
    unigrams_dict = Counter(word_list)
    bigrams_dict = Counter(generate_bigrams_list(word_list))
    bigrams_dict.pop(('<s>', '</s>'), None)
    bigrams_dict.pop(('</s>', '<s>'), None)
    return unigrams_dict,bigrams_dict

def uni_bi_grams(unigrams_dict, bigrams_dict):

    unigrams_len = sum(unigrams_dict.itervalues())
    unigrams_probability_dict = {}

    for key, value in unigrams_dict.iteritems():
        unigrams_probability_dict[key] = round(value/float(unigrams_len), 6)
    bigrams_probability_dict = {}

    for key, value in bigrams_dict.iteritems():
        bigrams_probability_dict[key] = round(value/float(unigrams_dict[key[0]]), 6)

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

    return re.sub('<[^<]+?>', "", re.sub(' +([,:!?])', r'\1', sentence))


def bigram_random_sentence_generator(bigrams_prob_dict, sent_length=30):
    sentence = ''
    mru_word = '<s>'
    while len(sentence.split()) < sent_length and '</s>' not in sentence:
        rand_prob = round(random.uniform(0.0, 1.1), 6)
        short_bigram_dict = dict((k, v) for k, v in bigrams_prob_dict.iteritems() if k[0] == mru_word)
        closest_key, closest_val = min(short_bigram_dict.iteritems(), key=lambda (k, v): abs(v - rand_prob))
        for k1, v1 in short_bigram_dict.iteritems():
            if rand_prob - 0.4 <= v1 <= rand_prob + 0.4:
                closest_key, closest_value = k1, v1

        if '<s>' not in sentence:
            if closest_key[0] == '<s>':
                sentence += closest_key[0] + " " + closest_key[1]
                mru_word = closest_key[1]
            else:
                continue
        else:
            sentence += " " + closest_key[1]
            mru_word = closest_key[1]
    return re.sub('<[^<]+?>', "", re.sub(' +([,:!?])', r'\1', sentence))


def read_file(filename):
    try:
        file_obj = open(filename)
    except IOError:
        print "ERR: %s not found in current directory", sys.argv[1]
        sys.exit()
    return file_obj.read()


def unk_unigram(word_list, unigrams_dict, unigrams_prob_dict):
    unigrams_unk_cnt = 0
    for word in word_list:
        if word not in unigrams_prob_dict.keys():
            unigrams_unk_cnt += 1

    unigrams_dict['<UNK>'] = unigrams_unk_cnt

    word_list_len = len(word_list)
    unigrams_unk_prob = unigrams_unk_cnt/float(word_list_len)
    unigrams_prob_dict['<UNK>'] = unigrams_unk_prob
    return unigrams_dict, unigrams_prob_dict


def unk_bigram(word_list, unigrams_dict, bigrams_dict, bigrams_prob_dict):
    bigram_unk_hash = defaultdict(int)
    key_set = unigrams_dict.keys()
    for i in range(len(word_list)-1):
        if word_list[i] not in key_set:
            if word_list[i+1] not in key_set:
                bigram_unk_hash[('<UNK>', '<UNK>')] += 1
            else:
                bigram_unk_hash[('<UNK>', word_list[i+1])] += 1
        else:
            if word_list[i+1] not in key_set:
                bigram_unk_hash[(word_list[i], '<UNK>')] += 1

    bigrams_dict = dict(bigram_unk_hash.items() + bigrams_dict.items())

    bigrams_unk_prob = {}
    for key, value in bigram_unk_hash.iteritems():
        bigrams_unk_prob[key] = round(value/float(unigrams_dict[key[0]]), 6)

    bigrams_prob_dict = dict(bigrams_unk_prob.items() + bigrams_prob_dict.items())
    return bigrams_dict, bigrams_prob_dict

def unigram_calculate_freq(frequency, unigram_freq_hash, unigrams_dict):
    count = 0
    if frequency in unigram_freq_hash:
        return unigram_freq_hash.get(frequency)
    for value in unigrams_dict.itervalues():
        if value == frequency:
            count += 1
    return count

def bigrams_calculate_freq(frequency, bigram_freq_hash, bigrams_dict):
    count = 0
    if frequency in bigram_freq_hash:
        return bigram_freq_hash.get(frequency)
    for value in bigrams_dict.itervalues():
        if value == frequency:
            count += 1
    bigram_freq_hash[frequency] = count

def uni_bigram_good_turing(unigrams_dict, bigrams_dict, unigrams_prob_dict, bigrams_prob_dict):
    #c* = (c+1) Nc+1/ Nc
    # Pgt = c*/N
    unigram_freq_hash = {}
    bigram_freq_hash = {}
    total_N_unigram = sum(unigrams_dict.itervalues)
    total_N_bigram = sum(bigrams_dict.itervalues)

    for key, c in unigrams_dict.iteritems():
        if c == 0:
            freq_c_1 = unigram_calculate_freq(c+1, unigram_freq_hash, unigrams_dict)
            unigram_freq_hash[c] = freq_c_1
            prob_good_turing = freq_c_1 / total_N_unigram
            unigrams_prob_dict[key] = prob_good_turing
        else:
            if c > 0 and c < 5:
                freq_c = unigram_calculate_freq(c, unigram_freq_hash, unigrams_dict)
                freq_c_1 = unigram_calculate_freq(c+1, unigram_freq_hash, unigrams_dict)
                unigram_freq_hash[c] = freq_c_1
                prob_good_turing = ((c + 1) * freq_c_1 / freq_c) / total_N_unigram
                unigrams_prob_dict[key] = prob_good_turing

    for key, c in bigrams_dict.iteritems():
        if c == 0:
            freq_c_1 = bigrams_calculate_freq(c+1, bigram_freq_hash, bigrams_dict)
            bigram_freq_hash[c] = freq_c_1
            prob_good_turing = freq_c_1 / total_N_bigram
            bigrams_prob_dict[key] = prob_good_turing
        else:
            if c > 0 and c < 5:
                freq_c = bigrams_calculate_freq(c, bigram_freq_hash, bigrams_dict)
                freq_c_1 = bigrams_calculate_freq(c+1, bigram_freq_hash, bigrams_dict)
                bigram_freq_hash[c] = freq_c_1
                prob_good_turing = ((c + 1) * freq_c_1 / freq_c) / total_N_bigram
                bigrams_prob_dict[key] = prob_good_turing

    return unigrams_prob_dict, bigrams_prob_dict

def main():
    if len(sys.argv) < 2:
        print "Please enter the name of a corpus file as a command line argument."
        sys.exit()


    file_contents = read_file(sys.argv[1])
    unigrams_dict,bigrams_dict = create_dict(file_contents)
    unigrams, bigrams = uni_bi_grams(unigrams_dict, bigrams_dict)
    print 'Unigram Random Sentence Generator...'
    unigrams_sent = unigram_random_sentence_generator(unigrams)
    print unigrams_sent
    print 'Bigram Random Sentence Generator...'
    for i in range(1, 5):
        bigrams_sent = bigram_random_sentence_generator(bigrams)
        print bigrams_sent
    return True

main()
