__author__ = 'harsh'

import re
import sys
import random
import operator
import nltk
import math
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


def unk_unigram(word_list, unigrams_dict):
    unigrams_unk_cnt = 0
    for word in word_list:
        if word not in unigrams_dict.keys():
            unigrams_unk_cnt += 1

    unigrams_dict['<UNK>'] = unigrams_unk_cnt
    return unigrams_dict


def unk_bigram(word_list, unigrams_dict, bigrams_dict):
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
    return bigrams_dict

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

def bigram_good_turing_prob(bigrams_dict, bigrams_prob_dict, bigram_gt_list):
    #c* = (c+1) Nc+1/ Nc
    # Pgt = c*/N
    total_N_bigram = sum(bigrams_dict.itervalues)

    for key, value in bigrams_dict.iteritems():
        if key in bigram_gt_list:
            prob_good_turing = value / total_N_bigram
            bigrams_prob_dict[key] = prob_good_turing

    return bigrams_prob_dict

def uni_bigram_good_turing_count(unigrams_dict, bigrams_dict):
    #c* = (c+1) Nc+1/ Nc
    # Pgt = c*/N
    unigram_freq_hash = {}
    bigram_freq_hash = {}
    bigram_gt_list = []

    for key, c in unigrams_dict.iteritems():
        if c == 0:
            freq_c_1 = unigram_calculate_freq(c+1, unigram_freq_hash, unigrams_dict)
            unigram_freq_hash[c] = freq_c_1
            if freq_c_1 != 0:
                unigrams_dict[key] = freq_c_1
        else:
            if c > 0 and c < 5:
                freq_c = unigram_calculate_freq(c, unigram_freq_hash, unigrams_dict)
                freq_c_1 = unigram_calculate_freq(c+1, unigram_freq_hash, unigrams_dict)
                unigram_freq_hash[c] = freq_c_1
                if freq_c_1 != 0:
                    updated_c = ((c + 1) * freq_c_1) / freq_c
                    unigrams_dict[key] = updated_c

    for key, c in bigrams_dict.iteritems():
        if c == 0:
            freq_c_1 = bigrams_calculate_freq(c+1, bigram_freq_hash, bigrams_dict)
            bigram_freq_hash[c] = freq_c_1
            if freq_c_1 != 0:
                bigrams_dict[key] = freq_c_1
                bigram_gt_list.append(key)

        else:
            if c > 0 and c < 5:
                freq_c = bigrams_calculate_freq(c, bigram_freq_hash, bigrams_dict)
                freq_c_1 = bigrams_calculate_freq(c+1, bigram_freq_hash, bigrams_dict)
                bigram_freq_hash[c] = freq_c_1
                if freq_c_1 != 0:
                    updated_c = ((c + 1) * freq_c_1) / freq_c
                    bigrams_dict[key] = updated_c
                    bigram_gt_list.append(key)

    return unigrams_dict, bigrams_dict, bigram_gt_list

def perplexity(count_sentence, file_contents, unigrams_probability_dict, bigrams_probability_dict):

    key_set = unigrams_probability_dict.keys()
    unigram_sentence_prob = 0
    bigram_sentence_prob = 0
    word_list = preprocess(file_contents)
    for item in word_list:
        if item != '<s>' or item != '</s>':
            if item in unigrams_probability_dict:
                word_prob = math.log(unigrams_probability_dict[item], 2) * unigrams_probability_dict[item]
            else:
                word_prob = math.log(unigrams_probability_dict['<UNK>'],2) * unigrams_probability_dict[item]
            unigram_sentence_prob = unigram_sentence_prob + word_prob
    #unigram_test_dict, bigram_test_dict = create_dict(file_contents)

    for i in range(len(word_list)-1):
        key = (word_list[i], word_list[i+1])
        if key not in bigrams_probability_dict:
            key = ('<UNK>', '<UNK>')

            if word_list[i] not in key_set:
                if word_list[i+1] in key_set:
                    key =('<UNK>', word_list[i+1])

            else:
                if word_list[i+1] not in key_set:
                    key =(word_list[i], '<UNK>')

        bi_word_prob = math.log(bigrams_probability_dict[key], 2) * bigrams_probability_dict[key]
        bigram_sentence_prob = bigram_sentence_prob + bi_word_prob

    print "Unigram Perplexity for test set"
    print(2 ** (-1 * unigram_sentence_prob))

    print "Bigram Perplexity for test set"
    print(2 ** (-1 * bigram_sentence_prob))

def lang_model(file_contents_train, file_contents_validation):

    #Calculate the unigram and bigram count on training set
    unigrams_dict,bigrams_dict = create_dict(file_contents_train)

    #Calculate unigram and bigram count from validation set accounting only unknown words
    word_list = preprocess(file_contents_validation)
    unigrams_dict = unk_unigram(word_list, unigrams_dict)
    bigrams_dict = unk_bigram(word_list, unigrams_dict, bigrams_dict)

    #Adjust the counts for low frequency words using good turing smoothing.
    unigrams_dict, bigrams_dict, bigram_gt_list = uni_bigram_good_turing_count(unigrams_dict, bigrams_dict)

    #Calculate unigram and bigram probabilities
    unigrams_probability_dict, biigrams_probability_dict = uni_bi_grams(unigrams_dict, bigrams_dict)

    #Replace the probabilities for threshold k values under good turing(applies only to bigram)
    bigram_good_turing_prob(bigrams_dict, biigrams_probability_dict, bigram_gt_list)

    #Generate random sentence
    print 'Unigram Random Sentence Generator...'
    for i in range(1, 5):
        unigrams_sent = unigram_random_sentence_generator(unigrams_probability_dict)
        print unigrams_sent
    print 'Bigram Random Sentence Generator...'
    for i in range(1, 5):
        bigrams_sent = bigram_random_sentence_generator(biigrams_probability_dict)
        print bigrams_sent

    # Model is trained on N words
    unigrams_len = sum(unigrams_dict.itervalues())
    count_sentence = unigrams_len / 4

    #Calculate the perplexity on N/4 words sentence
    perplexity(count_sentence)



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