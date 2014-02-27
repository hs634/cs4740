__author__ = 'harsh'

import re
import sys
import random
import operator
import math
from nltk import clean_html, tokenize, PunktWordTokenizer
from collections import Counter, defaultdict
from itertools import tee, islice


def preprocess(file_contents, add_sent_markers=True):
    """
    :rtype : object
    :param file_contents: contents of the file that needs to be preprocessed
    :param add_sent_markers: flag to enable addition of sentence start and end markers. True by default.
    :return: list of tokenized words
    """
    raw = clean_html(file_contents)
    raw = re.sub(r'\d+:\d+|\d+,\d+,|IsTruthFul,IsPositive,review', "", raw)
    sentence_list = tokenize.sent_tokenize(raw)
    if add_sent_markers:
        sentence_list = [('<s> ' + sentence + ' </s>') for sentence in sentence_list]
    word_lists = [PunktWordTokenizer().tokenize(sentence) for sentence in sentence_list]
    word_list = [item for sublist in word_lists for item in sublist]
    return word_list


def generate_bigrams_list(word_list):
    """
        Function to generate bigrams given a list of words
    """
    return zip(word_list, word_list[1:])


def create_dict(file_contents, pop_sent_markers=True):
    """
    creates unigram and bigram dictionary containing the counts of each token
    :param file_contents: file contents
    :param pop_sent_markers: flag to keep or remove sentence start and end markers from the dictionaries
    """
    word_list = preprocess(file_contents)
    unigrams_dict = Counter(word_list)
    bigrams_dict = Counter(generate_bigrams_list(word_list))
    if pop_sent_markers:
        bigrams_dict.pop(('<s>', '</s>'), None)
        bigrams_dict.pop(('</s>', '<s>'), None)
    unigrams_dict['<UNK>'] = 0
    bigrams_dict['<UNK>', '<UNK>'] = 0
    return unigrams_dict, bigrams_dict


def uni_bi_grams(unigrams_dict, bigrams_dict):
    """
    Creates unigram and bigram probability
    :param unigrams_dict: unigram counts
    :param bigrams_dict: bigrams
    """
    unigrams_len = sum(unigrams_dict.itervalues())
    unigrams_probability_dict = {}

    for key, value in unigrams_dict.iteritems():
        unigrams_probability_dict[key] = round(value / float(unigrams_len), 6)
    bigrams_probability_dict = {}

    for key, value in bigrams_dict.iteritems():
        bigrams_probability_dict[key] = round(value / float(unigrams_dict[key[0]]), 6)

    return unigrams_probability_dict, bigrams_probability_dict


def unigram_random_sentence_generator(unigrams_prob_dict, sent_length=20):
    """
    unigram random sentence generator
    :param unigrams_prob_dict: unigram probability dictionary
    :param sent_length: default sentence length
    :return: unigram generated random sentence
    """
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
    """
    bigram random sentence generator
    :param bigrams_prob_dict: bigram probability dictionary
    :param sent_length: default sentence length
    :return: bigram generated random sentence
    """
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
    """

    :param filename: name of the file.
    :return: file contents
    """
    try:
        file_obj = open(filename)
    except IOError:
        print "ERR: %s not found in current directory", sys.argv[1]
        sys.exit()
    return file_obj.read()


def unk_unigram(word_list, unigrams_dict):
    """
    Add Unknowns from validation file
    :param word_list: list of words
    :param unigrams_dict: unigram dictionary
    :return: unigram dict is unknown
    """
    unigrams_unk_cnt = 0
    for word in word_list:
        if word not in unigrams_dict.keys():
            unigrams_unk_cnt += 1

    unigrams_dict['<UNK>'] = unigrams_unk_cnt
    return unigrams_dict


def unk_bigram(word_list, unigrams_dict, bigrams_dict):
    """
    Add Unknowns from validation file
    :param word_list: list of words
    :param unigrams_dict: unigram dictionary
    :param bigrams_dict: bigram dictionary
    :return: bigram dict with unknown
    """
    bigram_unk_hash = defaultdict(int)
    key_set = unigrams_dict.keys()
    for i in range(len(word_list) - 1):
        if word_list[i] not in key_set:
            if word_list[i + 1] not in key_set:
                bigram_unk_hash[('<UNK>', '<UNK>')] += 1
            else:
                bigram_unk_hash[('<UNK>', word_list[i + 1])] += 1
        else:
            if word_list[i + 1] not in key_set:
                bigram_unk_hash[(word_list[i], '<UNK>')] += 1

    bigrams_dict = dict(bigram_unk_hash.items() + bigrams_dict.items())
    return bigrams_dict


def unigram_calculate_freq(frequency, unigram_freq_hash, unigrams_dict):
    """
    frequency calculation
    :param frequency:
    :param unigram_freq_hash:
    :param unigrams_dict:
    :return:
    """
    count = 0
    if frequency in unigram_freq_hash:
        return unigram_freq_hash.get(frequency)
    for value in unigrams_dict.itervalues():
        if value == frequency:
            count += 1
    return count


def bigrams_calculate_freq(frequency, bigram_freq_hash, bigrams_dict):
    """
    frequency calculation
    :param frequency:
    :param bigram_freq_hash:
    :param bigrams_dict:
    :return:
    """
    count = 0
    if frequency in bigram_freq_hash:
        return bigram_freq_hash.get(frequency)
    for value in bigrams_dict.itervalues():
        if value == frequency:
            count += 1
    bigram_freq_hash[frequency] = count
    return count


def bigram_good_turing_prob(bigrams_dict, bigrams_prob_dict, bigram_gt_list):
    """
    Bigram good turing implementation
    :param bigrams_dict:
    :param bigrams_prob_dict:
    :param bigram_gt_list:
    :return:
    """
    total_N_bigram = sum(bigrams_dict.itervalues())
    for key, value in bigram_gt_list.iteritems():
        prob_good_turing = float(value) / total_N_bigram
        bigrams_prob_dict[key] = prob_good_turing

    return bigrams_prob_dict


def uni_bigram_good_turing_count(unigrams_dict, bigrams_dict):
    """
    Unigram and bigram good turing count calculation
    :param unigrams_dict:
    :param bigrams_dict:
    :return:
    """
    unigram_freq_hash = {}
    bigram_freq_hash = {}
    bigram_gt_list = defaultdict(float)
    unigram_freq_dict = Counter(unigrams_dict.itervalues())
    bigrams_freq_dict = Counter(bigrams_dict.itervalues())

    for key, c in unigrams_dict.iteritems():
        if c < 1:
            freq_c_1 = unigram_freq_dict[c + 1]
            if freq_c_1 != 0:
                unigrams_dict[key] = freq_c_1
        else:
            if 1 <= c < 5:
                freq_c = unigram_freq_dict[c]
                freq_c_1 = unigram_freq_dict[c + 1]
                unigram_freq_hash[c + 1] = freq_c_1
                if freq_c_1 != 0:
                    updated_c = ((c + 1) * freq_c_1) / float(freq_c)
                    unigrams_dict[key] = updated_c

    for key, c in bigrams_dict.iteritems():
        if c < 1:
            freq_c_1 = bigrams_freq_dict[c + 1]
            if freq_c_1 != 0:
                bigrams_dict[key] = freq_c_1
                bigram_gt_list[key] = freq_c_1

        else:
            if 1 <= c < 5:
                freq_c = bigrams_freq_dict[c]
                freq_c_1 = bigrams_freq_dict[c + 1]
                if freq_c_1 != 0:
                    updated_c = ((c + 1) * freq_c_1) / float(freq_c)
                    bigrams_dict[key] = updated_c
                    bigram_gt_list[key] = updated_c

    return unigrams_dict, bigrams_dict, bigram_gt_list


def perplexity(file_contents_test, unigrams_probability_dict, bigrams_probability_dict):
    """
    Perplexity calculation
    :param file_contents_test:
    :param unigrams_probability_dict:
    :param bigrams_probability_dict:
    :return:
    """
    key_set = unigrams_probability_dict.keys()
    unigram_sentence_prob = 0
    bigram_sentence_prob = 0
    word_list = preprocess(file_contents_test)
    for item in word_list:
        #if item != '<s>' or item != '</s>':
        if item in unigrams_probability_dict:
            word_prob = math.log(unigrams_probability_dict[item], 2)
        else:
            word_prob = math.log(unigrams_probability_dict['<UNK>'], 2)
        unigram_sentence_prob += word_prob

    N = float(len(word_list))

    logP = round(((-1 * unigram_sentence_prob) / N), 6)
    # print 'unigram log probability'
    # print logP

    print "Unigram Perplexity for test set"
    unigram_perplexity = 2 ** (logP)
    print unigram_perplexity

    for i in range(len(word_list) - 1):
        key = (word_list[i], word_list[i + 1])
        if key not in bigrams_probability_dict:
            key = ('<UNK>', '<UNK>')

            if word_list[i] not in key_set:
                if word_list[i + 1] in key_set:
                    key = ('<UNK>', word_list[i + 1])

            else:
                if word_list[i + 1] not in key_set:
                    key = (word_list[i], '<UNK>')
        if key not in bigrams_probability_dict:
            key = ('<UNK>', '<UNK>')

        bi_word_prob = math.log(bigrams_probability_dict[key], 2)
        bigram_sentence_prob += bi_word_prob

    logP = round(((-1 * bigram_sentence_prob) / N), 6)
    #print 'bigram log probability'
    #print logP
    print "Bigram Perplexity for test set"
    bigram_perplexity = (2 ** (logP))
    print bigram_perplexity
    return unigram_perplexity, bigram_perplexity


def lang_model(file_contents_train, file_contents_validation, file_contents_test):
    """
    Method to generate the language model, smoothing, perplexity and random sentences
    :param file_contents_train:
    :param file_contents_validation:
    :param file_contents_test:
    """
    print "Calculate the unigram and bigram count on training set"
    unigrams_dict, bigrams_dict = create_dict(file_contents_train)

    # print 'Calculate unigram and bigram count from validation set accounting only unknown words'
    # word_list = preprocess(file_contents_validation)
    # unigrams_dict = unk_unigram(word_list, unigrams_dict)
    # bigrams_dict = unk_bigram(word_list, unigrams_dict, bigrams_dict)

    print 'Adjust the counts for low frequency words using good turing smoothing.'
    unigrams_dict, bigrams_dict, bigram_gt_list = uni_bigram_good_turing_count(unigrams_dict, bigrams_dict)

    print 'Calculate unigram and bigram probabilities'
    unigrams_probability_dict, bigrams_probability_dict = uni_bi_grams(unigrams_dict, bigrams_dict)

    print 'Replace the probabilities for threshold k values under good turing(applies only to bigram)'
    bigram_good_turing_prob(bigrams_dict, bigrams_probability_dict, bigram_gt_list)

    print 'Generate random sentence'
    print 'Unigram Random Sentence Generator...'
    for i in range(1, 5):
        unigrams_sent = unigram_random_sentence_generator(unigrams_probability_dict)
        print unigrams_sent
    print 'Bigram Random Sentence Generator...'
    for i in range(1, 5):
        bigrams_sent = bigram_random_sentence_generator(bigrams_probability_dict)
        print bigrams_sent

    print 'Calculate the perplexity'
    perplexity(file_contents_test, unigrams_probability_dict, bigrams_probability_dict)


def preprocess_hotel_review(file_contents, file_contents_test):
    """
    Hotel review preprocess and truthfulness of the hotel review
    :param file_contents:
    :param file_contents_test:
    """
    raw = clean_html(file_contents)
    raw = re.sub(r'IsTruthFul,IsPositive,review', "", raw)
    sentence_list = tokenize.line_tokenize(raw)
    print sentence_list
    truth_sentences = []
    false_sentences = []
    for sentence in sentence_list:
        sent_arr = re.split(r',', sentence)
        try:
            is_truthful = int(sent_arr[0])
        except ValueError:
            print "is_truthful is not an integer"

        if is_truthful == 1:
            truth_sentences.append(sent_arr[2])
        elif is_truthful == 0:
            false_sentences.append(sent_arr[2])

    truth_uni_prob_dict, truth_bi_prob_dict = process_prob(" ".join(truth_sentences))
    false_uni_prob_dict, false_bi_prob_dict = process_prob(" ".join(false_sentences))

    raw_test = clean_html(file_contents_test)
    raw_test = re.sub(r'IsTruthFul,review', "", raw_test)
    sentence_list_test = tokenize.line_tokenize(raw_test)
    test_list = []
    test_truth_false_list = []
    truth_count = false_count = i = 0
    for sentence in sentence_list_test:
        sent_arr = re.split(r',', sentence)
        truth_uni_perplex, truth_bi_perplex = perplexity(sent_arr[1], truth_uni_prob_dict, truth_bi_prob_dict)
        false_uni_perplex, false_bi_perplex = perplexity(sent_arr[1], false_uni_prob_dict, false_bi_prob_dict)
        test_list.append((sent_arr[1], truth_bi_perplex, false_bi_perplex))
        truth_or_false = 1 if truth_bi_perplex < false_bi_perplex else 0
        #truth_or_false = 1 if truth_uni_perplex < false_uni_perplex else 0
        if truth_or_false:
            truth_count += 1
        else:
            false_count += 1
        test_truth_false_list.append([i, truth_or_false])
        i += 1

    import csv

    with open("kaggle_sharp.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows([['Id', 'Label']])
        writer.writerows(test_truth_false_list)
    print test_list
    print test_truth_false_list
    print truth_count
    print false_count


def process_prob(file_contents):

    """
    Generate the probabilites of unigram and bigram and apply good turing
    :param file_contents:
    :return:
    """
    print "Calculate the unigram and bigram count on training set"
    unigrams_dict, bigrams_dict = create_dict(file_contents)

    # print 'Calculate unigram and bigram count from validation set accounting only unknown words'
    # word_list = preprocess(file_contents_validation)
    # unigrams_dict = unk_unigram(word_list, unigrams_dict)
    # bigrams_dict = unk_bigram(word_list, unigrams_dict, bigrams_dict)

    print 'Adjust the counts for low frequency words using good turing smoothing.'
    unigrams_dict, bigrams_dict, bigram_gt_list = uni_bigram_good_turing_count(unigrams_dict, bigrams_dict)

    print 'Calculate unigram and bigram probabilities'
    unigrams_probability_dict, bigrams_probability_dict = uni_bi_grams(unigrams_dict, bigrams_dict)

    print 'Replace the probabilities for threshold k values under good turing(applies only to bigram)'
    bigram_good_turing_prob(bigrams_dict, bigrams_probability_dict, bigram_gt_list)

    return unigrams_probability_dict, bigrams_probability_dict


def ngram(lst, n):
    """
    Generic generator for generating ngrams
    :param lst: list of words
    :param n: the 'n' of ngrams
    """
    tlst = lst
    while True:
        a, b = tee(tlst)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            tlst = b
        else:
            break


def trigrams_prob(file_contents, bigrams_dict):
    """
    Trigram extension
    :param file_contents:
    :param bigrams_dict:
    :return:
    """
    word_list = preprocess(file_contents)
    trigrams_dict = Counter(ngram(word_list, 3))
    trigrams_probability_dict = defaultdict(float)
    for key, value in trigrams_dict.iteritems():
        trigrams_probability_dict[key] = round(value / float(bigrams_dict[(key[0], key[1])]), 6)

    return trigrams_probability_dict


def trigram_random_sentence_generator(trigrams_prob_dict, sent_length=30):
    """
    Trigram random sentence generator
    :param trigrams_prob_dict:
    :param sent_length:
    :return:
    """
    sentence = ''
    mru_word = '<s>'
    while len(sentence.split()) < sent_length and '</s>' not in sentence:
        rand_prob = round(random.uniform(0.0, 1.1), 6)
        short_trigram_dict = dict((k, v) for k, v in trigrams_prob_dict.iteritems() if k[0] == mru_word)
        closest_key, closest_val = min(short_trigram_dict.iteritems(), key=lambda (k, v): abs(v - rand_prob))
        for k1, v1 in short_trigram_dict.iteritems():
            if rand_prob - 0.02 <= v1 <= rand_prob + 0.02:
                closest_key, closest_value = k1, v1

        if '<s>' not in sentence:
            if closest_key[0] == '<s>':
                sentence += closest_key[0] + " " + closest_key[1] + " " + closest_key[2]
                mru_word = closest_key[2]
            else:
                continue
        else:
            sentence += " " + closest_key[1] + " " + closest_key[2]
            mru_word = closest_key[2]
    return re.sub('<[^<]+?>', "", re.sub(' +([,:!?])', r'\1', sentence))


def trigram_extension(file_contents_train):
    """
    Method for trigram extension
    :param file_contents_train:
    """
    unigrams_dict, bigrams_dict = create_dict(file_contents_train, False)
    trigram_prob_dict = trigrams_prob(file_contents_train, bigrams_dict)
    for i in range(1, 5):
        tri_sent = trigram_random_sentence_generator(trigram_prob_dict)
        print tri_sent


def main():
    if len(sys.argv) < 2:
        print "Please enter the names of training validation and test files names as command line arguments."
        sys.exit()
    #read all three files
    file_contents_train = read_file(sys.argv[1])
    file_contents_validation = read_file(sys.argv[2])
    file_contents_test = read_file(sys.argv[3])
    #language model, smoothing and good turing runner
    lang_model(file_contents_train, file_contents_validation, file_contents_test)
    #hotel review truthfulness runner
    preprocess_hotel_review(file_contents_train, file_contents_test)
    #trigram extension runner
    trigram_extension(file_contents_train)


main()