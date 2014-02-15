__author__ = 'harsh'
import nltk
from itertools import tee, islice
from collections import Counter

def preprocess(filename):
    f = open(filename)
    html = f.read()
    raw = nltk.clean_html(html)
    words = raw.split()
    return list(words)

def ngram(lst, n):
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



def ngrams(filename, n):
    filename = 'test1.html'
    ngrams_dict = Counter(ngram(preprocess(filename), n))
    ngrams_len = sum(ngrams_dict.itervalues())

    ngrams_probability_dict = {}
    for key, value in ngrams_dict.iteritems():
        ngrams_probability_dict[key] = value/float(ngrams_len)

    return ngrams_probability_dict

def main():
    unigrams = ngrams('test1.html', 1)
    bigrams = ngrams('test1.html', 2)
    print "Unigrams .........."
    print unigrams
    print "Bigrams................"
    print bigrams
    return 1

main()
