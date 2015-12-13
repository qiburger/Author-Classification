__author__ = 'Qi_He'

import os, math, operator, sys
from collections import defaultdict, Counter, OrderedDict
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.corpus import wordnet as wn
from nltk import FreqDist, word_tokenize, sent_tokenize
import numpy as np
from subprocess import Popen, PIPE
import cPickle
import re
from nltk.cluster.util import cosine_distance
from nltk.stem.wordnet import WordNetLemmatizer
import time


def tokenize(rawexcerpt):
    """
    :param Rawexcerpt: string input
    :return: Tokenized list of words (maintain capitalized letters)
    """
    decoded_list = word_tokenize(rawexcerpt.decode('utf8'))
    return [word.encode('utf8').lower() for word in decoded_list]


def get_synonyms(word):
    """
    Must input tokenized string, all lower !!
    :param word: Tokenized string word
    :return: set of synonyms, all lowercase
    """
    output = []
    synsets = wn.synsets(word.decode('utf8'))
    if not synsets:
        return set([word.decode("utf8")])
    else:
        for i, j in enumerate(synsets):
            for lemma in j.lemma_names():
                output.append(lemma.lower())
        return set(output)


def get_match_score(paragraph, tokens_by_author, tokens_not_by_author, corpus_distribution, stoplist, lemma_bank):
    """

    :param stoplist: set of relevant stoplist
    :param paragraph: string of the paragraph in test set. (no new line)
    :param tokens_by_author: list of tokenized words from the author (not unique)
    :param tokens_not_by_author: list of tokenized words from other authors
    :param corpus_distribution: global word frequency of tokens in training set + test set
    """

    # Get word distribution for author and non authors
    author_score = 0
    non_author_score = 0

    # Here capitalization matters!
    paragraph_list = tokenize(paragraph)
    unique_test_words = set(paragraph_list)
    unique_author_words = set(tokens_by_author)
    unique_not_author_words = set(tokens_not_by_author)

    paragraph_distribution = Counter(paragraph_list)
    paragraph_total_words = len(paragraph_list)

    author_distribution = Counter(tokens_by_author)
    author_total_words = len(tokens_by_author)

    non_author_distribution = Counter(tokens_not_by_author)
    non_author_total_words = len(tokens_not_by_author)

    for test_word in unique_test_words:
        for author_word in unique_author_words:
            # Here I kept UPPER vs lower case difference
            if test_word == author_word and (test_word not in stoplist):  # stoplist is all lowercase
                # score = min(normalized frequency * 1000) * sum of global freq of syn / global freq of target word
                temp_match_score = min((float(author_distribution[author_word]) * 1000.0 / author_total_words), (
                    float(paragraph_distribution[author_word]) * 1000.0 / paragraph_total_words)) * float(
                    get_sum_synonyms(author_word, corpus_distribution, lemma_bank)) / float(corpus_distribution[author_word])
                # if temp_match_score == 0:
                    # raise ValueError('Something is zero with %s' % (test_word))
                author_score += temp_match_score

        for non_author_word in unique_not_author_words:
            if test_word == non_author_word and (test_word not in stoplist):
                temp_match_score = min((float(non_author_distribution[test_word]) * 1000.0 / non_author_total_words), (
                    float(paragraph_distribution[test_word]) * 1000.0 / paragraph_total_words)) * float(
                    get_sum_synonyms(test_word, corpus_distribution, lemma_bank)) / float(corpus_distribution[test_word])
                non_author_score += temp_match_score

    print author_score
    print non_author_score
    return author_score, non_author_score



def get_sum_synonyms(target_word, corpus_distribution, lemma_bank):
    """
    Get the sum over the global frequencies of all synonyms of target word
    NOTE: here capitalization matters!
    :param target_word: target token
    :param corpus_distribution: Counter dictionary of the corpus
    :param lemma_bank: mapping of lemma to actual tokens in corpus
    """
    out = 0
    # lmtzr = WordNetLemmatizer()
    target_synonym = get_synonyms(target_word)

    set_of_actual_tokens = set()

    for syn in target_synonym:
        set_of_actual_tokens.update(lemma_bank[syn])

    for actual_token in set_of_actual_tokens:
        out += corpus_distribution[actual_token]

    # if out == 0:
        # raise ValueError('Something is zero with %s' % (target_word))
    return out


def build_lemma_bank(corpus_distribution):
    """
    :param corpus_distribution: global freq of words that occur in corpus
    :return: a mapping of lemmas to actual tokens in corpus
    """
    start_time = time.time()

    lemma_bank = defaultdict(list)

    for token in corpus_distribution:
        lemma_list = set()
        synsets = wn.synsets(token.decode('utf8'))

        if not synsets:
            lemma_bank[token.decode('utf8')].append(token)
        else:
            for i, j in enumerate(synsets):
                lemma_list.add(j.name().rsplit('.',2)[0])
            for lemma in lemma_list:
                lemma_bank[lemma].append(token)

    print("--- %s seconds ---" % (time.time() - start_time))
    return lemma_bank


def setup():
    """
    NOTE: Corpus is here defined as training set + test set
    :return: return 3 lists of tokens, one list by the author, the second list of tokens not by the author, and corpus
    """
    kolata_list = []
    kolata_paragraph = []
    non_kolata_list = []
    non_kolata_paragraph = []
    test_paragraphs = []
    test_tokens = []

    counter = 0

    with open("project_articles_train", "r") as fin:
        for line in fin:
            temp = line.rstrip().split('\t')
            if len(temp) != 2:
                raise ValueError("Maybe more than one tab at line " + str(counter))
            if int(temp[1]) == 0:
                non_kolata_list.extend(tokenize(temp[0]))
                non_kolata_paragraph.append(temp[0])
            elif int(temp[1]) == 1:
                kolata_list.extend(tokenize(temp[0]))
                kolata_paragraph.append(temp[0])
            else:
                raise ValueError("Something wrong with the 0/1 at line " + str(counter))
            counter += 1

    with open("project_articles_test", "r") as testin:
        for testline in testin:
            temp = testline.rstrip()
            test_paragraphs.append(temp)
            test_tokens.extend(tokenize(temp))

    # Corpus is here defined as training set + entire test set
    corpus = kolata_list + non_kolata_list + test_tokens
    corpus_distribution = Counter(corpus)

    stop_words = get_stopwords()

    lemma_bank = build_lemma_bank(corpus_distribution)

    # Serialize all relevant lists
    cPickle.dump(kolata_list, open('kolata_list.p', 'wb'))
    cPickle.dump(kolata_paragraph, open('kolata_paragraph.p', 'wb'))

    cPickle.dump(non_kolata_list, open('non_kolata_list.p', 'wb'))
    cPickle.dump(non_kolata_list, open('non_kolata_paragraph.p', 'wb'))

    cPickle.dump(corpus_distribution, open('corpus_distribution.p', 'wb'))
    cPickle.dump(test_paragraphs, open('test_paragraphs.p', 'wb'))
    cPickle.dump(stop_words, open('stop_words.p', 'wb'))
    cPickle.dump(lemma_bank, open('lemma_bank.p', 'wb'))

    print "setup complete"

    return test_paragraphs , kolata_list, non_kolata_list, corpus_distribution, stop_words, kolata_paragraph, non_kolata_paragraph, lemma_bank


def get_stopwords():
    """
    This function builds the stop list using the list given by the professor (but I removed the symbols from the list)
    along with the wn.corpus stop list and the census (chose top 2500 first and last names)

    All the symbols left are ", . : 's"

    :return: a set of stop words. Maintain set structure for better "not in" speed
    """
    out_list = []
    with open("stopwords.txt", "r") as fin:
        for line in fin:
            out_list.append(line.rstrip())
    out_list.extend(stopwords.words('english'))

    # Note that we are adding the top 2500 first and last names (arbitrary number)
    # 1. add the top 1250 common last names
    out_list.extend(common_names("census-dist-2500-last.txt")[:1250])

    # 2. add the top common male and female first names, 625 each
    out_list.extend(common_names("census-dist-female-first.txt")[:625])
    out_list.extend(common_names("census-dist-male-first.txt")[:625])

    return set(out_list)


def common_names(file_address):
    """
    Get the common names from the census file. Each name string is lower cased!!!
    :param file_address: Census txt file detailing the distribution of common names
    :return: The full list of names in the census file
    """
    out_list = []
    with open(file_address, "r") as fin:
        for line in fin:
            out_list.append(line.split()[0].lower())  # Note the lower case here
    return out_list


def run_baseline():
    """
    Baseline algorithm: for each test paragraph, compare match score with author tokens and non author tokens
    Threshold algorithm: test each paragraph by the author again all training set to set thresholds
    """

    test_paragraphs, kolata_list, non_kolata_list, corpus_distribution, stop_words, kolata_paragraph, non_kolata_paragraph, lemma_bank = load_serialized_lists()
    with open("baseline_results.txt", "w") as fout:
        for paragraph in test_paragraphs:
            author_score, non_author_score = get_match_score(paragraph, kolata_list, non_kolata_list, corpus_distribution, stop_words, lemma_bank)
            if author_score <= non_author_score:
                fout.write("0\n")
            else:
                fout.write("1\n")


def get_kolata_threshold():
    """
    Test each paragraph by the author again all training set to examine thresholds
    """
    test_paragraphs, kolata_list, non_kolata_list, corpus_distribution, stop_words, kolata_paragraph, non_kolata_paragraph, lemma_bank  = load_serialized_lists()
    counter = 0
    min_score = 10000
    max_score = 0
    total_score = 0

    min_non_score = 10000
    max_non_score = 0
    total_non_score = 0

    for i in range(len(kolata_paragraph)):
        temp_paragraph = kolata_paragraph[i]
        temp_kolata_list = list(kolata_list)
        for token in tokenize(temp_paragraph):
            temp_kolata_list.remove(token)
        temp_kolata_score, temp_non_score = get_match_score(temp_paragraph, temp_kolata_list, non_kolata_list, corpus_distribution, stop_words, lemma_bank)
        min_score = min(temp_kolata_score, min_score)
        max_score = max(temp_kolata_score, max_score)
        total_score += temp_kolata_score

        min_non_score = min(temp_non_score, min_non_score)
        max_non_score = max(temp_non_score, max_non_score)
        total_non_score += temp_non_score

        print min_score

    with open("threshold_kolata_paragraphs.txt", "w") as fout:
        fout.write("Min score for a kolata paragraph vs kolata training sets: \n")
        fout.write(str(min_score))
        fout.write("\nMax score for a kolata paragraph vs kolata training sets: \n")
        fout.write(str(max_score))
        fout.write("\nAvg score for a kolata paragraph vs kolata training sets: \n")
        fout.write(str(float(total_score)/len(kolata_paragraph)))

        fout.write("\nMin score for a kolata paragraph vs non kolata training sets: \n")
        fout.write(str(min_non_score))
        fout.write("\nMax score for a kolata paragraph vs non kolata training sets: \n")
        fout.write(str(max_non_score))
        fout.write("\nAvg score for a kolata paragraph vs non kolata training sets: \n")
        fout.write(str(float(total_non_score)/len(kolata_paragraph)))

    # Serialize
    list_out = [min_score, max_score, total_score, min_non_score, max_non_score, total_non_score]
    cPickle.dump(list_out, open('thresholds_kolata.p', 'wb'))

    return list_out


def load_serialized_lists():
    if not os.path.isfile("lemma_bank.p"):
        return setup()
    test_paragraphs = cPickle.load(open('test_paragraphs.p', "rb"))
    kolata_list = cPickle.load(open('kolata_list.p', "rb"))
    kolata_paragraph = cPickle.load(open('kolata_paragraph.p', "rb"))
    non_kolata_list = cPickle.load(open('non_kolata_list.p', "rb"))
    non_kolata_paragraph = cPickle.load(open('non_kolata_paragraph.p', "rb"))
    corpus_distribution = cPickle.load(open('corpus_distribution.p', "rb"))
    stop_words = cPickle.load(open('stop_words.p', "rb"))
    lemma_bank = cPickle.load(open('lemma_bank.p', "rb"))

    return test_paragraphs, kolata_list, non_kolata_list, corpus_distribution, stop_words, kolata_paragraph, non_kolata_paragraph, lemma_bank


def run_modified_baseline(delta):
    """
    We observed false positives in the Baseline algorithm. Thus we will add some delta to the comparison.
    """
    counter = 0
    test_paragraphs, kolata_list, non_kolata_list, corpus_distribution, stop_words, kolata_paragraph, non_kolata_paragraph, lemma_bank = load_serialized_lists()
    with open("modified_results.txt", "w") as fout:
        for paragraph in test_paragraphs:
            print "Paragraph #" + str(counter)
            counter += 1
            author_score, non_author_score = get_match_score(paragraph, kolata_list, non_kolata_list, corpus_distribution, stop_words, lemma_bank)
            if (author_score - non_author_score) > delta: #The difference must be at least larger than delta
                fout.write("1\n")
            else:
                fout.write("0\n")


def remove_false_negative_baseline(delta):
    """
    This will attempt to remove the false negative and later combined with the SVM results.
    """
    counter = 0
    test_paragraphs, kolata_list, non_kolata_list, corpus_distribution, stop_words, kolata_paragraph, non_kolata_paragraph, lemma_bank = load_serialized_lists()
    with open("remove_false_neg_results.txt", "w") as fout:
        for paragraph in test_paragraphs:
            print "Paragraph #" + str(counter)
            counter += 1
            author_score, non_author_score = get_match_score(paragraph, kolata_list, non_kolata_list, corpus_distribution, stop_words, lemma_bank)
            if (non_author_score - author_score) > delta: #The difference must be at least larger than delta for it to be non kolata
                fout.write("0\n")
            else:
                fout.write("1\n")


if __name__ == "__main__":
    # author_list, non_list = setup()
    # kolata_list = cPickle.load(open('kolata_list.p', "rb"))
    # print stopwords.words('english')
    # print len(stopwords.words('english'))
    # print get_stopwords()
    # print len(get_stopwords())
    # print common_names("census-dist-2500-last.txt")
    # print os.path.isfile("kolata_list.p")
    # non_kolata_list = cPickle.load(open('non_kolata_list.p', "rb"))
    # print len(non_kolata_list)
    # corpus = cPickle.load(open('corpus.p', "rb"))
    # print (len(kolata_list) + len(non_kolata_list)) == len(corpus)
    # test = []
    # with open("project_articles_test", "r") as testin:
    #     for line in testin:
    #         test.append(line.rstrip())
    # print test
    # run_baseline()
    # get_kolata_threshold()
    # tester("ph.d.")
    
    # run_baseline()
    # get_kolata_threshold()
    # run_modified_baseline(4)
    remove_false_negative_baseline(3)

    print "done"
