#!/usr/bin/env python

"""Training code using Word2Vec.

This code is based on Angela Chapman's tutorial on Kaggle:

https://www.kaggle.com/c/word2vec-nlp-tutorial

"""

# ****** Read the two training sets and the test set

import logging

import pandas as pd
from nltk.corpus import stopwords
import nltk.data
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from utils import review_to_wordlist, review_to_sentences

# Beautiful Soup throws a lot of benign warnings which we will ignore
import warnings
warnings.filterwarnings("ignore")


# ****** Define functions to create average word vectors


def make_feature_vec(articles_words, model, num_features):
    """Average all of the word vectors in a given paragraph.

    articles_words: list of words (strings)
    model: instance of Word2Vec()
    num_features: word vector dimensionality
    """

    # Pre-initialize an empty numpy array (for speed)
    feature_vec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    # index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed.
    index2word_set = set(model.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in articles_words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model[word])

    # Divide the result by the number of words to get the average
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(articles, model, num_features):
    """Given a set of articles (each one a list of words), calculate
    the average feature vector for each one and return a 2D numpy array.

    articles: list of list of words (strings)
    model: instance of Word2Vec()
    num_features: word vector dimensionality
    """

    # Initialize a counter
    counter = 0

    # Pre-allocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(articles), num_features), dtype="float32")

    # Loop through the articles
    for review in articles:

       # Print a status message every 1000th review
       if counter % 1000 == 0:
           print "Review %d of %d" % (counter, len(articles))

       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = make_feature_vec(review, model, num_features)

       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs


def get_clean_reviews(articles):
    """Return a list of list of words."""

    clean_reviews = []
    for review_text in articles["review"]:
        clean_reviews.append(review_to_wordlist(review_text,
                                                remove_stopwords=True))
    return clean_reviews



if __name__ == '__main__':

    from os.path import join, dirname

    # Set logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    # Read data from files
    train_file = join(dirname(__file__), 'data', 'labeledTrainData.tsv')
    train = pd.read_csv(train_file, header=0, delimiter="\t", quoting=3)

    test_file = join(dirname(__file__), 'data', 'testData.tsv')
    test = pd.read_csv(test_file, header=0, delimiter="\t", quoting=3)

    unlbd_file = join(dirname(__file__), 'data', "unlabeledTrainData.tsv")
    unlabeled_train = pd.read_csv(unlbd_file, header=0, delimiter="\t",
                                  quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
     "and %d unlabeled reviews\n" % (train["review"].size,
     test["review"].size, unlabeled_train["review"].size )

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the labeled and unlabeled training sets into clean sentences

    sentences = []  # Initialize an empty list of sentences

    print "Parsing sentences from training set"
    for review_text in train["review"]:
        sentences += review_to_sentences(review_text, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review_text in unlabeled_train["review"]:
        sentences += review_to_sentences(review_text, tokenizer)

    # ****** Set parameters and train the word2vec model

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers,
                size=num_features, min_count=min_word_count,
                window=context, sample=downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context.w2v.model"
    model.save(model_name)

    # ****** Create average vectors for the training and test sets

    print "Creating average feature vecs for training reviews"

    trainDataVecs = get_avg_feature_vecs(get_clean_reviews(train), model, num_features)

    print "Creating average feature vecs for test reviews"

    testDataVecs = get_avg_feature_vecs(get_clean_reviews(test), model, num_features)


    # ****** Fit a random forest to the training set, then make predictions

    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # TODO: save the forest model
