from nltk import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import nltk
from sklearn.ensemble import AdaBoostClassifier
from sklearn import neighbors
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


class Baseline(object):
    def __init__(self, language):
        self.language = language
        self.TF = {}
        self.TF_root = {}
        self.words = []
        self.words_root = []
        self.sentences = {}
        self.sentences_pos = {}
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        # self.model_SVM = svm.SVC()  # 0.7
        # self.model_LR = LogisticRegression()  # 0.1
        # self.model_CLR = AdaBoostClassifier(n_estimators=100)
        self.model_RF = RandomForestClassifier()
        # self.model_KNN = neighbors.KNeighborsClassifier()

    def extract_features(self, word, id):

        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        words = word.split(' ')
        sum_depth = 0
        if id in self.sentences_pos:
            for w in words:
                if w in self.sentences_pos[id]:
                    pos_temp = self.sentences_pos[id][w]
                else:
                    pos_temp = wn.NOUN
                sum_depth += self.get_depth(w, pos_temp)

        word_tf = self.TF.get(word, 0)
        word_syn_len = len(wn.synsets(word))
        if self.language == 'english':
            word_root_tf = self.TF_root.get(self.target_stemming(word), 0)
        else:
            word_root_tf = self.TF_root.get(self.target_stemming_spanish(word), 0)
        return [len_chars, len_tokens, word_tf, word_root_tf, word_syn_len]
        # return [len_chars, len_tokens, word_tf, word_root_tf, word_syn_len, sum_depth]

    # ============================tf feature===================================#
    def tf_feature(self):
        self.TF = dict(Counter(self.words))
        self.TF_root = dict(Counter(self.words_root))

    # ========================================================================#

    def train(self, trainset):
        X = []
        y = []
        random.seed(2)
        R = random.random()
        random.shuffle(trainset, lambda: R)
        if self.language == 'english':
            for sent in trainset:
                self.words.append(sent['target_word'])
                self.sentences[sent['hit_id']] = sent['sentence']
                self.words_root.append(self.target_stemming(sent['target_word']))
        else:
            for sent in trainset:
                self.words.append(sent['target_word'])
                self.sentences[sent['hit_id']] = sent['sentence']
                self.words_root.append(self.target_stemming_spanish(sent['target_word']))

        self.tf_feature()
        self.get_words_pos()

        for sent in trainset:
            # print(sent['target_word'])
            X.append(self.extract_features(sent['target_word'], sent['hit_id']))
            y.append(sent['gold_label'])

        # print(X)
        # print(y)
        # self.model_LR.fit(X, y)
        # self.model_SVM.fit(X, y)
        # self.model_CLR.fit(X, y)
        self.model_RF.fit(X, y)
        # self.model_KNN.fit(X, y)
        # title = "Learning Curves (LogisticRegression)"
        # self.plot_learning_curve(self.model_LR, title, X, y)
        # plt.show()

    def doPredict(self, X):
        # result = []
        # predict_LR = self.model_LR.predict(X)
        # predict_SVM = self.model_SVM.predict(X)
        # predict_CLR = self.model_CLR.predict(X)
        # predict_RF = self.model_RF.predict(X)
        # predict_KNN = self.model_KNN.predict(X)
        # if self.language == 'english':
        #     w0 = 0.1  # LR
        #     w1 = 0.2  # SVM
        #     w2 = 0.15  # CLR
        #     w3 = 0.3  # RF
        #     w4 = 0.25  # KNN
        # else:
        #     w0 = 0.25  # LR
        #     w1 = 0.2  # SVM
        #     w2 = 0.2  # CLR
        #     w3 = 0.25  # RF
        #     w4 = 0.1  # KNN
        #
        # for i in range(len(predict_LR)):
        #     temp = w0 * int(predict_LR[i]) + w1 * int(predict_SVM[i]) + w2 * int(predict_CLR[i]) + w3 * int(
        #         predict_RF[i]) + w4 * int(predict_KNN[i])
        #     if temp > 0.5:
        #         result.append('1')
        #     else:
        #         result.append('0')
        return self.model_RF.predict(X)
        # return np.array(result)


    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'], sent['hit_id']))
        return self.doPredict(X)

    def target_stemming(self, words):
        result = ""
        wordset = words.split(" ")
        stemmer = PorterStemmer()
        for word in wordset:
            result += stemmer.stem(word) + "_"
        return result

    def target_stemming_spanish(self, words):
        result = ""
        wordset = words.split(" ")
        stemmer = SnowballStemmer('spanish')
        for word in wordset:
            result += stemmer.stem(word) + "_"
        return result

    def get_wordnet_pos(self, treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            # As default pos in lemmatization is Noun
            return wn.NOUN

    def get_words_pos(self):
        for key, val in self.sentences.items():
            self.sentences_pos[key] = nltk.pos_tag(nltk.word_tokenize(val))

    def get_depth(self, word, pos):
        word_synset = wn.synsets(word)
        for wordtemp in word_synset:
            wordname = wordtemp.name()
            jword = wordname.split(".")
            if (jword[0] == word and jword[1] == pos):
                return int(jword[2])
        return 0

    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt
