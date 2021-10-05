import numpy as np
import sklearn as skl
import time
import fasttext
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn import svm, tree

def main():
    
    start_time = time.time()
    model = fasttext.train_supervised('train_preprocessed.txt')
    print_results(*model.test('test_preprocessed.txt'))
    print('FastText: ', time.time() - start_time)


    start_time = time.time()
    f1 = open("train_nl.txt", encoding='utf8')
    f2 = open("train_labels.txt")
    f3 = open("test_nl.txt", encoding='utf8')
    f4 = open("test_labels.txt")
    train_data = f1.readlines()[:36000]
    train_labels = f2.readlines()[:36000]
    test_data = f3.readlines()[:4000]
    test_labels = f4.readlines()[:4000]

    
    for i in range(len(train_labels)):
        train_labels[i] = train_labels[i].rstrip('\n')

    for i in range(len(test_labels)):
        test_labels[i] = test_labels[i].rstrip('\n')
        
    vectorizer = TfidfVectorizer()

    print('everything loaded')
    print('everything:: ', time.time() - start_time)
    start_time = time.time()

    vectorizer.fit(train_data)
    tdvector_train = vectorizer.transform(train_data)
    tdvector_test = vectorizer.transform(test_data)
    print('tdfif loaded')
    print('tfidf: ', time.time() - start_time)
    start_time = time.time()

    vectorizer1 = CountVectorizer()
    vectorizer1.fit(train_data)
    cvector_train = vectorizer1.transform(train_data)
    cvector_test = vectorizer1.transform(test_data)
    print('count loaded')

    print('count: ', time.time() - start_time)

    testSVM(tdvector_train,cvector_train,train_labels,tdvector_test,cvector_test,test_labels)
    stochasticGD(tdvector_train,cvector_train,train_labels,tdvector_test,cvector_test,test_labels)
    trees(tdvector_train,cvector_train,train_labels,tdvector_test,cvector_test,test_labels)





def print_results(N, p, r):
    print("N\t" + str(N))
    print("Accuracy@ {}\t{:.3f}".format(1, p))

def testSVM(tdvector_train,cvector_train,train_labels,tdvector_test,cvector_test,test_labels):
    start_time = time.time()
    vclf = svm.LinearSVC()
    vclf.fit(tdvector_train,train_labels)

    pr = vclf.predict(tdvector_test)
    a = accuracy_score(test_labels, pr)
    print("LinearSVM TFIDF: ", a)
    print('time elaplsed: ', time.time() - start_time)

    start_time = time.time()
    cclf = svm.LinearSVC()
    cclf.fit(cvector_train,train_labels)

    pr = cclf.predict(cvector_test)
    b = accuracy_score(test_labels, pr)
    print("LinearSVM Count: ", b)
    print('time elaplsed: ', time.time() - start_time, '\n')

def stochasticGD(tdvector_train,cvector_train,train_labels,tdvector_test,cvector_test,test_labels):
    start_time = time.time()
    vclf = SGDClassifier()
    vclf.fit(tdvector_train,train_labels)

    pr = vclf.predict(tdvector_test)
    a = accuracy_score(test_labels, pr)
    print("SGDClassifier TFIDF: ", a)
    print('time elaplsed: ', time.time() - start_time)

    start_time = time.time()
    cclf = SGDClassifier()
    cclf.fit(cvector_train,train_labels)

    pr = cclf.predict(cvector_test)
    b = accuracy_score(test_labels, pr)
    print("SGDClassifier Count: ", b)
    print('time elaplsed: ', time.time() - start_time, '\n')

def trees(tdvector_train,cvector_train,train_labels,tdvector_test,cvector_test,test_labels):
    start_time = time.time()
    vclf = tree.DecisionTreeClassifier()
    vclf.fit(tdvector_train,train_labels)

    pr = vclf.predict(tdvector_test)
    a = accuracy_score(test_labels, pr)
    print("Decision Trees TFIDF: ", a)
    print('time elaplsed: ', time.time() - start_time)

    start_time = time.time()
    cclf = tree.DecisionTreeClassifier()
    cclf.fit(cvector_train,train_labels)

    pr = cclf.predict(cvector_test)
    b = accuracy_score(test_labels, pr)
    print("Decision Trees Count: ", b)
    print('time elaplsed: ', time.time() - start_time, '\n')

main()