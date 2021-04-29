

from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, csc_matrix
from collections import Counter
from konlpy.tag import Okt
import pandas as pd
import time
import re
import os


class text_preprocessing():
    
    def __init__(self, train, test):
        
        self.train = train
        self.test = test
        

    def prep(self):        
        # basic preprocessing only Korean words.
        self.cont = list(self.train.content)
        self.cont2 = [re.sub('[^가-힣]',' ',i).split() for i in self.cont]
        self.cont_test = list(self.test.content)
        self.cont2_test = [re.sub('[^가-힣]',' ',i).split() for i in self.cont_test]

                
    def feature_extraction(self):    
        # 명사추출
        self.okt = Okt()
        
        self.corpus_d = {}
        self.features = set()
        
        self.rows = []  # sparse matrix 를 만들기 위함.
        self.cols = []
        self.rows_test = []
        self.cols_test = []
        
        # train feature 추출.
        start = time.time()
        for num, i in enumerate(self.cont2):
            self.n = self.okt.nouns(' '.join(i))
            self.corpus_d[num] = {}
        
            for j in self.n:
                if len(j) > 1:  # 한 글자 제거.
                    self.features.add(j)
                    self.corpus_d[num][j] = 1
                    self.rows.append(num) # x
                    self.cols.append(j) # y
        
        end = time.time() - start
        f'{round(end/60,2)} 분'
        
        
        # test feature 추출.
        self.corpus_d_test = {}
        start = time.time()
        for num, i in enumerate(self.cont2_test):
            self.n = self.okt.nouns(' '.join(i))
            self.corpus_d_test[num] = {}
        
            for j in self.n:
                if len(j) > 1 and j in self.features:  # 한 글자 제거.
                    self.corpus_d_test[num][j] = 1
                    self.rows_test.append(num) # x
                    self.cols_test.append(j) # y
        
        end = time.time() - start
        f'{round(end/60,2)} 분'
        
        
        # train sparse matrix 생성
        self.w_to_n={}
        self.n_to_w={}
        for num, i in enumerate(self.features):
            self.w_to_n[i] = num
            self.n_to_w[num] = i
        
        self.data = [1 for i in self.rows]
        self.cols2 = [self.w_to_n[i] for i in self.cols]
        
        self.csr = csr_matrix((self.data, (self.rows, self.cols2)))
        self.csc = csc_matrix((self.data, (self.rows, self.cols2)))
        
        
        # test sparse matrix 생성
        self.data_test = [1 for i in self.rows_test]
        self.cols2_test = [self.w_to_n[i] for i in self.cols_test]
        
        self.csr = csr_matrix((self.data_test, (self.rows_test, self.cols2_test)))


    def model(self, n_est): 
        self.n_est = n_est
        
        # prediction model
        self.X = self.csc
        self.y = list(self.train['info'])
        start = time.time()
        self.rf = RandomForestClassifier(n_estimators=self.n_est , random_state=1)
        self.rf.fit(self.X, self.y)
        end = time.time() - start
        f'{round(end/60,2)} 분'


    def predict(self, submission):
        self.submission = submission
        
        # accuracy
        self.pred = self.rf.predict(self.csr)
        self.submission['info'] = self.pred
        return self.submission, self.pred
        
        
        
        
