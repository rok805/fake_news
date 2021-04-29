# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:40:59 2020

@author: user
"""

# 모듈
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, csc_matrix
from collections import Counter
from konlpy.tag import Okt
import pandas as pd
import time
import re


# 데이터 로드
train = pd.read_csv(r"C:\Users\user\Documents\dacon\data\news_train.csv")
test = pd.read_csv(r"C:\Users\user\Documents\dacon\data\news_test.csv")
submission = pd.read_csv(r"C:\Users\user\Documents\dacon\data\sample_submission.csv")


# 전처리
cont = list(train.content)
cont2 = [re.sub('[^가-힣]',' ',i).split() for i in cont]  # 한글만 남기고 제거.
cont_test = list(test.content)
cont2_test = [re.sub('[^가-힣]',' ',i).split() for i in cont_test]  # 한글만 남기고 제거.


# 명사추출
okt = Okt()

corpus_d = {}
features = set()

rows = []  # sparse matrix 를 만들기 위함.
cols = []
rows_test = []
cols_test = []

# train feature 추출.
start = time.time()
for num, i in enumerate(cont2):
    n = okt.nouns(' '.join(i))
    corpus_d[num] = {}

    for j in n:
        if len(j) > 1:  # 한 글자 제거.
            features.add(j)
            corpus_d[num][j] = 1
            rows.append(num) # x
            cols.append(j) # y

end = time.time() - start
f'{round(end/60,2)} 분'


# test feature 추출.
corpus_d_test = {}
start = time.time()
for num, i in enumerate(cont2_test):
    n = okt.nouns(' '.join(i))
    corpus_d_test[num] = {}

    for j in n:
        if len(j) > 1 and j in features:  # 한 글자 제거.
            corpus_d_test[num][j] = 1
            rows_test.append(num) # x
            cols_test.append(j) # y

end = time.time() - start
f'{round(end/60,2)} 분'


# train sparse matrix 생성
w_to_n={}
n_to_w={}
for num, i in enumerate(features):
    w_to_n[i] = num
    n_to_w[num] = i

data = [1 for i in rows]
cols2 = [w_to_n[i] for i in cols]

csr = csr_matrix((data, (rows, cols2)))
csc = csc_matrix((data, (rows, cols2)))


# test sparse matrix 생성
data_test = [1 for i in rows_test]
cols2_test = [w_to_n[i] for i in cols_test]

csr = csr_matrix((data_test, (rows_test, cols2_test)))


# prediction model
X = csc
y = list(train['info'])
start = time.time()
rf = RandomForestClassifier(n_estimators=1000, random_state=1)
rf.fit(X, y)
end = time.time() - start
f'{round(end/60,2)} 분'


# accuracy
pred = rf.predict(csr)
submission['info'] = pred


# save result
submission.to_csv('ab1000.csv')


