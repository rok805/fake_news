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
import os

import rf

# 데이터 로드
path = os.getcwd()
train = pd.read_csv(os.path.join(path,'data\\news_train.csv'))
test = pd.read_csv(os.path.join(path,'data\\news_test.csv'))
submission = pd.read_csv(os.path.join(path,"data\\sample_submission.csv"))

# 데이터 전처리
news = rf.text_preprocessing(train, test)
news.prep()

# 데이터 특징 추출
news.feature_extraction()

# 랜덤포레스트 학습
news.model(1000)

# 예측
result, pred = news.predict(submission)


# save result
submission.to_csv('ab1000.csv')