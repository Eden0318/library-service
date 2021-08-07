from flask import Flask, render_template, request
import json
import csv
import sys

#제가 추가한 library -> 안깔려서 에러뜰테니 주석처리하시는게 좋을듯
import pandas as pd
import numpy as np
from urllib.request import urlopen, Request
import time
import requests
import re
from glove import Corpus, Glove
from konlpy.tag import Okt
from gensim import models
from sklearn.model_selection import KFold, train_test_split
from keras_tuner import BayesianOptimization, HyperModel
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import keras_tuner
from skopt import BayesSearchCV
import xgboost as xgb
import joblib
import pickle


application = Flask(__name__)
data=["223","1231"]

@application.route("/index")
def index():
    return render_template("index.html")

#예측
#form 중 도서관코드를 l_c, isbn을 i_c에 저장해주세요.
#둘다 문자열로 저장해서 안꼬이게 부탁드립니다.
@application.route("/search")
def search():
    def preprocess(i_p, c_p):
        if len(i_p) != 13:
            return '잘못된 ISBN입니다. 13자리 ISBN인지 확인해주세요.'
    
        if c_p in lib_final.code.values:
            lib = lib_final.loc[lib_final.code == c_p,:].drop(columns = ['code'])
        else:
            return '지원하지 않는 도서관입니다.'

        url = f'http://seoji.nl.go.kr/landingPage/SearchApi.do?cert_key={oa_key}&isbn={i_p}&result_style=json&page_size=1&page_no=1'
        res = requests.get(url)
        try:
            info = res.json()['docs'][0]
        except:
            return '국립중앙도서관 OpenAPI에 등재되지 않은 도서입니다.'
    
        title = info['TITLE']
        words = [w for w in okt.nouns(title) if not w in stopword_title]
        if len(words) == 0:
            ebd = (np.zeros(300))
        else:
            ebd = (sum([ko_model.wv[ws] for ws in words])/len(words))
    
        if info['REAL_PUBLISH_DATE'] == '':
            publish_date = info['PUBLISH_PREDATE']
        else:
            publish_date = info['REAL_PUBLISH_DATE']
        days = (pd.to_datetime('20210630', format = '%Y%m%d') - pd.to_datetime(publish_date, format='%Y%m%d')).days
        if days < 30:
            days = 30
        if days > 365:
            days = 365
    
        if info['REAL_PRICE'] == '':
            price = info['PRE_PRICE']
        else:
            price = info['REAL_PRICE']
        pr = re.findall(r'\d+', price.replace(',',''))
        if pr == []:
            price = int(0)
        else:
            price = int(max(pr))
        
        try:
            ea_code = info['EA_ADD_CODE']
            subject_1 = ea_code[2]
            subject_2 = ea_code[3]
            reader = ea_code[0]
            pub_type = ea_code[1]
        except:
            return '국립중앙도서관 OpenAPI에 누락된 정보가 있기에 예측을 할 수 없습니다.'
    
        pages = info['PAGE']
        p_r = re.findall(r'\d+', pages)
        if p_r == []:
            pages = int(0)
        else:
            pages = int(max(p_r))
    
        author = info['AUTHOR']
        
        b_i = pd.concat([pd.DataFrame({'price' : price, 'subject_1' : subject_1, 'subject_2' : subject_2, 'reader' : reader, 'pub_type' : pub_type, 'pages' : pages, 'pub_date' : days}, index=[0]),pd.DataFrame(ebd).T, lib], axis = 1)
        b_i.columns = b_i.columns.map(str)
    
        a_l = [x for x in re.findall(u'[\u3131-\uD79D]+', str(author)) if x not in a_stopwords]
        for c in dummies_c:
            if c in a_l:
                b_i[c] = int(1)
            else:
                b_i[c] = int(0)
    
        data = b_i[column_order.to_list()]
        data = scaler.transform(data)
    
        return data

    X = prepreprocess(i_c, l_c)

    def lc_predict(X):
        if len(X[0]) == 856:
            return f'해당 책의 대출 예상 횟수는 {round((10 ** xgb.predict(X))[0])}번 입니다.'
        else:
            return X

    y_pred = lc_predict(X)

    def add_info(lcc):
        return f'{lib_perc_d.loc[lib_perc_d.code == lcc, "name"].item()}에서 6개월간 대출된 책 중 {round(lib_perc_d.loc[lib_perc_d.code == lcc, "percentage"].item(),1)}%가 1회만 대출됐습니다.'

    additional_info = add_info(l_c)

    ''' 
    y_pred가 예측값 혹은 에러값. ( 해당 책의 대출 예상 횟수는 3번입니다. or 지원하지않는 ISBN입니다. 이런식.)
    additional_info가 추가정보.( 세종도서관에서 6개월간 대출된 책 중 39%가 1회만 대출됐습니다. )
    두개다 반환해주시면되용
    '''
   
    return render_template("search.html")

@application.route("/result", methods=["POST","GET"])
def result():
    if request.method == "POST":
        result = request.form
        return render_template("result.html",result=result)


if __name__ == "__main__":
    lib_final = pd.read_csv('lib_final.csv', dtype = {'code' : 'object'})
    lib_final = lib_final.drop(columns=['name', 'dtl_region'])
    lib_perc_d = pd.read_csv('lib_perc_d.csv', dtype = {'code' : 'object'})
    oa_key = '578ca4ba507631e4a9b621f4029400eac427aaf6071b45611e599387b637b6dc'
    okt = Okt()
    with open('stopwords_title.txt', encoding = 'UTF-8') as f:
        content = f.readlines()
    stopword_title = [x.replace('\n','') for x in content]
    #ko_model = models.fasttext.load_facebook_model('cc.ko.300.bin.gz') : 얘가 단어를 임베딩 벡터로바꿔주는애인데 용량이 5기가짜리라서 깃헙에못올림. 애 어쩔지도 생각해야할듯.
    scaler = joblib.load('scaler_gt10.gz')
    a_stopwords = ['지음','지은이','그림','글','저자','집필자','옮긴이','옮김','원작']
    xgb = joblib.load('xgb_gt10.dat')
    with open("dummies_c.txt", "rb") as fp:  
        dummies_c = pickle.load(fp)
    with open("column_order.txt", "rb") as fp:   
        column_order = pickle.load(fp)
    application.run(host='0.0.0.0')