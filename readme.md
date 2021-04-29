### 뉴스 데이터를 활용하여 가짜 뉴스를 분류하는 분류 모델 생성.

 뉴스 데이터의 내용에서 한글 명사를 추출하여 특징을 추출해내고, 랜덤 포레스트 알고리즘을 사용하여 이진 분류를 하는 모델을 생성합니다.
 ~~데이터는 아래의 링크에 공유하였습니다.(현재 저작권으로 인한 공유 중지.)~~
 
학습 데이터는 news_train.csv 이며, 테스트 데이터는 news_test.csv 입니다.
그리고 마지막 실제 예측해야하는 검증 데이터는 sample_submission.csv 파일입니다.
실제로 변수는 데이터 셋의 'content' 와 'info' 변수만 사용합니다. content 는 feature 변수이고, info 가 binary class를 가진 이진 변수입니다.

source 폴더에 담긴 python 파일의 설명은 다음과 같습니다.


preprocessing.py
  * __init__: train, test 데이터를 받는 클래스가 담겨 있습니다. 초기화에서 train, test set이 매개변수에 입력으로 필요합니다.
  * prep: train, test 데이터에서 한글만을 남기고 모조리 제거해주는 전처리 함수입니다. 매개변수가 필요없습니다.
  * feature_extraction: Okt 모듈을 활용하여 명사를 추출하며, 이어서 scipy의 sparse matrix를 생성하여 feature extraction matrix를 생성합니다. 매개변수가 필요없습니다.
  * model(n_est): n_est는 random forest model의 hyper parameter인 bagging 시 tree 개수를 지정하는 매개변수 입니다. 기본 random_state=1로 설정되어 있습니다.
  * predict(submission): model 함수에서 생성된 모델로 unseen 데이터 셋을 매개변수로 하여 예측값을 생성해내는 함수입니다. 리턴값이 존재합니다. 리턴값은 예측값이 달려서 나오는 데이터프레임과 예측값 벡터입니다.
                
main.py: 데이터를 로드하고, rf 모듈을 실행하는 메인 함수입니다.

