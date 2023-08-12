import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# CSV 파일에서 데이터 불러오기
data = pd.read_csv('real_merge.csv')

# NaN 값을 빈 문자열로 대체하기
data['Tokens'].fillna('', inplace=True)

# 추출할 맥주 종류
target_beer_type = 'Kloud Original Gravity'  # 추출하고자 하는 맥주 종류

# 맥주 종류별 데이터 추출
data_filtered = data[data['beer_name'] == target_beer_type].copy()

# 텍스트 데이터와 레이블 준비하기
X = data_filtered['Tokens']  # 텍스트 데이터
y = data_filtered['Aroma'].apply(lambda x: 1 if x >= 3 else 0)  # Aroma 점수를 긍정(1) 또는 부정(0)으로 변환하여 사용
print(X[0])
print(len(X[0]))
# CountVectorizer를 사용하여 텍스트 데이터를 벡터화하기
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
print(X[0])
# 로지스틱 회귀분석 모델 학습하기
model = LogisticRegression()
model.fit(X, y)

# 각 단어별로 영향력 파악하기
feature_names = vectorizer.get_feature_names_out()  # 단어 목록
coef = model.coef_[0]  # 회귀분석 모델의 계수

word_impact = {}
for word, impact in zip(feature_names, coef):
    word_impact[word] = impact

# 결과 출력
print("Word Impact on Aroma Score:")
for word, impact in sorted(word_impact.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {impact}")
