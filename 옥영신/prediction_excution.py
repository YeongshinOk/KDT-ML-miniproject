import pandas as pd
from joblib import load

# # 저장된 모델과 OneHotEncoder 로드
# model = load('best_model.pkl')
encoder = load('preprocessor.pkl')  # OneHotEncoder 로드 가정

# 모듈 로딩
from joblib import load

# 전역 변수
model_file = './best_model.pkl'

# 모델 로딩
model=load(model_file)

# 사용자로부터 입력 받기
def get_input():
    data = {}
    data['ArrivingMonth'] = int(input("투숙 월 (숫자로 입력): "))
    data['SpecialRequest'] = int(input("특별 요청 횟수: "))
    data['Meal'] = input("식사 타입 (BB, FB, HB, SC 중 선택): ")
    data['LeadTime'] = int(input("예약에서 투숙까지의 기간 (일 수): "))
    data['WeekendStay'] = int(input("주말 숙박 일 수: "))
    data['WeekStay'] = int(input("평일 숙박 일 수: "))
    data['total_members'] = int(input("총 인원수: "))
    data['ParkSpace'] = int(input("필요한 주차 공간 수: "))

    return pd.DataFrame([data])


# 입력 데이터 전처리
def preprocess_input(input_df):
    # 범주형 데이터 처리
    categorical_data = input_df[['Meal']]
    categorical_transformed = encoder.transform(categorical_data)

    # 원-핫 인코딩된 데이터를 DataFrame으로 변환
    categorical_df = pd.DataFrame(categorical_transformed.toarray(),
                                  columns=encoder.get_feature_names_out(['Meal']))

    # 숫자형 데이터와 결합
    numeric_data = input_df.drop('Meal', axis=1)
    processed_df = pd.concat([numeric_data.reset_index(drop=True), categorical_df], axis=1)

    return processed_df


# 메인 함수: 예측 실행 및 결과 출력
def predict_adr():
    user_input = get_input()
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)
    print(f"예측된 평균 일일 요금(ADR): {prediction[0]}")


# 예측 실행
predict_adr()
import joblib

# 저장된 파이프라인을 로드합니다.
loaded_pipeline = joblib.load('full_pipeline_with_predictor.pkl')

# 새로운 입력 데이터에 대해 예측을 수행합니다.
# 예를 들어, user_input_df는 사용자로부터 받은 입력 데이터를 포함하는 pandas DataFrame입니다.
prediction = loaded_pipeline.predict(user_input_df)

print(f"예측된 ADR: {prediction}")