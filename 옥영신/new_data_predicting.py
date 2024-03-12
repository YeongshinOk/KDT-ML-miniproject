import pandas as pd
from joblib import load

# 저장된 파이프라인을 로드합니다.
loaded_pipeline = load('final_pipeline.joblib')

# 사용자로부터 입력 받기 (이 부분은 웹 애플리케이션 등에서 구현될 수 있습니다)
def get_user_input():
    # 사용자 입력을 받아 DataFrame 형태로 반환하는 로직 구현
    data = {
        'ArrivingMonth': [6],  # 예시 입력
        'SpecialRequest': [2],
        'Meal': ['BB'],
        'LeadTime': [60],
        'WeekendStay': [2],
        'WeekStay': [3],
        'total_members': [4],
        'ParkSpace': [1]
    }
    return pd.DataFrame(data)

# 새로운 입력 데이터에 대해 예측을 수행합니다.
user_input_df = get_user_input()
prediction = loaded_pipeline.predict(user_input_df)

print(f"예측된 평균 일일 요금(ADR): {prediction[0]}")
