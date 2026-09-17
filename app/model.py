'''
app/model.py
------------------
학습된 대출 심사 모델의 로딩, 전처리, 추론을 캡슐화한 모듈.

"스키마 - 모델 - API" 
3단계 중 2단계 작업
'''
import os
import logging
import joblib
import pandas as pd
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

# 번역 사전 의미
#   REST API는 국제표준을 따라 영어 필드명을 사용한다.
#   학습 데이터와 인코드는 데이터 분석가/데이터 기획자가 다루기 쉬운 한글 컬럼명으로
#   만들어졌다. 두 상황을 잇는 번역 사전 역할 --> FIELD_TO_COLUMN
FIELD_TO_COLUMN = {
    "age": "나이",
    "gender": "성별",
    "annual_income": "연소득",
    "employment_years": "근속연수",
    "housing_type": "주거형태",
    "credit_score": "신용점수",
    "existing_loan_count": "기존대출건수",
    "annual_card_usage": "연간카드사용액",
    "debt_ratio": "부채비율",
    "loan_amount": "대출신청액",
    "loan_purpose": "대출목적",
    "repayment_method": "상환방식",
    "loan_period": "대출기간",
}

class LoanModel:
    """
    대출 모델 아키텍처와 추론(예측) 규칙을 한 객체로 관리한다.
        파이프라인 뿐만 아니라 학습 시 사용한 범주형 인코더와 컬럼 순서도 함께 불러온다.
        운영 추론이 학습 전처리와 정확히 같아야 하는 MLOps의 핵심 원칙을 보여주는 구조.

    학습 때 쓴 전처리(인코딩, 스케일링)와 서빙 때 쓰는 전처리가 어긋나면 
    Training-Serving Skew가 발생해 학습 성능과 실서비스 성능이 크게 벌어진다.
    (이 클래스가 pipeline, label_encoders, feature_names까지 로드 하는 이유,
    학습 시 저장했던 전처리 도구를 그대로 재사용해야 두 단계의 결과가 일치한다.)    
    """
    def __init__(self):
        # 생성 직후에는 아직 적재되지 않은 상태.
        # main.py에서 /health (헬스체크) 를 할 때 self.pipeline이 None이면 초기상태를 표현 
        self.pipeline = None
        self.label_encoders: dict[str, Any] = {}
        self.feature_names: list[str] = []

        self.threshold: float = 0.5   # 임계값(확률을 가지고 승인/거절로 변환하는 기준값)
        # 0.5는 50%보다 높으면 승인 이라는 것!
        # 금융권처럼 부실대출(FP)의 비용이 큰 도메인에서는 이 값을 0.6~0.7로 올려
        #   더 보수적으로(엄격하게) 심사하기도 한다.

        self.model_version: str = '1.0.0'  # 추론 결과/로그에 모델 버전을 넣을 때 사용하는 메타데이터
        # 예) 재학습된 새 모델을 배포했을 때 예측 로그에 버전을 함께 남기면
        #       어느 버전의 모델이 이 예측을 했는지 추적할 수 있어
        #       드리프트 분석이나 문제 발생 시 원인 추적에 도움이 된다. 

    def load(self, model_dir: str = 'models') -> None:
        """
        모델, 레이블 인코더, 피처 특성(컬럼) 순서 파일을 디스크에서 불러온다.

        현재 작업폴더가 어디든 동일하게 동작하도록 이 파일의 위치를 기준으로 계산
        Dockerfile의 WORKDIR이 달라지든 이 코드는 항상 이 파일(model.py) 기준으로 models폴더를
        찾으므로 실행 위치에 영향을 받지 않는다.
        """
        script_dir = Path(__file__).parent # app 폴더 (현재 파일이 있는 폴더)
        model_path = script_dir.parent / model_dir  # mlops-loan/models

        pipeline_path = model_path / 'loan_pipeline.pkl'
        encoder_path = model_path / 'label_encoders.pkl'
        feature_names_path = model_path / 'feature_names.pkl'

        if not pipeline_path.exists():
            raise FileNotFoundError(f'모델 파일을 찾을 수 없습니다: {pipeline_path}')
        if not encoder_path.exists():
            raise FileNotFoundError(f'레이블 인코더 파일을 찾을 수 없습니다: {encoder_path}')
        if not feature_names_path.exists():
            raise FileNotFoundError(f'특성 파일을 찾을 수 없습니다: {feature_names_path}')

        self.pipeline = joblib.load(pipeline_path)
        self.label_encoders = joblib.load(encoder_path)
        self.feature_names = joblib.load(feature_names_path)

        logging.info('모델 로드 완료!')

    # @staticmethod (정적 메서드)
    #   클래스 내부에 정의하지만 인스턴스(self)나 클래스(cls) 정보를 받지 않는 메서드를 만들 때 사용
    #   이 클래스와 관련은 있지만, 인스턴스 상태를 쓸 필요가 없는 함수를 클래스 안에 묶어두는 용도
    @staticmethod
    def _map_to_korean(data: dict[str, Any]) -> dict[str, Any]:
        """
        API의 영문 키를 학습 데이터에서 사용한 한글 컬럼명으로 변환한다.
            매핑표에 없는 키는 원래 키를 유지하므로 확장 필드가 있어도 즉시 사라지지 않는다.
            이후 feature_names 선택 단계에서 실제 입력만 남는다.
        """
        # result = {}
        # for k, v in data.items():  # ("age": "나이")
        #     new_key = FIELD_TO_COLUMN.get(k, k)
        #     result[new_key] = v

        # return result # return {FIELD_TO_COLUMN.get(k, k): v for k, v in data.items()}
        result = {}
        for key, value in data.items():
            if key in FIELD_TO_COLUMN:
                korean_key = FIELD_TO_COLUMN[key]  # ex) 나이
            else:
                korean_key = key  # ex) age
            result[korean_key] = value



        return result

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        고객 한 명의 입력을 전처리하고 승인 확률과 위험 등급을 계산한다.
            파이프라인 전체 구현
        """
        if self.pipeline is None:
            raise RuntimeError('모델이 로드되지 않았습니다. load()함수를 먼저 호출하세요!')

        mapped = self._map_to_korean(data)  # 컬럼을 한글로 재정렬 작업 (위에서 정의한 함수 호출)

        # 학습을 진행할 데이터 프레임 생성
        df = pd.DataFrame([mapped])[self.feature_names]

        # 학습 때 저장한 LabelEncoder를 동일 컬럼에 적용
        #   운영 데이터에 학습 시 없던 범주가 들어오면 transform에서 ValueError가 발생할 수 있다.
        for col, encoder in self.label_encoders.items():
            df[col] = encoder.transform(df[col])

        # predict_proba 첫 행([0])에서 양성/승인 클래스([1]) 확률을 꺼낸다.
        # Numpy 스칼라를 float로 변환해 JSON 직렬화가 가능하게 만든다.
        # model.predict() --> 결과가 0/1
        # model.predict_proba() --> 확률을 줘서 임계값 조정과 A/B/C/D 등급 산정이 가능해진다.
        # [0, 1]의 뒤 인덱스 1의 의미
        #   클래스 1(승인)의 확률 열을 의미한다.
        #   0번 열은 0 -> 거절 확률을 의미한다.
        probability = float(self.pipeline.predict_proba(df)[0, 1])

        # 확률을 정책 임계값과 비교해 최종 승인 여부를 결정한다.
        approved = probability >= self.threshold
        risk_grade = self._get_risk_grade(probability)

        return {
            'approved': approved,
            'probability': probability,
            'risk_grade': risk_grade
        }
        

    @staticmethod
    def _get_risk_grade(probability: float) -> str:
        """
        승인 확률 구간을 사람이 해석하기 쉬운 A~D 등급으로 변환한다.
        """
        if probability >= 0.75:
            return 'A'
        elif probability >= 0.5:
            return 'B'
        elif probability >= 0.25:
            return 'C'
        else:
            return 'D'

    # ---------------------------------------------------------------------------
    # 배치 예측용 메서드 (신규 추가)
    # ---------------------------------------------------------------------------
    def predict_batch(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        여러 명의 입력을 한 번에 전처리하고, 승인 확률과 위험 등급을 계산한다.
        
        """
        if self.pipeline is None:
            raise RuntimeError('모델이 로드되지 않았습니다. load() 함수를 먼저 호출하세요!')

        if not data_list:
            return []

        mapped_list = [self._map_to_korean(data) for data in data_list]  # n개의 한글 dict

        # 학습을 진행할 데이터프레임 생성
        df = pd.DataFrame(mapped_list)[self.feature_names]

        # 학습 때 저장한 LabelEncoder를 동일 컬럼에 적용
        for col, encoder in self.label_encoders.items():
            df[col] = encoder.transform(df[col])

        # predict_proba를 n행짜리 df에 "한 번만 호출" --> [:, 1]로 승인(1) 확률 열만 꺼낸다.
        probabilities = self.pipeline.predict_proba(df)[:, 1]

        results = []        
        for probability in probabilities:
            probability = float(probability) # 실수형으로 변환해서 저장
            approved = (probability >= self.threshold)
            risk_grade = self._get_risk_grade(probability)
            results.append({
                'approved': approved,
                'probability': probability,
                'risk_grade': risk_grade
            })
        return results
        