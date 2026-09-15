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