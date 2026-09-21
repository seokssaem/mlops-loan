'''
tests/test_api.py
--------------------
대출 심사 예측 API 테스트 코드

실행 방법 - 프로젝트 루트(mlops-loan/)에서 실행한다.
핵심 전략 - 학습된 모델 파일(.pkl) 이 없어도 예측결과를 우리가 원하는 값으로 고정하고 테스트
단계 
1. 준비 (헬퍼 함수 + fixture)
2. 단위 테스트 - LoanModel 클래스 (가장 작은 단위)
3. 통합 테스트 - GET /, /health
4. 통합 테스트 - POST /predict 
5. 통합 테스트 - POST /predict/batch
6. 통합 테스트 - GET /model/info
'''
import uuid
from datetime import datetime, timedelta
from unitest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import FIELD_TO_COLUMN, LoanModel

# =================================================================
# 1. 준비 (헬퍼 함수 + fixture)
# =================================================================

# 학습 때 LabelEncoder로 변환했던 범주형(문자형) 컬럼 목록
CATEGORICAL_COLUMNS = ['성별', '주거형태', '대출목적', '상환방식']

def _make_valid_request(**overrides) -> dict:
    """
    스키마(LoanRequest)를 모두 통과하는 "정상 요청 데이터"를 만들어준다.
        함수로 만드는 이유 -> 테스트마다 13개 필드를 반복해서 적지 않아도 된다.
                            스키마가 바뀌면 이 함수 하나만 고치면 된다.
                            **overrides로 "일부 필드만"바꿔서 사용할 수 있다.   
    """
    data = {
        "age": 35,
        "gender": "남",
        "annual_income": 5000.0,
        "employment_years": 5,
        "housing_type": "자가",
        "credit_score": 720,
        "existing_loan_count": 2,
        "annual_card_usage": 2400.0,
        "debt_ratio": 35.5,
        "loan_amount": 3000.0,
        "loan_purpose": "주택구입",
        "repayment_method": "원리금균등",
        "loan_period": 36,
    }
    data.update(overrides) # 넘겨받은 값으로 덮어쓰기
    return data

def _make_mock_encoder() -> MagicMock:
    """
    가짜 LabelEncoder를 만든다. 

    side_effect -> 호출될 때마다 실행되는 함수를 지정하는 기능
    """   
    encoder = MagicMock()
    encoder.transform.side_effect = lambda values: [0] * len(values)
    return encoder

def _set_probabilities(model: LoanModel, *probs: float) -> None:
    """
    Mock 파이프라인이 돌려줄 "승인 확률"을 지정한다.

    predict_proba --> [[거절확률, 승인확률], ...] 형태의 2차원 배열을 반환한다.
    model.py는 이 중 [:, 1] 승인 확률만 꺼내쓰므로 같은 모양으로 만들어준다. 
    """
    model.pipeline.predict_proba.return_value = np.array([[1 - p, p] for p in probs])

@pytest.fixture()
def mock_model():
    """
    테스트용 "가짜 모델"을 만들어 app.state.model에 넣어준다.

    실제 서버는 lifespan에서 model.load()로 pkl파일을 읽지만, 테스트에서는 pkl파일이 없어도
    되도록 진짜 대신 가짜(MaickMock)를 끼워넣는다.

    yield 앞 = 테스트 시작 전 준비(setup)
    yield 뒤 = 테스트 끝난 뒤 정리(teardown)
    """
    model = LoanModel()

    # 진짜 파이프라인 대신 가짜. --> 기본 예측 결과는 "승인 확률 0.8"로 고정한다.
    model.pipeline = MagicMock()
    _set_probabilities(model, 0.8)

    # 범주형 컬럼마다 가짜 encoder를 하나씩 붙인다.
    model.label_encoders = {col: _make_mock_encoder() for col in CATEGORICAL_COLUMNS}

    # model.py가 DataFrame에서 [self.feature_names] 로 컬럼을 고르므로
    # 번역 사전(FIELD_TO_COLUMN)의 한글 컬럼명과 반드시 일치해야 한다.
    model.feature_names = list(FIELD_TO_COLUMN.values())

    app.state.model = model
    yield model  # 여기서 테스트가 실행된다. 

    # 테스트가 끝나면 app.state를 원래대로 되돌린다. (다음 테스트에 영향 방지)
    del app.state.model

@pytest.fixture()
def client(mock_model):
    """
    TestClient를 만들어준다.
    """
    return TestClient(app)

# ==================================================================
# 2. 단위 테스트 - LoanModel 클래스 (서버/HTTP 없이 함수 하나만 검사)
# ==================================================================
@pytest.mark.parametrize(
    'probability, expected_grade',
    [
        (0.0, "D"),
        (0.2499, "D"),   # C 등급 바로 아래
        (0.25, "C"),     # ← 경계값: 0.25 이상부터 C
        (0.4999, "C"),
        (0.5, "B"),      # ← 경계값: 0.5 이상부터 B
        (0.7499, "B"),
        (0.75, "A"),     # ← 경계값: 0.75 이상부터 A
        (1.0, "A"),        
    ]
)
def test_get_risk_grade_boundaries(probability, expected_grade):
    """
    @pytest.mark.parametrize = 같은 테스트를 여러 입력값으로 반복실행한다.
    한 줄이 각각 "독립된 테스트 1개"로 실행되고 결과도 따로 표시된다.

    경계값(0.25, 0.5, 0.75)에서 등급이 바뀌는지 검사하는 것이 핵심    
    """
    assert LoanModel._get_risk_grade(probability) == expected_grade

def test_map_to_korean_converts_field_names():
    """API의 영문 키가 학습 데이터의 한글 컬럼명으로 바뀌는지 검사한다."""
    mapped = LoanModel._map_to_korean({'age': 35, 'credit_score': 720})
    assert mapped == {'나이': 35, '신용점수': 720}

def test_map_to_korean_keeps_unknown_keys():
    """번역 사전에 없는 키는 사라지지 않고 원래 이름 그대로 유지된다."""
    mapped = LoanModel._map_to_korean({'age': 35, 'unknown_field': 'x'})
    assert mapped['나이'] == 35
    assert mapped['unknown_field'] == 'x'

def test_predict_without_load_raises_runtime_error():
    """
    모델을 load()하지 않고, predict()를 호출하면 RuntimeError가 발생해야 한다.

    pytest.raises = 이 코드에서 이 예외가 발생해야 테스트 성공!
    예외가 발생하지 않으면 오히려 테스트가 실패한다.
    match= --> 예외 메시지에 들어있어야 할 문자열(정규식)
    """
    model = LoanModel() 
    with pytest.raises(RuntimeError, match='로드되지 않았습니다'):
        model.predict(_make_valid_request())

def test_predict_batch_withdout_load_raises_runtime_error():
    """배치 예측도 모델 미적재 상태에서는 RuntimeError가 발생해야 한다."""
    model = LoanModel()
    with pytest.raises(RuntimeError, match='로드되지 않았습니다'):
        model.predict_batch([_make_valid_request()])

def test_predict_batch_empty_list_returns_empty(mock_model):
    """빈 리스트를 넣으면 모델을 호출하지 않고 빈 리스트를 반환한다."""
    assert mock_model.predict_batch([]) == []
    mock_model.pipeline.predict_proba.assert_not_called()

# =====================================================================
# 3. 통합 테스트 - GET /, /health
# =====================================================================
def test_root(client):
    """루트 엔드포인트가 200과 data 키를 돌려주는지 검사한다."""
    response = client.get('/')
    assert response.status_code == 200
    assert 'data' in response.json()

def test_health_when_model_loaded(client):
    """모델이 적재된 상태 - status 는 healthy, model_loaded 는 True"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'healthy', 'model_loaded': True}

def test_health_when_model_not_loaded(client, mock_model):
    """
    모델이 없는 상태 - 서버는 살아있으므로 200이 나오지만, status는 degraded

    mock_model과 app.state.model은 "같은 객체" 이므로
    pipeline을 None 으로 바꾸면 서버쪽에도 그대로 반영된다.
    """
    mock_model.pipeline = None

    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'degraded', 'model_loaded': False}

# =========================================================================
# 4. 통합 테스트 - POST /predict (정상 케이스) --> 승인/거절/임계값 경계
# =========================================================================
def test_predict_approved(client, mock_model):
    """승인 시나리오: 승인 확률 0.8 -> approved=True, 등급 A"""
    _set_probabilities(mock_model, 0.8)

    response = client.post('/predict', json=_make_valid_request())

    assert response.status_code == 200
    body = response.json()
    assert body['approved'] is True
    assert body['risk_grade'] == 'A'
    assert body['probability'] == pytest.approx(0.8) # 실수형일 때는 approx로 비교

def test_predict_rejected(client, mock_model):
    """거절 시나리오: 승인 확률 0.2 -> approved=False, 등급 D"""
    _set_probabilities(mock_model, 0.2)

    response = client.post('/predict', json=_make_valid_request())

    assert response.status_code == 200
    body = response.json()
    assert body['approved'] is False
    assert body['risk_grade'] == 'D'
    assert body['probability'] == pytest.approx(0.2) # 실수형일 때는 approx로 비교

def test_predict_threshold_boundary(client, mock_model):
    """
    임계값 경계: 확률이 정확히 0.5이면 승인으로 처리 (probability >= threshold)
    """
    _set_probabilities(mock_model, 0.5)

    body = client.post('/predict', json=_make_valid_request()).json()
    assert body['approved'] is True
    assert body['risk_grade'] == 'B'    

def test_predict_response_has_tracking_metadata(client):
    """
    응답에 요청 추적용 request_id(UUID), timestamp(UTC ISO 8601)가 들어있는지 검사한다.
    이 값들을 매번 달라지므로 "값"이 아니라 "형식"이 올바른지를 검사한다.
    """
    body = client.post('/predict', json=_make_valid_request()).json()

    # UUID 형식이 아니면 uuid.UUID()가 ValueError를 일으켜 테스트가 실패된다.
    uuid.UUID(body['request_id'])    

    # ISO 8601 문자열로 변환 가능해야 하고, 시간대는 UTC(오프셋 0)이어야 한다.
    parsed = datetime.fromtimestamp(body['timestamp'])
    assert parsed.utcoffset() == timedelta(0)

def test_predict_request_id_is_unique(client):
    """같은 요청을 두 번 보내도 request_id 는 매번 새로 발급되어야 한다."""
    payload = _make_valid_request()
    first = client.post('/predict', json=payload).json()['request_id']
    second = client.post('/predict', json=payload).json()['request_id']
    assert first != second

def test_predict_passes_features_in_training_order(client, mock_model):
    """
    모델에 전달되는 DataFrame의 컬럼 순서가 feature_names(학습 때 순서)와 같은지 검사한다.

    학습과 서빙의 컬럼 순서가 어긋나면 Training-Serving Skew 가 생겨 에러 없이 
    "엉뚱한 예측"이 나오게 된다. 반드시 테스트에서 검증해봐야 하는 부분!
    """
    client.post('/predict', json=_make_valid_request(age=42))

    # call_args.args[0] = predict_proba가 실제로 받은 첫번째 인자 (DataFrame)
    df = mock_model.pipeline.predict_proba.call_args.args[0]

    assert list(df.columns) == mock_model.feature_names
    assert df.iloc[0]['나이'] == 42 # 영문 age -> 한글 나이로 매핑되어 전달된다. 
