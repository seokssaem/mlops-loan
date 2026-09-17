'''
app/main.py
------------------
대출 승인 예측 모델을 HTTP API로 제공하는 FastAPI 애플리케이션
    1. 서버 시작 시 학습된 모델 아티팩트를 메모리에 적재한다.
    2. 상태 확인용 엔드포인트와 대출 승인 예측 엔드포인트를 노출한다.
    3. 모델 계층에서 발생한 예외를 HTTP 상태 코드로 변환한다.

"스키마 - 모델 - API" 
3단계 중 3단계 작업 (조립 담당)
    --> 분리하면 좋은 점
        모델을 XGBoost에서 다른 알고리즘으로 바꿔도 model.py만 고치면 되고,
        API 응답 형식을 바꿀 때도 main.py나 schemas.py만 보면 된다.
'''
import logging
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.model import LoanModel
from app.schemas import LoanRequest, LoanResponse, BatchLoanRequest, BatchLoanResponse, ModelInfoResponse, EnhancedLoanResponse

# 애플리케이션 전체의 기본 로그 레벨을 INFO로 설정한다.
# __name__ 기반 로거를 사용한 로그에 현재 모듈 이름이 함께 기록된다.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 서버의 시작(startup)과 종료(shutdown)  생명주기를 관리한다.

    yield 이전은 서버가 요청을 받기 전에 한 번 실행되고, 이후는 서버가 종료될 때 한 번 실행.
    모델을 요청마다 다시 읽지 않고 시작 시 한 번만 로드하므로 예측 지연과 디스크I/O를 줄일 수 있다.
    (전역 변수로 로드 --> 테스트가 어렵고 로드 실패 시 서버 자체가 멈추기 때문에 좋지 않은 방법)
   
    """
    logger.info('대출 심사 모델을 로드합니다.')

    # 먼저 래퍼 객체를 만든 뒤 직렬화된 모델/인코더/특성 목록을 로드한다.
    model = LoanModel()

    try:
        model.load()
        logger.info('모델 로드 성공')
    except Exception as e:
        logger.error(f'모델 로드 실패: {e}')
        logger.warning('/predict 엔드포인트는 모델 로드 후 사용 가능')

    # app.state --> FastAPI 앱 하나에 딸린 공용 보관함 같은 객체
    #       여기에 넣어둔 model 은 이후 /health, /predict 같은 모든 요청에서 app.state.model로 
    #       꺼내 사용할 수 있다.
    #       모델을 한 번만 로드해서 모든 요청이 같은 모델 인스턴스를 공유하게 만드는 부분이다.
    #       (전역 변수 대신 app.state를 쓰는 이유는 FastAPI 앱 객체에 속하므로
    #          테스트할 때 앱 인스턴스별로 독립적인 상태를 가질 수 있기 때문.)
    app.state.model = model

    yield # 제어권을 FastAPI에 넘겨 실제 요청 처리를 시작한다.

    logger.info('대출 심사 API를 종료합니다.')


app = FastAPI(
    title='대출 심사 예측 API',
    description='ML 모델 기반 대출 승인 여부를 예측하는 API',
    version='1.0.5',
    lifespan=lifespan
)

@app.get('/')
async def root():
    """서버가 기본 요청에 응답하는지 빠르게 확인하는 엔드포인트"""
    return {'data': '서버 동작~~!! 스타투!!'}

@app.get('/health')
async def health_check():
    """
    프로세스 뿐만 아니라 모델 적재 상태까지 포함한 헬스체크를 반환
    
    로드밸런서(ALB)가 이 엔드포인트를 주기적으로 호출해 서버 상태를 확인한다.
        단순하게 프로세스가 살아있는지만 확인하면 부족하다.
            웹 서버는 응답하지만, 모델이 로드되지 않아 모든 예측이 실패하는 상태를 놓칠 수 있다.
            model_loaded 값까지 함께 확인하는 것이 좋다.    
    """
    model = app.state.model

    # 파이프라인이 없으면 웹 서버는 살아있어도 예측 기능은 사용할 수 없다.
    model_loaded = model.pipeline is not None

    return {
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded
    }

@app.post('/predict', response_model=EnhancedLoanResponse)
async def predict(request: LoanRequest):
    """
    검증된 고객 정보를 모델에 전달하고 표준 응답 스키마로 반환한다.

    FastAPI는 함수 호출 전에 JSON 요청 본문을 LoanRequest로 검증하고, 반환값도 EnhancedLoanResponse 형식에
    맞는지 다시 확인 한다.    
    """
    model = app.state.model

    try:
        # Pydantic 객체를 순수한 dict로 바꾸어 모델 계층과 API계층을 분리한다. (느슨한 결합)
        result = model.predict(request.model_dump())

        # ------------------- 요청 추적용 메타데이터 (신규 추가) ----------------------
        # request_id / timestamp는 "무엇을 예측했는가"가 아니라, "언제, 어떤 요청이었는가"를
        #   남기는 정보이므로 model.py가 아니라 main.py API 계층에서 만든다.
        #   model.py의 predict()는 그대로 두고, 이 값만 API 등답에 얹는다.
        #   - request_id --> 매 요청마다 새로 발급되는 UUID. 로그에서 "이 요청 하나"를
        #                       정확히 찾아낼 수 있는 키가 된다.
        #   - timestamp --> UTC 기준 ISO 8601 문자열. timezone.utc를 명시해야 서버가 어느시간대에
        #                       있든 로그 시각 해석이 항상 같다.
        return EnhancedLoanResponse(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            approved=result['approved'],
            probability=result['probability'],
            risk_grade=result['risk_grade'],
        ) 
    except RuntimeError as e:
        # 모델이 준비되지 않은 상태는 일시적인 서비스 불가로 표현 
        # 503 에러 : Graceful Degradation의 HTTP 표현
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail='입력값처리오류')
    except Exception as e:
        logger.error(f'예측 처리 중 예상치 못한 오류 발생 : {e}', exc_info=True)
        raise HTTPException(status_code=500)

# ---------------------------------------------------------------------------
# 배치 예측용 엔드포인트 (신규 추가)
# ---------------------------------------------------------------------------    
@app.post('/predict/batch', response_model=BatchLoanResponse)
async def predict_batch(request: BatchLoanRequest):
    """
    여러 건의 대출 심사를 한 번의 요청으로 처리한다.
    """
    model = app.state.model

    try:
        data_list = [item.model_dump() for item in request.requests]
        results = model.predict_batch(data_list)
        return BatchLoanResponse(results=[LoanResponse(**r) for r in results])
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail='입력값처리오류')
    except Exception as e:
        logger.error(f'배치 예측 처리 중 예상치 못한 오류 발생: {e}', exc_info=True)
        raise HTTPException(status_code=500)

# ---------------------------------------------------------------------------
# 모델 정보 엔드포인트 (신규 추가)
#   API로 노출하면 좋은 점:
#       - CI/CD 파이프라인에서 배포 후 자동 검증 가능
#       - 모니터링 대시보드에서 실시간 모델 상태 표시
#       - 디버깅 시 "어떤 모델이 이 결과를 냈는지" 추적 가능
# ---------------------------------------------------------------------------    
@app.get('/model/info', response_model=ModelInfoResponse)
async def model_info():
    model = app.state.model

    return ModelInfoResponse(
        model_name='loan-approval-xgboost',
        model_version=model.model_version,
        features=model.feature_names,
        threshold=model.threshold,
    )