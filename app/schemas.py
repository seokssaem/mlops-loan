'''
app/schemas.py
------------------
대출 API 요청/응답의 데이터 계약을 정의하는 Pydantic 스키마

"스키마 - 모델 - API" 
3단계 중 가장 먼저 확정해야 하는 1단계 작업
여기서 필드 이름과 제약 조건이 흔들리면 model.py의 매핑과 main.py의 
엔드포인트 시그니처까지 함께 흔들리므로, 실무에서는 스키마 설계를 가장 먼저하고
리뷰를 거친다.
'''
from pydantic import BaseModel, Field

class LoanRequest(BaseModel):
    """
    대출 심사를 위한 고객 정보 입력 스키마
    
    ge/le --> 모델이 학습한 범위 밖의 입력을 API 단계에서 미리 차단
        ex) 나이 200세, 신용점수 9999점  처럼 학습 데이터에 없던 값을 모델에
        넣으면 예측 결과를 신뢰할 수 없다.
        Padantic이 이런 값을 모델 호출 전에 422 에러로 걸러주므로 
        model.py에는 별도의 방어코드가 필요없다.
    """
    age: int = Field(..., 
                     ge=19, # >=  이상  (gt >)
                     le=100,  # <=  이하  (lt <)
                     description='나이',
                     examples=[35],
    )

    gender: str = Field(...,
                        description='성별',
                        examples=['남'],
    )

    annual_income: float = Field(
        ...,
        ge=0,  # 연소득은 음수가 될 수 없다. 
        description='연소득',
        examples=[5000.0],
    )

    employment_years: int = Field(
        ...,
        ge=0,
        le=50,
        description='근속연수',
        examples=[5],
    )

    housing_type: str = Field(
        ...,
        description='주거형태',
        examples=['자가'],
    )

    credit_score: int = Field(
        ...,
        ge=300,
        le=900,
        description='신용점수',
        examples=[720],
    )

    existing_loan_count: int = Field(
        ...,
        ge=0,
        description='기존대출건수',
        examples=[2],
    )

    annual_card_usage: float = Field(
        ...,
        ge=0,  # 연간카드사용액 역시 음수 입력을 허용하지 않는다.
        description='연간카드사용액',
        examples=[2400.0],
    )

    debt_ratio: float = Field(
        ...,
        ge=0,  # 부채비율은 백분율 값으로 지정
        le=100,  # 0~100 범위로 제한한다.
        description='부채비율',
        examples=[35.5],
    )

    loan_amount: float = Field(
        ...,
        ge=100,
        description='대출신청액',
        examples=[3000.0],
    )

    repayment_method: str = Field(
        ...,
        description='상환방식',
        examples=['원리금균등'],  # 상환방식 --> 원금균등, 원리금균등, 만기일시
    )

    loan_period: int = Field(
        ...,
        ge=6,   # 대출기간은 개월 단위, 6개월~360개월로 제한한다.
        le=360,
        description='대출기간',
        examples=[36],
    )

class LoanResponse(BaseModel):
    """
    모델 추론 결과를 클라이언트에게 전달하는 응답 스키마.

    model.py가 돌려주는 dict에 혹시 다른 키가 섞여있어도, 이 스키마에 정의된 필드만
    골라서 응답하므로 내부 구현 세부사항이 실수로 클라이언트에게 노출되는 것을 막아준다.
    """
    # 임계값을 통과한 최종 승인 여부
    approved: bool = Field(
        ...,  
        description='승인 여부 (True=승인, False=거절)',
    )

    # 모델이 계산한 승인 클래스 확률
    probability: float = Field(
        ...,
        ge=0.0,  # 유효 범위를 다시 검증, model.py 내부에서 이미 0~1사이 값만 나온다.
        le=1.0,  # -> 응답 스키마에도 제한 두면 혹시 모를 계산 실수까지 이중으로 방어할 수 있다.
        description='승인 확률 (0.0 ~ 1.0)'
    )

    # 확률 구간을 A, B, C, D 로 단순화한 업무용 위험 등급
    risk_grade: str = Field(
        ...,
        description='리스크 등급 (A, B, C, D)'
    )
