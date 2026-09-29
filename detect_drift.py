'''
detect_drift.py
-----------------
간단한 통계 규칙으로 대출 모델의 데이터 드리프트를 점검하는 스크립트

PSI (Population Stability Index) 같은 통계지표로 드리프트 감지
--> 이 코드에서 그 개념을 이해하기 위해서 실습 (쉬운 버전)

원리 : 학습 데이터와 운영 데이터의 평균/비율을 비교해서 차이가 크면 
        "드리프트 발생"으로 판단 --> 재학습 필요

- 외부 모니터링 도구 없이도 MLOps의 기본 흐름 (기준 데이터 -> 운영 데이터 비교 -> 재학습 판단)을
    익히기 위한 코드
- 평균과 범주 비율만 비교하므로 실제 운영에서는 표본 수, 결측치, 통계적 유의성,
    분포 전체(PSI, KS 검정 등)도 함께 고려해야 한다.

1) 두 데이터를 불러와서 shape/표본 수부터 확인 --> load_data()
2) 수치형 컬럼 1개(예: 신용점수)로 먼저 확인 --> check_numerical_drift()
3) 범주형 컬럼 1개(예: 성별)로 확인 --> check_categorical_drift()
4) 최종 승인율 비교 --> check_prediction_drift()
5) 전체 조립 + 재학습 판단 --> main()
'''
import pandas as pd

def load_data():
    """
    학습 기준 데이터와 운영 예측 로그를 CSV에서 불러온다.

    data/loan_data.csv : 모델을 학습할 때 사용한 원본 데이터
                            드리프트를 판단하는 기준선(baseline) 역할
    data/prediction_logs.csv : /predict API가 실제로 받은 요청과
                                예측 결과를 쌓아둔 로그.
                                현재 상태를 대표하는 데이터                           
    """
    train_df = pd.read_csv('data/loan_data.csv')
    pred_df = pd.read_csv('data/prediction_logs.csv')

    # 비교에 사용되는 표본 수를 출력해 빈 파일이나 수집 누락을 쉽게 발견하도록 한다.
    # 운영 로그가 너무 적으면 통계적으로 신뢰할 수 없는 비교가 되므로, 실무에서는
    # "표본 수가 N건 이상일 때만 판정" 같은 방어 로직을 추가한다. 
    print(f'학습 데이터: {len(train_df)}건')
    print(f'예측 로그: {len(pred_df)}건')

    return train_df, pred_df

def check_numerical_drift(train_df, pred_df, columns, threshold=20):
    """
    수치형 피처의 드리프트를 감지한다.
        - 학습 데이터 평균 vs 운영 데이터 평균 
            비교해서 차이가 thredhold % 이상이면 드리프트로 판단
    """
    print('\n[ 수치형 피처 드리프트 체크 ]')
    print('-' * 55)
    print(f'{"피처":^10s}| {"학습 평균":>10s} | {"운영 평균":>10s} | {"차이(%)":>7s} | 결과')
    print('-' * 55)

    # 개별 컬럼의 드리프트 여부를 누적해 마지막 재학습 규칙에 사용
    drift_count = 0

    for col in columns:
        if col not in train_df.columns or col not in pred_df.columns:
            continue  # 컬럼이 없으면 비교할 수 없으므로 다음 컬럼으로 건너뛴다.

        # 산술 평균을 비교(실제 이상치가 많은 컬럼은 평균 대신 중앙값(median)를 사용하는 것이 낫다)
        train_mean = train_df[col].mean()
        pred_mean = pred_df[col].mean()

        # 학습 평균을 분모로 두고, 상대적인 평균 차이를 백분율로 계산
        #   상대 차이를 쓰는 이유 : 피처마다 단위와 크기가 전혀 다르기 때문
        #   abs() : 절대값
        if train_mean != 0:
            diff_pct = abs(train_mean  - pred_mean) / abs(train_mean) * 100
        else:
            diff_pct = 0

        # 임계값과 같은 경우도 드리프트로 판정하기로 한다. >=
        is_drift = (diff_pct >= threshold)
        status = 'DRIFT!' if is_drift else 'OK'

        if is_drift:
            drift_count += 1

        #         피처          학습 평균              운영 평균            차이(%)              결과
        print(f'{col:^10s}| {train_mean:>10.1f} | {pred_mean:>10.1f} | {diff_pct:>6.1f}% | {status}')

    print(f'\n  -> 수치형 드리프트: {drift_count}개 발견')
    return drift_count

def check_categorical_drift(train_df, pred_df, columns, threshold=10):
    """
    범주형 피처의 드리프트를 감지한다.
        - 각 카테고리의 비율을 비교해서 가장 많이 변한 카테고리의 차이가
             threshold%p 이상이면 드리프트
        - 범주형은 평균을 구할 수 없으므로 카테고리별 구성 비율(%) 비교로 드리프트 감지
    """
    print('\n[ 범주형 피처 드리프트 체크 ]')
    print('-' * 55)

    drift_count = 0

    for col in columns:
        if col not in train_df.columns or col not in pred_df.columns:
            continue

        # .value_counts(normalize=True) : 전체 행 수 대비 비율(0~1)을 반환
        #   학습 데이터와 운영 로그는 표본 수가 다르므로 개수가 아닌 비율로 비교해야
        #   공정한 비교가 된다.
        train_ratio = train_df[col].value_counts(normalize=True)
        pred_ratio = pred_df[col].value_counts(normalize=True)

        all_categories = set(train_ratio.index) | set(pred_ratio.index)
        max_diff = 0

        # 존재하지 않는 범주는 비율 0으로 보고, 비율을 %단위로 변환
        for cat in all_categories:
            t = train_ratio.get(cat, 0) * 100
            p = pred_ratio.get(cat, 0) * 100
            diff = abs(t - p)

            # 컬럼 내 여러 범주 중 가장 크게 변한 것을 대표 변화량(max_diff)으로 사용
            #   평균적으로 안 변했지만 특정 카테고리 하나만 급증하는 경우도 놓치지 않기 위한
            #   보수적인(민감한) 설계다.
            max_diff = max(max_diff, diff)

        is_drift = max_diff >= threshold
        status = 'DRIFT!' if is_drift else 'OK'

        if is_drift:
            drift_count += 1

        print(f'{col:^8s}| 최대 비율 차이: {max_diff:.1f}%p | {status}')

    print(f'\n -> 범주형 드리프트: {drift_count}개 발견')
    return drift_count

def check_prediction_drift(train_df, pred_df):
    """
    예측 결과(승인율)의 드리프트를 감지한다.
    학습 데이터의 승인율과 운영 데이터의 승인율을 비교
        - 피처 분포가 안정적이어도 예측 승인율이 달라질 수 있으므로 입력 드리프트와
          출력 드리프트를 별도로 관찰한다.
        - 입력이 그대로여도 환경 등(예: 심사 정책 변경, 경기 상황 변화 등) 모델이 실제로 
          표현하는 결과를 성향이 달라질 수 있기 때문에 두 종류를 모두 감시해야 
          진짜 문제를 놓지지 않을 수 있다. 
    """
    print('\n[ 예측 결과 드리프트 체크 ]')
    print('-' * 55)

    # 승인 여부가 bool 또는 0/1 --> 평균 == 승인 비율
    train_rate = train_df['승인여부'].mean() * 100
    pred_rate = pred_df['approved'].astype(int).mean() * 100

    # 두 비율의 차이는 상대 변화율(%)이 아니라 퍼센트포인트(%p)
    # (수치형 드리프트 함수의 diff_pct와 계산 방식이 다르다.)
    # % : 어떤 값 자체의 비율
    # %p : 두 비율 사이의 뺄셈 결과 (차이)
    diff = abs(train_rate - pred_rate)

    print(f' 학습 데이터 승인율: {train_rate:.1f}%')
    print(f' 운영 데이터 승인율: {pred_rate:0.1f}%')
    print(f' 차이: {diff:.1f}%p')

    return diff

def main():
    """
    모든 드리프트 검사를 순서대로 실행하고, 재학습 필요 여부를 반환한다.
    """
    print('=' * 55)
    print(' 데이터 드리프트 감지 리포트 ')
    print('=' * 55)

    # 1. 비교 기준이 되는 학습 데이터와 최근 운영 로그를 불러온다.
    train_df, pred_df = load_data()

    # 2. 수치형 피처: 학습 평균 대비 운영 평균 차이가 20% 이상인지 검사
    numerical_cols = [
        '나이', '연소득', '근속연수', '신용점수', '기존대출건수', '연간카드사용액',
        '부채비율', '대출신청액', '대출기간',
    ]
    num_drifted = check_numerical_drift(train_df, pred_df, numerical_cols)

    # 3. 범주형 피처: 어느 한 범주의 구성비 차이가 10%p 이상인지 검사
    categorical_cols = ['성별', '주거형태', '대출목적', '상환방식']
    cat_drifted = check_categorical_drift(train_df, pred_df, categorical_cols)

    # 4. 승인율의 변화
    pred_diff = check_prediction_drift(train_df, pred_df)

    # 5. 피처 드리프트 개수와 승인율 차이를 읽기 쉬운 리포트로 출력
    total_drifted = num_drifted + cat_drifted
    print('\n' + '=' * 55)
    print(' 종합 결론')
    print('=' * 55)
    print(f' 드리프트 발생 피처: {total_drifted}개')
    print(f' - 수치형: {num_drifted}개')
    print(f' - 범주형: {cat_drifted}개')
    print(f' 승인율 차이: {pred_diff:.1f}%p')

    # 6. 재학습 정책 
    #   - 드리프트가 발생한 피처가 3개 이상
    #   - 승인율 차이가 10%p 초과(>)
    #   (실무에서는 피처 중요도가 높은 컬럼(신용점수 등)에 가중치를 더 주거나, 
    #    드리프트가 며칠 연속 감지될 때만 재학습을 트리거하는 등 더 정교한 규칙을 사용)
    if total_drifted >= 3 or pred_diff > 10:
        print('\n ** 재학습이 필요합니다! **')
        print('   -> train.py를 실행하여 모델을 재학습하세요.')
        return True
    else:
        print('\n 현재 모델이 안정적입니다.')
        return False

if __name__ == '__main__':
    needs_retrain = main()

