// DOM 요소들을 미리 선택하여 변수에 저장합니다.
const loanForm = document.getElementById('loan-form');
const submitBtn = document.getElementById('submit-btn');
const btnSpinner = submitBtn.querySelector('.btn-spinner');
const formError = document.getElementById('form-error');

const resultEmpty = document.getElementById('result-empty');
const resultContent = document.getElementById('result-content');
const gaugeFill = document.getElementById('gauge-fill');
const gaugePercent = document.getElementById('gauge-percent');
const decisionBadge = document.getElementById('decision-badge');
const riskGrade = document.getElementById('risk-grade');
const requestId = document.getElementById('request-id');
const timestamp = document.getElementById('timestamp');

// 폼이 제출(submit)될 때 실행될 이벤트 리스너를 등록합니다.
loanForm.addEventListener('submit', async (event) => {
    // 페이지가 새로고침되는 기본 동작을 막습니다.
    event.preventDefault();

    // 이전에 발생했던 에러 메시지와 결과 영역을 초기화합니다.
    formError.textContent = '';
    resultContent.hidden = true;
    resultEmpty.hidden = false;

    // 중복 제출을 방지하기 위해 버튼을 비활성화하고 로딩 스피너를 표시합니다.
    submitBtn.disabled = true;
    btnSpinner.hidden = false;

    try {
        // HTML 폼의 입력 데이터를 쉽게 다루기 위해 FormData 객체로 생성합니다.
        const formData = new FormData(loanForm);
        
        // API 계약에 맞는 JSON 객체를 수동으로 구성하며, 숫자 필드는 반드시 Number()로 변환합니다.
        const requestData = {
            age: Number(formData.get('age')),
            gender: formData.get('gender'),
            annual_income: Number(formData.get('annual_income')),
            employment_years: Number(formData.get('employment_years')),
            housing_type: formData.get('housing_type'),
            credit_score: Number(formData.get('credit_score')),
            existing_loan_count: Number(formData.get('existing_loan_count')),
            annual_card_usage: Number(formData.get('annual_card_usage')),
            debt_ratio: Number(formData.get('debt_ratio')),
            loan_amount: Number(formData.get('loan_amount')),
            loan_purpose: formData.get('loan_purpose'),
            repayment_method: formData.get('repayment_method'),
            loan_period: Number(formData.get('loan_period'))
        };

        // /predict 엔드포인트로 fetch POST 요청을 보냅니다.
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        // 서버 응답이 실패(422, 503 등)일 경우 에러 처리를 수행합니다.
        if (!response.ok) {
            // 서버가 보낸 에러 응답(JSON)을 파싱합니다.
            const errorData = await response.json().catch(() => ({}));
            // 에러 메시지가 있다면 사용하고, 없으면 기본 상태 코드를 표시합니다.
            const message = errorData.detail || `서버 오류가 발생했습니다. (상태 코드: ${response.status})`;
            throw new Error(message);
        }

        // 성공 응답 데이터를 JSON으로 파싱합니다.
        const result = await response.json();

        // 1. 승인 확률(0.0 ~ 1.0)을 백분율(0 ~ 100) 문자열로 변환합니다.
        const percentValue = Math.round(result.probability * 100);
        gaugePercent.textContent = `${percentValue}%`;

        // 2. 원형 게이지(SVG circle)의 둘레 길이를 이용해 확률 바를 채웁니다 (둘레 약 439.8).
        const circumference = 439.8;
        const dashOffset = circumference - (circumference * percentValue) / 100;
        gaugeFill.style.strokeDashoffset = dashOffset;

        // 3. 승인 여부(approved: true/false)에 따라 뱃지 텍스트와 스타일을 설정합니다.
        if (result.approved) {
            decisionBadge.textContent = '대출 승인';
            decisionBadge.className = 'decision-badge approved'; // 승인 스타일 클래스 적용
        } else {
            decisionBadge.textContent = '대출 거절';
            decisionBadge.className = 'decision-badge rejected'; // 거절 스타일 클래스 적용
        }

        // 4. 리스크 등급, 요청 ID, 예측 시각 메타 정보를 화면에 반영합니다.
        riskGrade.textContent = result.risk_grade;
        requestId.textContent = result.request_id;
        timestamp.textContent = new Date(result.timestamp).toLocaleString(); // 보기 쉬운 날짜 형식으로 변환

        // 5. 빈 화면 영역을 숨기고 결과 콘텐츠 영역을 화면에 표시합니다.
        resultEmpty.hidden = true;
        resultContent.hidden = false;

    } catch (error) {
        // 실패 시(422, 503 또는 네트워크 오류 등) alert 대신 화면(#form-error)에 에러 메시지를 표시합니다.
        formError.textContent = error.message;
    } finally {
        // 통신이 성공하든 실패하든 비활성화했던 버튼을 복구하고 로딩 스피너를 숨깁니다.
        submitBtn.disabled = false;
        btnSpinner.hidden = true;
    }
});