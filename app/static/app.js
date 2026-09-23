const form = document.querySelector("#loan-form");
const steps = [...document.querySelectorAll(".form-step")];
const indicators = [...document.querySelectorAll("[data-step-indicator]")];
const nextButton = document.querySelector("#next-button");
const prevButton = document.querySelector("#prev-button");
const submitButton = document.querySelector("#submit-button");
const errorBox = document.querySelector("#form-error");
const resultPanel = document.querySelector("#result-panel");
const debtRange = form.elements.debt_ratio;
const debtOutput = document.querySelector("#debt-output");

let currentStep = 1;

const formatNumber = (value) => new Intl.NumberFormat("ko-KR").format(Number(value));

function setStep(nextStep) {
  currentStep = nextStep;
  steps.forEach((step) => {
    const isActive = Number(step.dataset.step) === currentStep;
    step.classList.toggle("is-active", isActive);
    step.hidden = !isActive;
  });
  indicators.forEach((indicator) => {
    const number = Number(indicator.dataset.stepIndicator);
    indicator.classList.toggle("is-active", number === currentStep);
    indicator.classList.toggle("is-complete", number < currentStep);
  });
  prevButton.hidden = currentStep === 1;
  nextButton.hidden = currentStep === steps.length;
  submitButton.hidden = currentStep !== steps.length;
  hideError();
}

function currentFields() {
  return [...steps[currentStep - 1].querySelectorAll("input, select")];
}

function validateStep() {
  const fields = currentFields();
  const invalid = fields.find((field) => !field.checkValidity());
  if (!invalid) return true;
  invalid.reportValidity();
  invalid.focus();
  showError("입력 범위와 필수 항목을 다시 확인해주세요.");
  return false;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function payloadFromForm() {
  const data = new FormData(form);
  return {
    age: Number(data.get("age")),
    gender: data.get("gender"),
    annual_income: Number(data.get("annual_income")),
    employment_years: Number(data.get("employment_years")),
    housing_type: data.get("housing_type"),
    credit_score: Number(data.get("credit_score")),
    existing_loan_count: Number(data.get("existing_loan_count")),
    annual_card_usage: Number(data.get("annual_card_usage")),
    debt_ratio: Number(data.get("debt_ratio")),
    loan_amount: Number(data.get("loan_amount")),
    loan_purpose: data.get("loan_purpose"),
    repayment_method: data.get("repayment_method"),
    loan_period: Number(data.get("loan_period")),
  };
}

function responseError(response, detail) {
  if (response.status === 422) {
    return "입력값을 처리할 수 없습니다. 선택 항목과 숫자 범위를 다시 확인해주세요.";
  }
  if (response.status === 503) {
    return "현재 예측 모델을 불러오지 못했습니다. 잠시 후 다시 시도해주세요.";
  }
  if (response.status >= 500) {
    return "예측 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.";
  }
  return typeof detail.detail === "string" ? detail.detail : "예측 요청을 처리하지 못했습니다.";
}

function renderResult(result, payload) {
  const percent = Math.round(result.probability * 100);
  const approved = result.approved;
  const title = approved ? "승인 가능성이 높게 예측됐어요" : "승인 가능성이 낮게 예측됐어요";
  const description = approved
    ? "현재 입력 조건은 모델이 학습한 승인 사례와 비교적 가깝습니다. 다만 실제 심사에서는 더 많은 정보와 금융사의 정책이 함께 고려됩니다."
    : "현재 입력 조건은 모델이 학습한 승인 사례와 거리가 있습니다. 어떤 하나의 항목보다 여러 조건의 조합이 결과에 영향을 줍니다.";

  document.querySelector("#result-title").textContent = title;
  document.querySelector("#result-description").textContent = description;
  document.querySelector("#score-value").textContent = `${percent}%`;
  document.querySelector("#score-ring").style.setProperty("--score", `${percent}%`);
  document.querySelector("#risk-grade").textContent = result.risk_grade;
  document.querySelector("#grade-label").textContent = approved ? "상대적으로 낮은 모델 위험" : "상대적으로 높은 모델 위험";
  document.querySelector("#summary-amount").textContent = `${formatNumber(payload.loan_amount)}만원`;
  document.querySelector("#summary-period").textContent = `${payload.loan_period}개월`;
  document.querySelector("#summary-credit").textContent = `${payload.credit_score}점`;
  document.querySelector("#summary-id").textContent = result.request_id.slice(0, 8);
  document.querySelector("#raw-result code").textContent = JSON.stringify(result, null, 2);
  resultPanel.hidden = false;
  resultPanel.dataset.status = approved ? "approved" : "rejected";
  resultPanel.scrollIntoView({ behavior: "smooth", block: "center" });
}

nextButton.addEventListener("click", () => {
  if (validateStep()) setStep(Math.min(currentStep + 1, steps.length));
});

prevButton.addEventListener("click", () => setStep(Math.max(currentStep - 1, 1)));

debtRange.addEventListener("input", () => {
  debtOutput.textContent = `${debtRange.value}%`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!validateStep()) return;

  const payload = payloadFromForm();
  hideError();
  submitButton.disabled = true;
  submitButton.innerHTML = "예측 중…";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(responseError(response, detail));
    }
    renderResult(await response.json(), payload);
  } catch (error) {
    showError(error.message === "Failed to fetch"
      ? "서버에 연결할 수 없습니다. API가 실행 중인지 확인해주세요."
      : String(error.message));
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = "AI 예측 실행 <span aria-hidden=\"true\">↗</span>";
  }
});

document.querySelector("#edit-button").addEventListener("click", () => {
  resultPanel.hidden = true;
  form.scrollIntoView({ behavior: "smooth", block: "start" });
});

document.querySelector("#toggle-raw").addEventListener("click", (event) => {
  const raw = document.querySelector("#raw-result");
  raw.hidden = !raw.hidden;
  event.currentTarget.setAttribute("aria-expanded", String(!raw.hidden));
  event.currentTarget.textContent = raw.hidden ? "API 응답 보기" : "API 응답 닫기";
});

setStep(1);

