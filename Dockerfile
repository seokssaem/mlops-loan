# 1. 베이스 이미지
FROM python:3.10-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 의존성 파일 복사
COPY requirements.txt .

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 보안: non-root 사용자 생성
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# 6. 애플리케이션 코드와 추론에 필요한 모델 아티팩트 복사
COPY app/ ./app/
COPY models/ ./models/

# 7. 포트 문서화
EXPOSE 8000

# 8. 실행 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

