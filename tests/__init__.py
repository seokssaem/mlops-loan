# tests 폴더를 "패키지"로 인식시키기 위한 파일
# 내용은 비워있어도 괜찮다.
# 이 파일이 있으면 pytest가 프로젝트 루트를 import 경로에 자동으로
# 추가해줘서 test_api.py안의 
# from app.main import app 이 잘 동작하게 된다.