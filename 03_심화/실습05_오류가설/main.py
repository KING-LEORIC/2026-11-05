# 파일 및 디렉터리 경로를 객체 형태로 다루기 위한 모듈
from pathlib import Path
# 시스템 설정 및 JSON 처리용 모듈
import sys,json
# 수치 계산 라이브러리
import numpy as np
# 데이터프레임 처리 라이브러리
import pandas as pd
# PyTorch 딥러닝 프레임워크
import torch
# 현재 실행 중인 파이썬 파일이 위치한 디렉터리
BASE=Path(__file__).resolve().parent
# day14_vision 폴더를 모듈 탐색 경로에 추가
# - engine.py, models.py, explain.py 등을 import 가능하게 함
sys.path.insert(0,str(BASE.parent/"day14_vision"))
# 학습/검증 데이터가 저장된 폴더 경로
DATA=BASE.parent/"data"
# 결과 저장 폴더 경로
OUT=BASE/"outputs"
# outputs 폴더가 없으면 생성
OUT.mkdir(exist_ok=True)
# 실행 재현성을 위해 CPU 스레드 수 제한
torch.set_num_threads(1)
# 실험 실행, 데이터셋 처리, 평가 모듈
from engine import experiment,datasets,metrics
# 모델 생성 및 설정 함수
from models import build_model,configure
# 모델 설명(Explainability) 결과 저장 함수
from explain import save_explanations
# 고정된 초기 1회 학습 모델의 실제 오답을 진단합니다. 최종 모델이 아닙니다.
"""1회만 학습한 초기 모델이 어떤 데이터를 틀렸는지 찾아서 오답 원인을 분석하기 위한 코드"""
# 모델 학습 및 예측 수행
# - model : 학습된 모델
# - summary : 성능 요약 정보
# - history : 학습 과정 기록
# - prediction : 검증 데이터 예측 결과 테이블
model,summary,history,prediction=experiment(DATA,epochs=1)
# TODO D05: 실제 라벨과 예측 라벨이 다른 행을 남기세요.
# BEGIN D05

# 오답 샘플만 추출
# - actual : 실제 정답 라벨
# - predicted : 모델 예측 라벨
# - actual != predicted : 분류 실패 샘플 선택
# - copy() : 원본과 독립된 데이터프레임 생성
wrong=prediction[prediction["actual"]!=prediction["predicted"]].copy()  # 오답 행을 선택하세요.
# END D05
# 오답 원인에 대한 가설 작성용 컬럼 추가
wrong["hypothesis"]="직접 관찰 후 작성: 데이터/라벨/도메인/모델"
# 가설 검증을 위한 추가 확인 사항 작성 컬럼 추가
wrong["next_check"]="가설을 검증할 추가 비교를 작성"
# 오답 분석 결과 저장
wrong.to_csv(OUT/"error_hypotheses.csv",index=False)
# 모델 설명 결과 생성 및 저장
save_explanations(DATA,OUT,model,prediction)
# 발견된 오답 샘플 개수 출력
print("오답:",len(wrong),"없는 오류 유형은 만들어내지 않습니다.")
