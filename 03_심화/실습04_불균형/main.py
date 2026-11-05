# 파일 및 디렉터리 경로를 객체 형태로 다루기 위한 모듈
from pathlib import Path
# sys : 파이썬 실행 환경 제어
# json : JSON 데이터 처리
import sys,json
# 수치 계산용 라이브러리
import numpy as np
# 데이터프레임 처리용 라이브러리
import pandas as pd
# PyTorch 딥러닝 프레임워크
import torch
# 현재 실행 중인 파이썬 파일 위치
BASE=Path(__file__).resolve().parent
# day14_vision 디렉터리를 import 검색 경로에 추가
sys.path.insert(0,str(BASE.parent/"day14_vision"))
# 데이터셋 디렉터리 경로
DATA=BASE.parent/"data"
# 실험 결과 저장 디렉터리
OUT=BASE/"outputs"
# outputs 폴더가 없으면 생성
OUT.mkdir(exist_ok=True)
# CPU 스레드 수를 1개로 제한하여 실행 결과 변동 최소화
torch.set_num_threads(1)
# experiment : 학습 및 평가 실행 함수
# datasets : 데이터셋 관련 기능
# metrics : 평가 지표 계산 기능
from engine import experiment,datasets,metrics
# build_model : 모델 생성 함수
# configure : 모델 설정 함수
from models import build_model,configure

# 각 실험 결과(summary)를 저장할 리스트
rows=[]
# TODO D04: none, weight, sampler를 각각 독립 후보로 비교하세요. 동시에 적용하지 않습니다.
# BEGIN D04

# 비교할 클래스 불균형 처리 방법 목록
# - none : 불균형 처리 없음
# - weight : 손실함수에 클래스 가중치 적용
# - sampler : 학습 데이터 재샘플링 적용
# 세 방법을 각각 독립적으로 실험
methods=["none", "weight", "sampler"]  # weight와 sampler를 별개 후보로 추가하세요.
# END D04
for method in methods:
    # 지정된 불균형 처리 방법으로 모델 학습 및 평가 수행
    # epochs=8 : 총 8회 학습
    _,summary,history,_=experiment(DATA,balance=method,epochs=8)
    # 실험 결과 요약 저장
    # epoch별 학습 이력 저장
    rows.append(summary);history.to_csv(OUT/f"{method}_history.csv",index=False)
# 모든 실험 결과를 하나의 데이터프레임으로 변환
# 비교 결과 CSV 저장
result=pd.DataFrame(rows);result.to_csv(OUT/"imbalance_comparison.csv",index=False)
# 주요 평가 지표 출력
# - balance : 사용한 불균형 처리 방식
# - accuracy : 전체 정확도
# - macro_f1 : 클래스별 F1 평균
# - triangle_recall : triangle 클래스 재현율
print(result[["balance","accuracy","macro_f1","triangle_recall"]].round(4).to_string(index=False))\
# sampler 적용 시 학습(train) 데이터만 재샘플링함 //검증(valid) 데이터는 실제 분포를 유지하여 평가의 공정성을 확보
print("sampler는 train만 재추출합니다. valid는 원래 분포 그대로입니다.")

"""
본 실험에서는 클래스 불균형 처리 방법(None, Class Weighting, Sampling)을 비교하였다.
실험 결과 세 방법 모두 Accuracy, Macro F1, Triangle Recall이 1.0으로 동일하게 나타났다.
따라서 현재 데이터셋에서는 클래스 불균형이 성능 저하의 주요 원인이 아니며,
Weighting 및 Sampling 적용에 따른 추가 성능 향상은 관찰되지 않았다.
이는 데이터셋이 충분히 분리 가능하거나 문제 난이도가 낮기 때문으로 해석할 수 있다.
"""
