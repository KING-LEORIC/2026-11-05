# 파일 및 폴더 경로를 처리하기 위한 기능
from pathlib import Path
# 배열 및 수치 연산을 위한 NumPy
import numpy as np
# CSV 파일을 읽고 표 형태의 데이터를 처리하기 위한 pandas
import pandas as pd
# 딥러닝 연산을 위한 PyTorch
import torch
# 검증 데이터를 배치 단위로 제공하는 기능
from torch.utils.data import DataLoader
# 학습·검증 데이터셋 생성 및 평가 함수
from engine import datasets,evaluate
# 학습 때 사용한 모델 구조를 다시 생성하는 함수
from models import build_model
# 현재 파일을 기준으로 데이터 폴더와 결과 폴더 경로 설정
# - __file__ : 현재 실행 중인 Python 파일의 경로
# - resolve().parent : 현재 파일이 위치한 절대 경로의 상위 폴더
# - DATA : 이미지 데이터가 저장된 폴더
# - OUT : 학습 결과가 저장된 폴더
BASE=Path(__file__).resolve().parent;DATA=BASE/"data";OUT=BASE/"outputs"
# CPU 연산 스레드 수를 1개로 제한
torch.set_num_threads(1)
# 직접 생성한 파일만 읽습니다. 임의 출처의 직렬화 파일을 로드하지 않습니다.
# 저장된 최적 모델 정보 불러오기
# - OUT/"best_model.pt" : main.py에서 저장한 최적 모델 파일
# - map_location="cpu" : 저장된 Tensor를 CPU 메모리로 불러오기
# - weights_only=True : 가중치와 기본 자료형 중심으로 안전하게 불러오기
# - saved : 저장된 모델 가중치와 클래스·입력 정보가 담긴 딕셔너리
saved=torch.load(OUT/"best_model.pt",map_location="cpu",weights_only=True)
# 학습 때와 동일한 구조의 모델 생성
# - build_model(DATA) : models.py의 모델 생성 함수 실행
# - 기본값 pretrained=True : 제공된 source 가중치로 모델 초기화
# - 이후 저장된 최적 가중치를 적용하여 학습 당시 모델 상태 복원
model=build_model(DATA)
# TODO D10: model.load_state_dict로 저장 가중치를 복원하세요.
# BEGIN D10
# pass  # 저장 모델 가중치를 적용하세요.

# 저장된 최적 모델의 가중치를 현재 모델에 적용
# - saved["model_state_dict"] : main.py에서 저장한 최적 모델 가중치
# - load_state_dict() : 저장된 가중치와 현재 모델의 계층별 파라미터를 대응시켜 복원
model.load_state_dict(saved["model_state_dict"])
# END D10
# 복원한 모델을 평가 모드로 전환하고 검증 데이터셋 불러오기
# - model.eval() : 평가 모드로 전환
# - datasets(DATA) : 학습·검증 데이터셋 생성
# - valid : 검증 데이터셋
model.eval();_,valid=datasets(DATA)
# 복원된 모델로 검증 데이터의 예측 결과 계산
# - DataLoader() : 검증 데이터를 배치 단위로 제공
# - batch_size=32 : 한 번에 이미지 32개씩 처리
# - shuffle=False : 검증 데이터 순서를 섞지 않음
# - evaluate() : 검증 손실, 평가 지표, 실제 라벨, 예측 확률 반환
# - summary : 평가 지표
# - y : 실제 정답 라벨
# - prob : 클래스별 예측 확률
summary,y,prob=evaluate(model,DataLoader(valid,batch_size=32,shuffle=False))
# main.py에서 저장한 기존 예측 결과 CSV 파일 불러오기
# - predictions.csv : 학습 당시 최적 모델의 이미지별 예측 결과
# - expected : 기존 예측 결과 DataFrame
expected=pd.read_csv(OUT/"predictions.csv")
# 검증 데이터의 이미지 경로 목록 생성
# - valid.samples : 이미지 경로와 라벨의 목록
# - Path(p).relative_to(DATA) : 데이터 폴더 기준 상대 경로 생성
# - paths : 검증 이미지의 상대 경로 목록
paths=[str(Path(p).relative_to(DATA)) for p,_ in valid.samples]
# 복원한 모델의 이미지 순서가 기존 예측 결과와 같은지 확인
# - paths : 현재 검증 데이터의 이미지 경로 목록
# - expected.path : 기존 예측 결과에 기록된 이미지 경로 목록
# - == : 두 목록의 순서와 내용이 모두 같은지 비교
print("ID 순서:",paths==expected.path.tolist())
# 복원한 모델의 예측 확률이 기존 예측 확률과 같은지 확인
# - prob : 복원한 모델의 클래스별 예측 확률
# - expected[["p0","p1","p2"]] : 기존에 저장한 클래스별 예측 확률
# - to_numpy() : DataFrame을 NumPy 배열로 변환
# - np.allclose() : 두 배열의 값이 허용 오차 안에서 같은지 확인
# - atol=1e-6 : 절대 오차 허용 범위
# - rtol=1e-5 : 상대 오차 허용 범위
print("확률 일치:",np.allclose(prob,expected[["p0","p1","p2"]].to_numpy(),atol=1e-6,rtol=1e-5))
# 복원한 모델의 최종 예측 라벨이 기존 예측 라벨과 같은지 확인
# - prob.argmax(1) : 각 이미지에서 가장 높은 확률을 가진 클래스 선택
# - expected.predicted : 기존에 저장한 예측 라벨
# - np.array_equal() : 두 배열의 모든 값과 순서가 같은지 확인
print("예측 일치:",np.array_equal(prob.argmax(1),expected.predicted.to_numpy()))
