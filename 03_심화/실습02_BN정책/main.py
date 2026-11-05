from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
# 현재 파일이 위치한 디렉터리 경로
# - __file__ : 현재 실행 중인 파이썬 파일
# - resolve() : 절대 경로 변환
# - parent : 상위 디렉터리
BASE=Path(__file__).resolve().parent
# day14_vision 디렉터리를 모듈 검색 경로에 추가
# - engine, models 등의 사용자 정의 모듈 import 가능
sys.path.insert(0,str(BASE.parent/"day14_vision"))
# 데이터셋 디렉터리 경로
DATA=BASE.parent/"data"
# 결과 저장 디렉터리 경로
OUT=BASE/"outputs"
# outputs 디렉터리 생성
# - 이미 존재하면 오류 발생하지 않음
OUT.mkdir(exist_ok=True)
# PyTorch CPU 스레드 수 제한
# - 실험 환경에 따른 성능 편차 감소
torch.set_num_threads(1)
from engine import experiment,datasets,metrics
from models import build_model,configure

# 각 실험 결과를 저장할 리스트
rows=[]
# BatchNorm 고정/비고정 두 가지 정책 비교
# - True : BatchNorm 통계 고정
# - False : BatchNorm 통계 갱신
"""BatchNorm을 고정(freeze)했을 때와 학습 중 업데이트했을 때 성능과 running mean 변화량(bn_shift)을 비교"""
for frozen in [True,False]:
    # TODO D02: freeze_bn에 현재 frozen을 전달하세요.
    # BEGIN D02

    # BatchNorm 정책을 지정하여 모델 학습 수행
    # - DATA : 학습/검증 데이터가 저장된 경로
    # - freeze_bn=frozen : 현재 BatchNorm 고정 여부(True/False)
    # - epochs=6 : 총 6회 학습 수행
    # - model : 학습 완료된 모델
    # - summary : 성능 평가 결과(accuracy, macro_f1 등)
    # - history : epoch별 학습 이력
    # - prediction : 검증 데이터 예측 결과
    model,summary,history,prediction=experiment(DATA,freeze_bn=frozen,epochs=6)  # frozen 값을 전달하세요.
    # END D02
    # 초기 상태의 모델 생성
    # - BatchNorm running statistics 비교용 기준 모델
    source=build_model(DATA)
    # 결과 요약 정보에 BatchNorm 정책 기록
    summary["freeze_bn"]=frozen
    # BatchNorm running_mean 변화량 계산
    # - model[1] : BatchNorm 계층 가정
    # - 현재 모델과 초기 모델의 running_mean 차이
    # - abs().max() : 최대 절대 변화량
    summary["bn_shift"]=(model[1].running_mean-source[1].running_mean).abs().max().item()
    # 실험 결과 저장
    rows.append(summary)
# 실험 결과를 데이터프레임으로 변환
# - 각 행은 BatchNorm 정책별 결과
# 결과 CSV 파일 저장
# - accuracy, f1, bn_shift 등 포함
table=pd.DataFrame(rows);table.to_csv(OUT/"bn_policy.csv",index=False)
# 핵심 성능 지표 출력
# - freeze_bn : BatchNorm 고정 여부
# - accuracy : 정확도
# - macro_f1 : Macro F1 score
# - bn_shift : BatchNorm 통계 변화량
print(table[["freeze_bn","accuracy","macro_f1","bn_shift"]].round(4).to_string(index=False))
