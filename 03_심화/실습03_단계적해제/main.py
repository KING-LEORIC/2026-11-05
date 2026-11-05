# 파일/디렉터리 경로를 객체 형태로 편리하게 다루기 위한 모듈
from pathlib import Path
# sys : 모듈 검색 경로 수정 등에 사용
# json : JSON 데이터 처리용
import sys,json
# 수치 계산용 라이브러리
import numpy as np
# 데이터프레임 처리용 라이브러리
import pandas as pd
# PyTorch 딥러닝 프레임워크
import torch
# 현재 실행 중인 파일이 위치한 디렉터리 경로
BASE=Path(__file__).resolve().parent
# day14_vision 폴더를 모듈 탐색 경로에 추가하여 engine, models 등 사용자 정의 모듈 import 가능하게 설정
sys.path.insert(0,str(BASE.parent/"day14_vision"))
# 데이터셋이 저장된 폴더 경로
DATA=BASE.parent/"data"
# 실험 결과 저장 폴더 경로
OUT=BASE/"outputs"
# outputs 폴더가 없으면 생성 //이미 존재하면 오류 발생시키지 않음
OUT.mkdir(exist_ok=True)
# CPU 연산 스레드를 1개로 제한하여 실험 환경 차이로 인한 성능 변동을 최소화
torch.set_num_threads(1)
# experiment : 학습/평가 실행 함수
# datasets : 데이터셋 관련 기능
# metrics : 성능 평가 지표 기능
from engine import experiment,datasets,metrics
# build_model : 모델 생성 함수
# configure : 모델 학습 설정 함수
from models import build_model,configure

# 각 실험 결과(summary)를 저장할 리스트
rows=[]
"""
"처음부터 partial fine-tuning 하는 방법"과
"Head만 먼저 학습한 뒤 5 epoch부터 partial fine-tuning 하는 방법" 중
어느 전략이 더 좋은 성능을 내는지 Accuracy와 Macro-F1으로 비교하는 전이학습 실험
"""
# 두 가지 학습 전략을 순차적으로 비교
# - partial_from_start : 처음부터 partial fine-tuning
# - head_then_partial : head 학습 후 partial fine-tuning
for plan in ["partial_from_start","head_then_partial"]:
    # 현재 계획에 따라 초기 학습 전략 결정
    # partial_from_start → partial
    # head_then_partial → head
    strategy="partial" if plan=="partial_from_start" else "head"
    # TODO D03: head_then_partial은 epoch 5부터 partial을 시작하세요(총 8회).
    # BEGIN D03
    switch=5  # 순차 계획일 때 5를 선택하세요.
    # END D03
    # 실험 수행
    # DATA : 데이터셋 경로
    # strategy : 초기 학습 전략
    # epochs=8 : 총 8 epoch 학습
    # switch_epoch : 전략 전환 시점
    # summary : 최종 성능 요약 정보
    # history : epoch별 학습 기록
    _,summary,history,_=experiment(DATA,strategy=strategy,epochs=8,switch_epoch=switch)
    # 결과에 실험 계획 이름 추가
    # 현재 실험 결과 저장
    summary["plan"]=plan;rows.append(summary)
    # epoch별 성능 기록을 CSV 파일로 저장
    history.to_csv(OUT/f"{plan}_history.csv",index=False)
# 각 실험 결과를 데이터프레임으로 정리하여 출력
# plan : 실험 전략
# accuracy : 정확도
# macro_f1 : Macro F1 점수
# best_epoch : 최고 성능 epoch
# round(4) : 소수점 넷째 자리까지 표시   
print(pd.DataFrame(rows)[["plan","accuracy","macro_f1","best_epoch"]].round(4).to_string(index=False))
# 두 실험의 비교 조건 설명
# epoch 5에서 optimizer를 재구성하여
# head 학습 단계의 optimizer 상태를 초기화한 뒤
# partial 학습을 수행함
# 총 학습 epoch 수는 두 실험 모두 8로 동일
print("epoch 5에서 optimizer 재구성: Head의 이전 적응 상태 초기화. 같은 총 8회 예산 비교.")
# 최종 비교 결과를 CSV 파일로 저장
pd.DataFrame(rows).to_csv(OUT/"stage_comparison.csv",index=False)

"""
partial_from_start와 head_then_partial 전략을 비교한 결과, 두 방법 모두 Accuracy와 Macro-F1이 1.0으로 동일하게 나타났다.
또한 최고 성능 Epoch도 모두 8로 동일하였다. 따라서 본 실험에서는 Epoch 5에서 optimizer를 재구성하여 단계적으로 fine-tuning을 수행하는 전략이
처음부터 partial fine-tuning을 수행하는 전략보다 성능상 이점을 보이지 않았다.
즉, 동일한 학습 예산(8 epochs)에서는 두 전략의 최종 성능이 사실상 동등하다고 결론 내릴 수 있다.
"""