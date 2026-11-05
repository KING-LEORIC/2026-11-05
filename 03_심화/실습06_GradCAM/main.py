# Path 클래스 import
# - 파일 및 디렉터리 경로를 객체 형태로 다루기 위해 사용
from pathlib import Path
# sys 모듈 import
# - 모듈 검색 경로(sys.path) 수정에 사용
import sys,json
# 수치 계산 라이브러리 import
# - 배열 및 행렬 연산에 사용
import numpy as np
# 데이터 처리 라이브러리 import
# - 데이터프레임 처리에 사용
import pandas as pd
# PyTorch import
# - 딥러닝 모델 학습 및 추론에 사용
import torch
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/"day14_vision"))
DATA=BASE.parent/"data"
OUT=BASE/"outputs"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
# 실험 수행에 필요한 모듈 로드
# - experiment : 모델 학습 및 평가 수행
# - datasets : 학습/검증 데이터셋 로드
# - metrics : 성능 평가 함수
from engine import experiment,datasets,metrics
# 모델 생성 및 설정 함수 로드
# - build_model : 모델 생성
# - configure : 모델 설정
from models import build_model,configure
# CAM(Class Activation Map) 생성 함수
# - cam_for(model,input,target)
# - 특정 클래스 예측에 기여한 영역을 시각화
from explain import cam_for
import cv2
# 모델 학습 및 예측 수행
# - epochs=8 : 8 epoch 학습
# - model : 학습 완료된 모델
# - prediction : 예측 결과
model,_,_,prediction=experiment(DATA,epochs=8)
# 검증 데이터셋 로드
# - valid[0] : 첫 번째 검증 샘플
# - x : 입력 이미지
# - _ : 정답 라벨(사용하지 않음)
# 배치 차원 추가
# - [C,H,W] -> [1,C,H,W]
# - 모델 입력 형태에 맞춤
_,valid=datasets(DATA);x,_=valid[0];x=x[None]
"""원본 이미지와 일부가 가려진 이미지를 동일한 target 클래스 기준으로 CAM을 비교하여 모델이 어떤 영역 정보를 활용해 분류하는지 분석"""
# 중심 24×24만 유지하는 조작. 대상 일부도 지울 수 있으므로 원인 증명으로 쓰지 않습니다.
# 입력 이미지 가장자리를 0으로 만들어
# 주변 정보를 제거한 가림(occlusion) 이미지를 생성
changed=x.clone();changed[:,:,:4,:]=0;changed[:,:,-4:,:]=0;changed[:,:,:,:4]=0;changed[:,:,:,-4:]=0
# 원본 이미지와 가림 이미지의 예측 확률 계산
# - softmax(1) : 클래스별 확률 변환
with torch.no_grad():
    original_prob=model(x).softmax(1)
    changed_prob=model(changed).softmax(1)
# 원본 이미지의 예측 클래스 선택
# - argmax(1) : 가장 높은 확률의 클래스
# - target : CAM 비교 기준 클래스
target=int(original_prob.argmax(1).item())
# TODO D06: 두 입력에 같은 target을 고정하여 비교하세요.
# BEGIN D06

# 두 이미지 모두 동일한 target 클래스 기준으로 CAM 생성
# - cams[0] : 원본 이미지 CAM
# - cams[1] : 가림 이미지 CAM
# - 같은 클래스 설명을 비교해야 변화 원인을 분석 가능
cams=[cam_for(model,x,target),cam_for(model,changed,target)]  # 두 번째는 changed를 사용하세요.
# END D06
# CAM 결과를 원본 이미지 위에 덧씌운 후 저장
# - rgb : 입력 이미지를 시각화용 uint8 변환
# - applyColorMap : CAM을 Heatmap으로 변환
# - addWeighted : 이미지와 CAM 합성
# - cv2.imwrite : 결과 이미지 저장
for name,tensor,cam in zip(["original","occluded"],[x,changed],cams):
    rgb=np.uint8(tensor[0].permute(1,2,0).numpy()*255)
    overlay=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.6,cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET),.4,0)
    cv2.imwrite(str(OUT/f"{name}.png"),cv2.resize(overlay,(256,256)))
# 비교에 사용한 target 클래스 출력
print("고정 target:",target)
# target 클래스 확률 변화 출력
# - 원본 이미지 확률
# - 가림 이미지 확률
# - 가림으로 인해 예측 신뢰도가 얼마나 감소했는지 확인
print("원본 확률:",round(original_prob[0,target].item(),4),"가림 확률:",round(changed_prob[0,target].item(),4))

"""
원본 이미지에서는 target 클래스(0)에 대한 예측 확률이 0.9993으로 매우 높았으나, 이미지 가장자리를 제거한 후에는 0.2356으로 크게 감소하였다.
또한 동일한 target에 대해 CAM을 비교함으로써 모델이 분류 판단 시 가장자리 영역의 정보를 상당 부분 활용하고 있음을 확인할 수 있었다.
따라서 해당 모델은 객체 자체뿐 아니라 주변 배경 또는 가장자리 특징에도 의존하여 예측을 수행하는 것으로 해석할 수 있다.
"""