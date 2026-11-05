"""소형 source Backbone과 Head. ImageNet/ResNet18 결과로 표기하지 않습니다."""
# 파일 및 폴더 경로를 처리하기 위한 기능
from pathlib import Path
# NumPy 배열 및 수치 연산을 위한 라이브러리
import numpy as np
# 딥러닝 연산을 위한 PyTorch
import torch
# 신경망 계층과 학습 관련 기능
from torch import nn


# 소형 CNN 모델을 생성하는 함수
# - data : source_backbone.npz 파일이 들어 있는 데이터 폴더
# - pretrained : 제공된 사전 학습 가중치를 사용할지 여부
# - seed : 새로 생성하는 모델의 난수 기준값
# - return : 생성된 Sequential 모델
def build_model(data: Path, pretrained: bool = True, seed: int = 14) -> nn.Sequential:
    """제공된 source 표현을 재사용하고 새 Head는 같은 seed로 초기화합니다."""
    # 모델의 초기 가중치 생성을 위한 난수 기준값 지정
    # - manual_seed(seed) : 같은 seed를 사용하면 초기 가중치 생성의 재현성을 높일 수 있음
    torch.manual_seed(seed)
    # 합성곱 신경망의 계층을 순서대로 연결하여 모델 생성
    # - Conv2d : 이미지에서 특징을 추출하는 합성곱 계층
    # - BatchNorm2d : 채널별 입력값을 정규화하는 계층
    # - ReLU : 음수 값을 0으로 만드는 활성화 함수
    # - MaxPool2d : 특징 지도의 크기를 줄이는 풀링 계층
    # - AdaptiveAvgPool2d : 특징 지도를 지정한 크기로 평균 풀링
    # - Flatten : 다차원 특징 지도를 1차원으로 펼침
    # - Linear : 최종 클래스 점수를 계산하는 완전연결 계층
    # - 입력 이미지 : RGB 3채널 이미지
    # - 출력 클래스 : 3개
    model = nn.Sequential(
        nn.Conv2d(3,8,3,padding=1,bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8,16,3,padding=1,bias=False), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(), nn.Linear(256,3),
    )
    # 사전 학습 가중치를 사용할 경우 저장된 가중치를 모델에 적용
    # - pretrained=True : 제공된 source 가중치 사용
    # - np.load() : npz 파일 불러오기
    # - allow_pickle=False : Pickle 객체 로딩을 허용하지 않음
    if pretrained:
        with np.load(Path(data)/'source_backbone.npz', allow_pickle=False) as saved:
            # 현재 모델의 파라미터와 버퍼를 딕셔너리로 가져오기
            # - state_dict() : 모델의 가중치와 BatchNorm 상태 등을 반환
            state=model.state_dict()
            # npz 파일에 저장된 가중치를 PyTorch Tensor로 변환하여 기존 상태에 반영
            # - saved.files : npz 파일에 저장된 배열 이름 목록
            # - saved[key] : 해당 이름의 NumPy 배열
            # - copy() : 배열의 복사본 생성
            # - torch.from_numpy() : NumPy 배열을 Tensor로 변환
            state.update({key:torch.from_numpy(saved[key].copy()) for key in saved.files})
        # 저장된 사전 학습 가중치를 모델에 적용
        # - load_state_dict() : 모델 파라미터와 버퍼를 전달받은 값으로 변경
        model.load_state_dict(state)
    # 사전 학습 가중치가 적용된 모델 또는 새로 초기화된 모델 반환
    return model


# 학습할 모델 영역을 설정하고 옵티마이저를 생성하는 함수
# - model : 학습할 신경망 모델
# - strategy : head, partial, full 중 학습 전략
# - head_lr : 분류 Head의 학습률
# - backbone_lr : Backbone의 학습률
# - return : 설정된 파라미터 그룹을 사용하는 AdamW 옵티마이저
def configure(model, strategy='head', head_lr=.01, backbone_lr=.001):
    """Head / 마지막 Conv+Head / 전체. Parameter 그룹을 겹치지 않게 만듭니다."""
    # 지정한 학습 전략이 허용된 값인지 확인
    # - head : 분류 Head만 학습
    # - partial : 마지막 Conv 계층과 Head 학습
    # - full : Backbone과 Head 전체 학습
    # - 허용되지 않은 값이면 ValueError 발생
    if strategy not in {'head','partial','full'}:
        raise ValueError('strategy must be head, partial, or full')
    # 우선 모델의 모든 파라미터를 학습하지 않도록 설정
    # - model.parameters() : 모델의 모든 학습 파라미터
    # - requires_grad_(False) : 해당 파라미터의 기울기 계산 비활성화
    for p in model.parameters():p.requires_grad_(False)
    # 분류 Head의 파라미터만 학습 가능하도록 설정
    # - model[9] : Sequential 모델의 10번째 계층인 Linear 분류 계층
    # - requires_grad_(True) : 해당 파라미터의 기울기 계산 허용
    for p in model[9].parameters():p.requires_grad_(True)
    # Head 파라미터를 옵티마이저 그룹에 등록
    # - params : 학습할 파라미터 목록
    # - lr : 해당 파라미터 그룹에 적용할 학습률
    groups=[{'params':list(model[9].parameters()), 'lr':head_lr}]
    # head 전략이 아니라면 Backbone 일부 또는 전체도 학습하도록 설정
    if strategy != 'head':
        # 학습 전략에 따라 Backbone 파라미터 선택
        # - partial : model[4:6], 즉 마지막 Conv와 BatchNorm 계층
        # - full : model[:9], 즉 분류 Head를 제외한 Backbone 전체
        backbone=list(model[4:6].parameters()) if strategy=='partial' else list(model[:9].parameters())
        # 선택된 Backbone 파라미터의 기울기 계산 허용
        for p in backbone:p.requires_grad_(True)
        # Backbone 파라미터를 별도의 학습률로 옵티마이저에 등록
        groups.append({'params':backbone,'lr':backbone_lr})
    # 설정한 파라미터 그룹을 이용해 AdamW 옵티마이저 생성
    # - AdamW : 가중치 감쇠를 분리하여 적용하는 최적화 알고리즘
    # - weight_decay=.0001 : 가중치 감쇠 계수
    return torch.optim.AdamW(groups,weight_decay=.0001)


# 모델의 학습 모드와 BatchNorm 동작을 설정하는 함수
# - model : 설정할 신경망 모델
# - freeze_bn : BatchNorm 통계값을 고정할지 여부
def training_mode(model, freeze_bn=True):
    """eval은 autograd를 끄지 않습니다. BN 통계와 affine gradient는 별도 정책입니다."""
    # 모델 전체를 학습 모드로 전환
    # - train() : Dropout과 BatchNorm 등이 학습 방식으로 동작하도록 설정
    model.train()
    # BatchNorm 통계값을 고정하는 설정이 켜져 있으면 BatchNorm 계층을 평가 모드로 전환
    if freeze_bn:
        # 모델에 포함된 모든 계층을 순회
        for layer in model.modules():
            # 현재 계층이 BatchNorm 계층인지 확인하고 평가 모드로 설정
            # - isinstance() : 객체가 지정한 클래스에 속하는지 확인
            # - _BatchNorm : BatchNorm 계층의 공통 부모 클래스
            # - eval() : BatchNorm의 running mean과 running variance 갱신 중지
            # - affine 파라미터의 기울기 계산 여부는 requires_grad 설정에 따름
            if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()
