"""공식 ImageNet ResNet18 선택 경로. 실행: python resnet18_official.py --strategy head
공식 가중치가 없으면 실제 오류를 내며, weights=None으로 자동 대체하지 않습니다.
"""
# 파일 및 폴더 경로를 처리하기 위한 기능
from pathlib import Path
# 명령행 인자 처리, JSON 저장, 시간 측정, 모델 가중치 복사 기능
import argparse,json,time,copy
# 딥러닝 연산을 위한 PyTorch
import torch
# 신경망 계층과 손실 함수 사용
from torch import nn
# 데이터를 배치 단위로 제공하는 기능
from torch.utils.data import DataLoader
# 폴더 구조를 이용해 이미지 데이터셋을 만드는 기능
from torchvision.datasets import ImageFolder
# 공식 ResNet18 모델과 ImageNet 사전 학습 가중치 사용
from torchvision.models import resnet18,ResNet18_Weights
# 모델의 분류 성능을 계산하는 평가 지표
from sklearn.metrics import accuracy_score,f1_score


# 공식 ImageNet ResNet18 모델을 학습하고 평가하는 메인 함수
def main():
    # 명령행에서 전달받을 실행 옵션 설정
    # - ArgumentParser() : 명령행 인자 해석기 생성
    # - --strategy : 학습 전략 선택
    # - choices : 허용할 학습 전략 목록
    # - default='head' : 별도 옵션이 없으면 Head만 학습
    # - --epochs : 학습 반복 횟수
    # - type=int : 입력값을 정수로 변환
    # - default=2 : 기본 학습 횟수는 2 epoch
    p=argparse.ArgumentParser();p.add_argument('--strategy',choices=['head','partial'],default='head');p.add_argument('--epochs',type=int,default=2);a=p.parse_args()
    # CPU 연산 스레드와 난수 기준값 설정
    # - torch.set_num_threads(1) : CPU 연산 스레드 수를 1개로 제한
    # - torch.manual_seed(14) : 난수 생성 기준값 지정
    torch.set_num_threads(1);torch.manual_seed(14)
    # 프로젝트 폴더와 데이터·결과 폴더 경로 설정
    # - __file__ : 현재 Python 파일의 경로
    # - resolve().parent : 현재 파일이 위치한 절대 경로의 상위 폴더
    # - data : 학습·검증 이미지 폴더
    # - outputs/imagenet : 공식 ImageNet 모델의 결과 저장 폴더
    # - mkdir(parents=True,exist_ok=True) : 상위 폴더까지 필요한 폴더 생성
    base=Path(__file__).resolve().parent;data=base/'data';out=base/'outputs'/'imagenet';out.mkdir(parents=True,exist_ok=True)
    # 공식 ImageNet 사전 학습 가중치 선택
    # - IMAGENET1K_V1 : ImageNet 데이터셋으로 사전 학습된 ResNet18 가중치
    weights=ResNet18_Weights.IMAGENET1K_V1
    # 공식 사전 학습 가중치를 사용하는 ResNet18 모델 생성
    # - resnet18() : ResNet18 신경망 구조 생성
    # - weights=weights : 공식 ImageNet 사전 학습 가중치 적용
    # - 공식 가중치 다운로드 또는 캐시 로딩에 실패하면 여기서 오류 발생
    model=resnet18(weights=weights)  # 공식 다운로드 또는 공식 캐시. 실패 시 여기서 중단.
    # 기존 ResNet18의 모든 파라미터를 우선 학습하지 않도록 설정
    # - model.parameters() : 모델의 모든 학습 파라미터
    # - requires_grad_(False) : 기울기 계산 비활성화
    for param in model.parameters():param.requires_grad_(False)
    # 기존 ResNet18의 최종 분류 계층을 3개 클래스 분류 계층으로 교체
    # - model.fc : ResNet18의 최종 완전연결 분류 계층
    # - in_features : 기존 분류 계층의 입력 특성 수
    # - nn.Linear() : 완전연결 계층 생성
    # - 3 : circle, square, triangle 세 클래스로 분류
    model.fc=nn.Linear(model.fc.in_features,3)
    # 새로 만든 분류 Head를 학습 파라미터 그룹에 등록
    # - params : 학습할 파라미터 목록
    # - lr=.001 : Head의 학습률
    groups=[{'params':list(model.fc.parameters()),'lr':.001}]
    # partial 전략이면 ResNet18의 마지막 residual block도 학습하도록 설정
    if a.strategy=='partial':
        # layer4의 파라미터를 학습 가능하도록 변경
        # - layer4 : ResNet18의 마지막 residual block
        # - requires_grad_(True) : 기울기 계산 허용
        for param in model.layer4.parameters():param.requires_grad_(True)
        # layer4 파라미터를 별도의 학습률로 옵티마이저에 등록
        # - lr=.0001 : 마지막 residual block의 학습률
        groups.append({'params':list(model.layer4.parameters()),'lr':.0001})
    # 실행 가능한 연산 장치 선택 및 모델 이동
    # - torch.cuda.is_available() : CUDA GPU 사용 가능 여부 확인
    # - cuda : CUDA GPU 사용
    # - cpu : GPU를 사용할 수 없으면 CPU 사용
    # - model.to(device) : 모델을 선택한 연산 장치로 이동
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    # 공식 ResNet18 가중치에 맞는 이미지 전처리 과정 지정
    # - weights.transforms() : 사전 학습 가중치에 맞는 전처리 설정 반환
    # - RGB : RGB 색상 이미지 사용
    # - Resize(256) : 이미지 크기를 조정
    # - CenterCrop(224) : 중앙 영역을 224×224 크기로 자름
    # - ImageNet Normalize : ImageNet 학습에 사용한 평균과 표준편차로 정규화
    transform=weights.transforms()  # RGB / Resize256 / CenterCrop224 / ImageNet Normalize
    # 학습 이미지와 검증 이미지 데이터셋 생성
    # - ImageFolder() : 폴더 이름을 클래스 이름으로 인식
    # - transform : 공식 ImageNet 전처리 적용
    train=ImageFolder(data/'train',transform);valid=ImageFolder(data/'valid',transform)
    # 데이터 순서 섞기에 사용할 난수 생성기 생성
    # - manual_seed(14) : 난수 기준값 지정
    g=torch.Generator().manual_seed(14)
    # 학습 및 검증 DataLoader 생성
    # - batch_size=8 : 한 번에 이미지 8개씩 모델에 전달
    # - shuffle=True : 학습 데이터 순서를 무작위로 섞음
    # - shuffle=False : 검증 데이터 순서는 유지
    # - generator=g : 학습 데이터 순서 섞기에 사용할 난수 생성기
    # - num_workers=0 : 별도의 데이터 로딩 프로세스를 사용하지 않음
    tl=DataLoader(train,batch_size=8,shuffle=True,generator=g,num_workers=0);vl=DataLoader(valid,batch_size=8,shuffle=False,num_workers=0)
    # 학습에 필요한 옵티마이저와 손실 함수, 기록 변수 초기화
    # - AdamW : 가중치 감쇠를 분리하여 적용하는 최적화 알고리즘
    # - groups : 학습할 Head와 layer4 파라미터 그룹
    # - weight_decay=.0001 : 가중치 감쇠 계수
    # - CrossEntropyLoss() : 다중 클래스 분류 손실 함수
    # - history : epoch별 평가 기록
    # - best : 가장 낮은 검증 손실
    opt=torch.optim.AdamW(groups,weight_decay=.0001);criterion=nn.CrossEntropyLoss();history=[];best=float('inf')
    # 지정된 epoch 수만큼 학습과 검증 반복
    for epoch in range(1,a.epochs+1):
        # 모델을 학습 모드로 전환
        # - train() : 학습 시 사용하는 계층 동작 활성화
        model.train()
        # 모델 안의 BatchNorm 계층을 찾아 통계값 고정
        for module in model.modules():
            # BatchNorm 계층을 평가 모드로 전환
            # - isinstance() : 해당 계층이 BatchNorm2d인지 확인
            # - eval() : running mean과 running variance 갱신 중지
            if isinstance(module,nn.BatchNorm2d):module.eval()  # 동결/부분학습 모두 BN 통계 고정
        # 학습 데이터를 배치 단위로 순회하며 모델 가중치 갱신
        for x,y in tl:
            # 이미지와 정답 라벨을 연산 장치로 이동
            # - x : 이미지 Tensor
            # - y : 정답 라벨 Tensor
            # - to(device) : CPU 또는 CUDA 장치로 이동
            # 기울기를 초기화하고 예측·손실·역전파·가중치 갱신 수행
            # - zero_grad(set_to_none=True) : 이전 기울기 초기화
            # - model(x) : 순전파 수행
            # - criterion() : 예측값과 정답으로 손실 계산
            # - backward() : 역전파로 기울기 계산
            # - step() : 모델 파라미터 갱신
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True);loss=criterion(model(x),y);loss.backward();opt.step()
        # 검증을 위해 모델을 평가 모드로 전환하고 결과 기록 변수 초기화
        # - ys : 실제 정답 라벨 목록
        # - preds : 예측 라벨 목록
        # - total : 검증 손실 합계
        model.eval();ys=[];preds=[];total=0
        # 검증 단계에서는 기울기 계산을 수행하지 않음
        with torch.no_grad():
            # 검증 데이터를 배치 단위로 순회
            for x,y in vl:
                # 이미지와 정답 라벨을 연산 장치로 이동하고 모델 예측 수행
                # - x : 모델 입력 이미지
                # - yy : 연산 장치로 이동한 정답 라벨
                # - logits : 모델의 클래스별 예측 점수
                # 검증 손실과 실제 정답·예측 라벨 누적
                # - cross_entropy(...,reduction='sum') : 배치 손실 합계
                # - item() : Tensor 값을 Python 숫자로 변환
                # - y.tolist() : 실제 라벨을 Python 목록으로 변환
                # - logits.argmax(1) : 가장 높은 점수의 클래스 선택
                # - cpu().tolist() : 예측 라벨을 CPU의 Python 목록으로 변환
                x=x.to(device);yy=y.to(device);logits=model(x);total+=nn.functional.cross_entropy(logits,yy,reduction='sum').item();ys.extend(y.tolist());preds.extend(logits.argmax(1).cpu().tolist())
        # 현재 epoch의 검증 평가 결과 계산 및 기록
        # - epoch : 현재 학습 반복 번호
        # - valid_loss : 검증 데이터의 평균 손실
        # - accuracy : 전체 예측 중 정답 비율
        # - macro_f1 : 클래스별 F1 점수를 동일 비중으로 평균
        # - len(valid) : 전체 검증 이미지 수
        result={'epoch':epoch,'valid_loss':total/len(valid),'accuracy':accuracy_score(ys,preds),'macro_f1':f1_score(ys,preds,labels=[0,1,2],average='macro',zero_division=0)};history.append(result)
        # 현재 검증 손실이 이전 최저 손실보다 낮으면 최적 가중치 저장
        # - best : 지금까지 가장 낮은 검증 손실
        # - state : 최적 시점의 모델 가중치 복사본
        # - detach() : 계산 그래프에서 분리
        # - cpu() : CPU 메모리로 이동
        # - clone() : 독립적인 Tensor 복사본 생성
        if result['valid_loss']<best:best=result['valid_loss'];state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # 최적 모델 가중치와 학습 기록을 파일로 저장
    # - state : 최저 검증 손실을 기록한 모델 가중치
    # - class_to_idx : 클래스 이름과 정수 라벨 대응 정보
    # - weights : 사용한 공식 ImageNet 가중치 이름
    # - transform : 입력 이미지 전처리 정보
    # - strategy : 사용한 학습 전략
    # - torch.save() : 딕셔너리를 PyTorch 파일로 저장
    torch.save({'model':state,'class_to_idx':train.class_to_idx,'weights':'IMAGENET1K_V1','transform':str(transform),'strategy':a.strategy},out/(a.strategy+'.pt'))
    # epoch별 학습 기록과 모델 출처 정보를 JSON 파일로 저장
    # - history : epoch별 평가 결과
    # - source : 모델의 사전 학습 출처
    # - strategy : 사용한 학습 전략
    # - test_used=False : test 데이터를 사용하지 않았음을 기록
    (out/(a.strategy+'.json')).write_text(json.dumps({'history':history,'source':'ImageNet','strategy':a.strategy,'test_used':False},indent=2),encoding='utf-8')
    # 마지막 epoch의 평가 결과를 JSON 문자열 형태로 화면에 출력
    # - history[-1] : 마지막 epoch의 결과
    # - sort_keys=True : JSON 키를 정렬하여 출력
    print(json.dumps(history[-1],sort_keys=True))

# 이 파일을 직접 실행했을 때만 main() 함수 실행
# - __name__ : 현재 모듈의 실행 상태를 나타내는 변수
# - '__main__' : 해당 파일을 직접 실행한 경우의 값
if __name__=='__main__':main()
