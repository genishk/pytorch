# PyTorch MNIST Classifier

간단한 CNN을 사용한 MNIST 손글씨 숫자 분류기입니다.

## 설치 방법

가상환경 생성 및 활성화:

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

필요한 패키지 설치:

```
pip install -r requirements.txt
```

## 실행 방법

```
python train.py
```

## 프로젝트 구조

```
├── models/             # 모델 구조 정의
│   └── cnn_model.py    # CNN 모델 클래스
├── utils/              # 유틸리티 함수
│   └── data_loader.py  # 데이터 로딩 함수
├── configs/            # 설정 파일
│   └── config.py       # 학습 설정
├── data/               # MNIST 데이터셋 저장 (자동 생성)
├── runs/               # TensorBoard 로그 (자동 생성)
├── train.py            # 학습 스크립트
└── README.md           # 프로젝트 설명
```

## 결과 확인

학습 진행 상황은 TensorBoard를 통해 확인할 수 있습니다:

```
tensorboard --logdir=runs
```

## 주요 기능

- MNIST 데이터셋 자동 다운로드
- CNN 모델을 사용한 이미지 분류
- TensorBoard를 통한 학습 과정 시각화
- 학습된 모델 저장

## 모델 구조

- 2개의 컨볼루션 레이어
- Dropout을 통한 과적합 방지
- 2개의 완전연결 레이어
- Softmax 활성화 함수

## 학습 설정

- Batch Size: 64
- Epochs: 10
- Learning Rate: 0.01
- Optimizer: SGD with momentum
