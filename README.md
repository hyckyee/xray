# X-ray 골절 감지 프로젝트

이 프로젝트는 X-ray 이미지를 사용하여 골절을 감지하는 딥러닝 모델을 개발하는 것입니다. 오토인코더라는 모델을 활용하여 X-ray 이미지를 분석하고, 골절 여부를 판단합니다. 이 문서에서는 프로젝트의 구성, 데이터 준비 방법, 모델의 원리, 학습 및 평가 방법 등을 상세히 설명합니다.

## 프로젝트 구조

이 프로젝트는 다음과 같은 폴더와 파일로 구성되어 있습니다:

```
xray/
│
├── data/
│   ├── images/            # X-ray 이미지 데이터가 저장되는 폴더
│
├── models/
│   ├── __init__.py
│   ├── autoencoder.py     # 오토인코더 모델 정의
│
├── scripts/
│   ├── train_autoencoder.py  # 오토인코더 모델 학습 스크립트
│   ├── evaluate_autoencoder.py  # 오토인코더 모델 평가 스크립트
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py         # 데이터 로더 및 전처리 함수
│
└── README.md              # 이 문서
```

## 데이터 준비

### 데이터셋 다운로드

1. **X-ray 이미지 다운로드**: X-ray 이미지 파일을 다운로드하여 `data/images` 폴더에 저장합니다. 데이터셋은 X-ray 이미지로 구성되어야 합니다.

2. **파일 구조**: 데이터는 `data/images` 폴더에 저장되어야 합니다. 폴더에는 이미지 파일만 있어야 하며, 하위 디렉토리 없이 직접 저장되어야 합니다. 예시 구조는 다음과 같습니다:

   ```
   data/
   ├── images/
       ├── image1.png
       ├── image2.png
       └── ...
   ```

### 데이터 전처리

데이터는 모델 학습 전에 전처리 과정이 필요합니다. 전처리 과정에는 다음이 포함됩니다:

- **크기 조정(Resizing)**: 이미지를 일정한 크기로 조정합니다. 예를 들어, 모든 이미지를 1024x1024 픽셀로 조정합니다.

- **정규화(Normalization)**: 이미지의 픽셀 값을 정규화하여 모델이 더 잘 학습할 수 있도록 합니다. 일반적으로 픽셀 값을 0과 1 사이로 조정합니다.

- **텐서 변환(Tensor Transformation)**: 이미지를 PyTorch 텐서로 변환하여 모델에 입력할 수 있도록 준비합니다. 텐서는 다차원 배열로, 컴퓨터에서 효율적으로 계산할 수 있습니다.

## 모델 원리

### 오토인코더(Autoencoder)

오토인코더는 입력 데이터를 압축하고 다시 복원하는 신경망 모델입니다. 오토인코더의 주요 구성 요소는 다음과 같습니다:

- **인코더(Encoder)**: 입력 이미지를 잠재 공간(latent space)으로 압축합니다. 인코더는 여러 개의 합성곱(convolutional) 레이어로 구성되어 있으며, 이미지의 중요한 특징을 추출합니다.

- **디코더(Decoder)**: 잠재 공간의 벡터를 원래 이미지로 복원합니다. 디코더는 역합성곱(transposed convolution) 레이어로 구성되어 있으며, 인코더가 압축한 정보를 바탕으로 이미지를 재구성합니다.

- **손실 함수(Loss Function)**: 모델의 출력 이미지와 실제 이미지 간의 차이를 측정하는 함수입니다. 평균 제곱 오차(MSE) 등을 사용하여 차이를 최소화하는 방향으로 모델을 학습시킵니다.

### PyTorch 및 관련 패키지

- **PyTorch**: 딥러닝 모델을 구축하고 학습시키기 위한 오픈 소스 라이브러리입니다. 텐서 연산, 자동 미분, 신경망 구축 등의 기능을 제공합니다. PyTorch의 주요 구성 요소는 `torch` 모듈로, 텐서 연산과 신경망 모듈을 포함합니다.

  - **텐서(Tensor)**: PyTorch의 기본 데이터 구조로, 다차원 배열을 지원합니다. 텐서는 GPU에서 계산을 수행할 수 있어 딥러닝 모델의 학습과 추론을 가속화합니다.

  - **자동 미분(Autograd)**: PyTorch는 자동으로 그래디언트를 계산하여 역전파 알고리즘을 사용해 모델을 학습합니다. `torch.autograd` 모듈을 사용하여 미분 연산을 자동으로 처리합니다.

  - **신경망 모듈(nn.Module)**: PyTorch의 신경망 모듈은 신경망의 각 레이어를 정의하고, 이를 연결하여 모델을 구축합니다. `torch.nn` 모듈을 사용하여 다양한 신경망 구성 요소를 구현할 수 있습니다.

- **torchvision**: PyTorch의 공식 라이브러리로, 이미지 데이터셋을 쉽게 다루기 위한 다양한 기능을 제공합니다. `datasets` 모듈을 통해 미리 정의된 데이터셋을 로드할 수 있으며, `transforms` 모듈을 통해 이미지 전처리를 수행할 수 있습니다.

  - **데이터셋 로딩**: `datasets.ImageFolder` 클래스를 사용하여 이미지 데이터셋을 로드합니다. 이미지 폴더 구조를 기반으로 클래스 레이블을 자동으로 할당합니다.

  - **전처리(Transforms)**: `transforms` 모듈을 사용하여 이미지 크기 조정, 정규화, 텐서 변환 등의 전처리 작업을 수행합니다.

- **PIL (Python Imaging Library)**: Python Imaging Library(PIL)는 이미지를 처리하고 조작하는 기능을 제공합니다. `PIL.Image` 모듈을 통해 이미지의 열기, 저장, 변환 등의 작업을 수행할 수 있습니다.

## 학습 방법

1. **학습 스크립트 실행**:

   모델을 학습시키기 위해 `scripts/train_autoencoder.py` 스크립트를 실행합니다. 이 스크립트는 오토인코더 모델을 학습시키며, 학습된 모델의 가중치를 `autoencoder.pth` 파일에 저장합니다.

   ```bash
   python scripts/train_autoencoder.py
   ```

2. **학습 과정**:

   - **데이터 로딩**: 스크립트는 `data/images` 폴더에 저장된 X-ray 이미지를 로드합니다. 데이터 로더는 배치 크기, 샘플링 방법, 데이터 전처리 등을 설정합니다.

   - **모델 정의**: `models/autoencoder.py` 파일에서 정의된 오토인코더 모델이 사용됩니다. 모델은 인코더와 디코더로 구성되며, 입력 이미지를 압축하고 복원합니다.

   - **학습 파라미터 설정**: 학습률, 배치 크기, 에폭 수 등은 학습 성능에 영향을 미칩니다. `train_autoencoder.py` 스크립트에서 이 값들을 설정할 수 있으며, 실험을 통해 최적의 값을 찾는 것이 좋습니다.

   - **손실 함수 및 옵티마이저**: 손실 함수는 평균 제곱 오차(MSE)이며, 옵티마이저는 Adam 옵티마이저를 사용합니다. 손실 함수는 모델의 출력과 실제 이미지 간의 차이를 측정하며, 옵티마이저는 모델의 파라미터를 업데이트하는 데 사용됩니다.

   - **학습 과정 모니터링**: 학습 과정 동안 손실 값이 출력되며, 손실 값이 감소하는지 확인하여 모델의 학습 상태를 모니터링합니다.

## 평가 방법

1. **평가 스크립트 실행**:

   학습된 모델을 평가하기 위해 `scripts/evaluate_autoencoder.py` 스크립트를 실행합니다. 이 스크립트는 모델을 평가하고, 평가 결과를 출력합니다.

   ```bash
   python scripts/evaluate_autoencoder.py
   ```

2. **평가 과정**:

   - **모델 로드**: 평가 스크립트는 `autoencoder.pth` 파일에서 저장된 모델의 가중치를 로드합니다.

   - **평가 지표**: 모델의 성능을 측정하기 위해 다양한 평가 지표를 사용할 수 있습니다. 일반적으로 오토인코더에서는 평균 제곱 오차(MSE)와 같은 지표를 사용하여 입력 이미지와 복원된 이미지 간의 차이를 평가합니다.

## 모델 파일

- **저장된 모델**: 학습이 완료된 모델은 `autoencoder.pth` 파일로 저장됩니다. 이 파일은 학습된 모델의 가중치를 포함하며, 이후 평가 및 추가 학습에 사용할 수 있습니다. 모델 파일은 학습 과정 중에 지정된 경로에 자동으로 저장됩니다.

## 참고 사항

- **GPU 사용**: GPU가 설정되어 있어야 합니다. GPU를 사용하면 학습 속도가 빨라지며, 모델 학습과 평가의 효율성이 높아집니다.

- **필요한 패키지**: 이 프로젝트를 실행하기 위해 필요한 패키지는 PyTorch, torchvision, PIL 등이 있습니다. 패키지 설치는 `requirements.txt` 파일을 통해 수행할 수 있습니다.

- **추가 정보**: 코드의 세부 사항이나 추가적인 도움이 필요하면, 코드 내 주석을 참조하거나 관련 문서 및 자료를 검토하십시오.

