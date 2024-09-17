import torch
from models.autoencoder import Autoencoder
from utils.dataset import get_data_loaders
from models.clustering import extract_features

if __name__ == '__main__':
    train_loader = get_data_loaders('data/images')
    model = Autoencoder()
    model.load_state_dict(torch.load('autoencoder.pth'))
    
    features = extract_features(model, train_loader)
    
    # PCA를 사용하여 차원 축소 및 클러스터링 결과 출력
    # 이미 클러스터링을 위한 코드는 `models/clustering.py`에 포함되어 있으므로 별도 작성 필요 없음
