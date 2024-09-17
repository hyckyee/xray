import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from models.autoencoder import Autoencoder
from utils.dataset import get_data_loaders

def extract_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model.encoder(inputs).view(inputs.size(0), -1)
            features.append(outputs.numpy())
    return np.concatenate(features)

if __name__ == '__main__':
    train_loader = get_data_loaders('data/images')
    model = Autoencoder()
    model.load_state_dict(torch.load('autoencoder.pth'))
    
    features = extract_features(model, train_loader)
    
    # PCA를 사용하여 차원 축소
    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(features)
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_features)
    
    # 클러스터링 결과 출력
    print("Cluster centers:", kmeans.cluster_centers_)
    print("Labels:", kmeans.labels_)
