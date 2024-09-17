import sys
import os
import torch
import torch.optim as optim

# 프로젝트 루트 디렉토리 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder import Autoencoder
from utils.dataset import get_data_loaders

def train_autoencoder(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

if __name__ == '__main__':
    train_loader, _ = get_data_loaders('data/images', 'data/images')  # 테스트용으로 동일한 디렉토리 사용
    model = Autoencoder()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 데이터 확인
    for inputs in train_loader:
        print(f'Batch shape: {inputs.shape}')
        break  # 첫 배치만 확인

    train_autoencoder(model, train_loader, criterion, optimizer, epochs=10)
    torch.save(model.state_dict(), 'autoencoder.pth')
