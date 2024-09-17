import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) if img_name.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_data_loaders(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(),  # X-ray 이미지는 단일 채널일 수 있음
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize based on X-ray images
    ])

    train_dataset = CustomImageDataset(image_dir=train_dir, transform=transform)
    test_dataset = CustomImageDataset(image_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
