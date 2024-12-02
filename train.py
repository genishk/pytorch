import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F

from models.cnn_model import CNN
from utils.data_loader import get_data_loaders
from configs.config import Config

def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % Config.log_interval == 0:
            writer.add_scalar('Loss/train', loss.item(), 
                            epoch * len(train_loader) + batch_idx)
            pbar.set_postfix({'Loss': loss.item()})

def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')

def main():
    # 재현성을 위한 시드 설정
    torch.manual_seed(Config.seed)
    random.seed(Config.seed)
    np.random.seed(Config.seed)

    # CUDA 사용 가능 여부 확인
    use_cuda = Config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 데이터 로더 생성
    train_loader, test_loader = get_data_loaders(Config.batch_size)

    # 모델 초기화
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate,
                         momentum=Config.momentum)

    # TensorBoard 설정
    writer = SummaryWriter()

    # 학습 시작
    for epoch in range(1, Config.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, writer)
        test(model, device, test_loader, epoch, writer)

    # 모델 저장
    if Config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    writer.close()

if __name__ == '__main__':
    main() 