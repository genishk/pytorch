import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from models.cnn_model import CNN
from utils.data_loader import get_data_loaders
from configs.config import Config

def train(model, device, train_loader, optimizer, scheduler, epoch, writer, scaler):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        try:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)  # 더 효율적인 방법
            
            # Mixed precision training
            with autocast():
                output = model(data)
                loss = F.nll_loss(output, target)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

            if batch_idx % Config.log_interval == 0:
                writer.add_scalar('Loss/train', loss.item(), 
                                epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],
                                epoch * len(train_loader) + batch_idx)
                pbar.set_postfix({'Loss': loss.item(), 'LR': optimizer.param_groups[0]['lr']})
                
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {str(e)}")
            continue
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            try:
                data, target = data.to(device), target.to(device)
                with autocast():
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            except Exception as e:
                print(f"Error in testing batch: {str(e)}")
                continue

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_accuracy': best_accuracy
    }
    torch.save(checkpoint, filename)

def main():
    # 재현성을 위한 시드 설정
    torch.manual_seed(Config.seed)
    random.seed(Config.seed)
    np.random.seed(Config.seed)

    # CUDA 설정
    use_cuda = Config.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if use_cuda:
        torch.backends.cudnn.benchmark = True  # 성능 향상을 위한 설정

    # 데이터 로더 생성
    train_loader, test_loader = get_data_loaders(Config.batch_size)

    # 모델 초기화
    model = CNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate,
                          weight_decay=1e-4)  # AdamW로 변경
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                verbose=True)
    scaler = GradScaler()

    # TensorBoard 설정
    writer = SummaryWriter(log_dir=os.path.join('runs', Config.experiment_name))
    
    best_accuracy = 0
    best_model_path = os.path.join(Config.checkpoint_dir, 'best_model.pt')
    
    try:
        # 학습 시작
        for epoch in range(1, Config.epochs + 1):
            train_loss = train(model, device, train_loader, optimizer, scheduler,
                             epoch, writer, scaler)
            test_loss, accuracy = test(model, device, test_loader, epoch, writer)
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy,
                              best_model_path)
                
    except KeyboardInterrupt:
        print("Training interrupted. Saving current state...")
        save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy,
                       'interrupted_checkpoint.pt')
    
    finally:
        writer.close()

if __name__ == '__main__':
    main() 