from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from model_class import *

import argparse
import time


def train_epoch():
    model.train()
    top1 = AverageMeter()
    with tqdm(total=len(train_loader), desc='Train-Progress', ncols=100) as pbar:
        for k, (image, label) in enumerate(train_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image[::2] = pgd(image[::2], label[::2])
            if config.use_amp:
                with torch.cuda.amp.autocast():
                    logit = model(image)
                    loss = F.cross_entropy(logit, label)
                loss = scaler.scale(loss)
            else:
                logit = model(image)
                loss = F.cross_entropy(logit, label)
            optimizer.zero_grad()
            loss.backward()
            if config.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def test_epoch():
    model.eval()
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Test-Progress ', ncols=100) as pbar:
        for k, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image[::2] = pgd(image[::2], label[::2])
            logit = model(image)
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def main(Reload):
    global model_load_path
    for epoch in range(config.epochs):
        start = time.time()
        train_acc = train_epoch()
        test_acc = test_epoch()
        scheduler.step()
        if test_acc > config.best_acc:
            model_load_path = os.path.join(model_save_path, f'{config.model_name}_{test_acc*100:.2f}.pt')
            torch.save(model.state_dict(), model_load_path)
            if os.path.exists(model_load_path.replace(f'{test_acc*100:.2f}', f'{config.best_acc*100:.2f}')):
                os.remove(model_load_path.replace(f'{test_acc*100:.2f}', f'{config.best_acc*100:.2f}'))
            config.best_acc = test_acc
        print(f'Model: {config.model_name}  '
              f'Reload: {Reload + 1}/{config.reload}  Epoch: {epoch + 1}/{config.epochs}  '
              f'Train-Top1: {train_acc * 100:.2f}%  Test-Top1: {test_acc * 100:.2f}%  '
              f'Best-Top1: {config.best_acc * 100:.2f}%  Time: {time.time() - start:.0f}s')


def load_model():
    classifier = globals()[config.model_name](num_classes, svd_channels)
    if model_load_path != '':
        classifier.load_state_dict(torch.load(model_load_path, map_location=device))
    classifier.to(device)
    classifier.eval()

    return classifier


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'miniimagenet'])
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--milestones', type=tuple, default=(50, 75, 100))
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--reload', type=int, default=5)

    parser.add_argument('--model_name', type=str, default='SDFEMWideResNet28x10')
    parser.add_argument('--model_load_path', type=str, default=r'')
    parser.add_argument('--model_save_path', type=str, default=r'adv_model')

    parser.add_argument('--pre_model_name', type=str, default='ResNet18')
    parser.add_argument('--pre_model_load_path', type=str, default=r'')

    parser.add_argument('--best_acc', type=float, default=0)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()

    assert os.path.exists(config.pre_model_load_path)
    assert config.dataset_name in ['cifar10', 'miniimagenet']

    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)

    device = torch.device(f'cuda:{config.device}')
    model_load_path = config.model_load_path

    if config.dataset_name == 'cifar10':
        num_classes = 10
        svd_channels = 96
        trainSet = datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=transform_cifar10_train)
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_cifar10_test)
        model_save_path = os.path.join(config.model_save_path, 'cifar10')
    else:
        num_classes = 100
        svd_channels = 192
        trainSet = MiniImageNet(root=config.data_path, train=True, pixel='64', transform=transform_miniimagenet_train)
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='64', transform=transform_miniimagenet_test)
        model_save_path = os.path.join(config.model_save_path, 'miniimagenet')

    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    sub_model = globals()[config.pre_model_name](num_classes)
    sub_model.load_state_dict(torch.load(config.pre_model_load_path, map_location=device))
    sub_model.to(device)
    sub_model.eval()
    pgd = torchattacks.PGD(sub_model, steps=20)

    for reload in range(config.reload):
        model = load_model()
        print('>' * 100)
        print(f'Model: {config.model_name}  '
              f'Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')
        optimizer = optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
        scaler = torch.cuda.amp.GradScaler()
        main(reload)
