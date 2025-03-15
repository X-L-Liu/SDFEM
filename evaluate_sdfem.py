from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from model_class import *

import argparse


def eval_(attack):
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Test-Progress ', ncols=100) as pbar:
        for _, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image = attack(image, label)
            logit = model(image)
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.acc_rate


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'miniimagenet'])   
    parser.add_argument('--batch_size', type=int, default=512)

    parser.add_argument('--model_name', type=str, default='SDFEMWideResNet28x10')
    parser.add_argument('--model_load_path', type=str, default=r'')

    parser.add_argument('--sub_model_name', type=str, default='')
    parser.add_argument('--sub_model_load_path', type=str, default=r'')

    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()

    assert os.path.exists(config.model_load_path)
    assert os.path.exists(config.sub_model_load_path)
    assert config.dataset_name in ['cifar10', 'miniimagenet']

    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)

    device = torch.device(f'cuda:{config.device}')

    if config.dataset_name == 'cifar10':
        num_classes = 10
        svd_channels = 96
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_cifar10_test)
    else:
        num_classes = 100
        svd_channels = 192
        testSet = MiniImageNet(root=config.data_path, train=False, pixel='64', transform=transform_miniimagenet_test)

    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model = globals()[config.model_name](num_classes, svd_channels)
    model.load_state_dict(torch.load(config.model_load_path, map_location=device))
    model.to(device)
    model.eval()

    sub_model = globals()[config.sub_model_name](num_classes)
    sub_model.load_state_dict(torch.load(config.sub_model_load_path, map_location=device))
    sub_model.to(device)
    sub_model.eval()

    attacks = {
        'Clean': torchattacks.VANILA(sub_model),
        'FGSM': torchattacks.FGSM(sub_model),
        'BIM': torchattacks.BIM(sub_model, steps=100),
        'DIM': torchattacks.DIFGSM(sub_model, steps=100),
        'VMIM': torchattacks.VMIFGSM(sub_model, steps=100),
        'VNIM': torchattacks.VNIFGSM(sub_model, steps=100)
    }

    print('>' * 100)
    print(f'{config.model_name}  Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')

    for att_name in attacks.keys():
        torch.cuda.synchronize()
        acc = eval_(attacks[att_name])
        torch.cuda.synchronize()
        print(f'{att_name}: {acc * 100:.2f}%')
