# -*- coding: utf-8 -*-
import torch
import torchvision

from dataset import HandDataset as HandDataset
from model import MaskRcnn as MaskRcnn
import utils


def get_transform(train):
    t = []
    # if train:
    #     t.append(torchvision.transforms.transforms.RandomHorizontalFlip(0.5))
    t.append(torchvision.transforms.transforms.ToTensor())

    return torchvision.transforms.transforms.Compose(t)


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("load dataset")
    num_classes = 2
    data = HandDataset(args.data_path, get_transform(train=True))

    indices = torch.randperm(len(data)).tolist()
    test_cnt = int(len(data) / 10)
    dataset = torch.utils.data.Subset(data, indices[:-test_cnt])
    dataset_test = torch.utils.data.Subset(dataset, indices[-test_cnt:])

    # 定义训练和验证数据加载器
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=lambda x: tuple(zip(*x)))

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                   collate_fn=lambda x: tuple(zip(*x)))

    print("load model")
    model = MaskRcnn.get_pretrained_resnet50_model(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    print("begin train")
    num_epochs = 10

    for epoch in range(num_epochs):
        utils.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    # evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='', help='dataset')
    parser.add_argument('-b', '--batch-size', default=2, type=int, help='batch size')
    parser.add_argument('--epochs', default=13, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    args = parser.parse_args()

    print(args)
    if args.output_dir:
        utils.mkdir(args.output_dir)
    main(args)
