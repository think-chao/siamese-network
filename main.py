import torchvision
from dataset import HRDataset
from snn import Model
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import os
from PIL import Image

training_data = '../datasets/'
batch_size = 4
train_batch_size = 4
epoch = 20
lr = 0.001


def train(tl):
    criterion = nn.BCELoss()
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # if torch.cuda.is_available():
    #     print('GPU ready')
    # model = model.cuda()
    for i in range(epoch):
        model.train()
        for iter, sample in enumerate(tl):
            result = 'Not Same'
            input1 = sample['Image1']
            input2 = sample['Image2']
            label = sample['Similar']
            optimizer.zero_grad()
            score = model(input1, input2)
            print(score, label)
            loss = criterion(score, label.float())
            print(loss.data)
            loss.backward()
            optimizer.step()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, "../checkpoints/new/" + str(i) + '.tar')


def main():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((105, 105)),
        torchvision.transforms.ToTensor()
    ])
    train_set = HRDataset(training_data, train_transform)
    train_dl = DataLoader(train_set, batch_size, shuffle=True)
    train(train_dl)
    # inference()


def inference():
    model = Model()
    checkpoint = torch.load('/home/ai/Desktop/project3/checkpoints/10.tar')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.cuda()
    img_shu1 = Image.open('../datasets/shumann/40.jpg').convert('L')
    img_shu2 = Image.open('../datasets/shumann/10.jpg').convert('L')
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((105, 105)),
        torchvision.transforms.ToTensor()
    ])
    input1 = torch.unsqueeze(test_transform(img_shu1),
                             dim=0).cuda()  # model is expecting an array of inputs, rather than a single input, so transform [img_content] into [[img_content]]

    input2 = torch.unsqueeze(test_transform(img_shu2), dim=0).cuda()
    score = model(input1, input2)
    print(score)


def api(face):
    face = Image.fromarray(face).convert('L')
    max = 0
    name = ''
    model = Model()
    checkpoint = torch.load('/home/ai/Desktop/project3/checkpoints/new/18.tar')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.cuda()
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((105, 105)),
        torchvision.transforms.ToTensor()
    ])

    test = HRDataset(training_data, test_transform)
    input1 = torch.unsqueeze(test_transform(face), dim=0).cuda()
    for i in test.all_image_name:
        img = Image.open(i[0]).convert('L')
        input2 = torch.unsqueeze(test_transform(img), dim=0).cuda()
        score = model(input1, input2)
        # print(i.split('/')[2], score.data)
        if max < model(input1, input2):
            max = model(input1, input2)
            name = i[0].split('/')[2]
    # print(max)
    return name

# main()
