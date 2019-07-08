import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, n_class=1):
        super(Model, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv2d(1, 64,10),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 7),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.MaxPool2d(2),


            nn.Conv2d(128, 256, 4),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.n_class = n_class

        self.cls = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, n_class),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        feature1 = self.cov(input1)
        feature2 = self.cov(input2)

        # print(feature1.size())
        a = feature1.view(batch_size, 256 * 6 * 6)
        b = feature2.view(batch_size, 256 * 6 * 6)
        difference = torch.abs(torch.sub(a,b))
        simi_score = self.cls(difference)
        return simi_score

if __name__ == '__main__':
    m = Model()
    x1 = torch.ones(1,1, 105, 105)
    x2 = torch.ones(1,1, 105, 105)
    m(x1,x2)
