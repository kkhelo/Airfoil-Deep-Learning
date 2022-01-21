from torch import nn

class vgg16(nn.Module):
  def __init__(self, in_channel : int ,out_classes : int):
    
    super().__init__()
    net = []

    # network construct
    # block 1
    net.append(nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(64))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(64))
    net.append(nn.ReLU())
    net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    

    # block 2
    
    net.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(128))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(128))
    net.append(nn.ReLU())
    net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    

    # block 3

    net.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(256))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(256))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(256))
    net.append(nn.ReLU())
    net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    

    # block 4 

    net.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(512))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(512))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(512))
    net.append(nn.ReLU())
    net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    

    #block 5

    net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(512))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(512))
    net.append(nn.ReLU())
    net.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1))
    net.append(nn.BatchNorm2d(512))
    net.append(nn.ReLU())
    net.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    

    # add net into class property 
    self.feature = nn.Sequential(*net)

    # classifier constuct
    classifier = []
    classifier.append(nn.Linear(in_features=512*1*1, out_features=4096))
    classifier.append(nn.ReLU())
    classifier.append(nn.Dropout()) 
    classifier.append(nn.Linear(in_features=4096, out_features=4096))
    classifier.append(nn.ReLU())
    classifier.append(nn.Dropout())
    classifier.append(nn.Linear(in_features=4096, out_features=out_classes))
    classifier.append(nn.Softmax(dim=1))

    self.classifier = nn.Sequential(*classifier)
      
  def forward(self, x):
    out = self.feature(x)
    out = out.view(x.size(0), -1)
    classify = self.classifier(out)
    return classify