import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TRAIN_PATH = './test'
TEST_PATH = './train'

transform = transforms.Compose([
    transforms.ToTensor(), # make them tensors
    transforms.Resize((256, 256)), # make all images the same size
])

trainset = ImageFolder(TRAIN_PATH, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = ImageFolder(TEST_PATH, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = trainset.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DarknetUpdated(nn.Module):
  def __init__(self):
    super(DarknetUpdated, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)

    self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(32)

    self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(64)

    self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    self.bn4 = nn.BatchNorm2d(128)

    self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
    self.bn5 = nn.BatchNorm2d(256)

    self.conv6 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
    self.bn6 = nn.BatchNorm2d(512)

    self.conv7 = nn.Conv2d(512, 1024, 3, padding=1, bias=False)
    self.bn7 = nn.BatchNorm2d(1024)

    self.fc4 = nn.Linear(1024, 4) # 4 categories

  def forward(self, x):
    # Input 256x256x3
    x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2) # 128x128x16
    x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2) # 64x64x32
    x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2) # 32x32x64
    x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=2, stride=2) # 16x16x128
    x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), kernel_size=2, stride=2) # 8x8x256
    x = F.max_pool2d(F.relu(self.bn6(self.conv6(x))), kernel_size=2, stride=2) # 4x4x512
    x = F.max_pool2d(F.relu(self.bn7(self.conv7(x))), kernel_size=2, stride=2) # 2x2x1024
    x = F.adaptive_avg_pool2d(x, 1) # 1x1x1024
    x = torch.flatten(x, 1)
    x = self.fc4(x)
    return x

# https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing#scrollTo=9tH2B0mlQywk
def train(net, dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005, 
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
  net.to(device)
  net.train()
  losses = []
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

  # Load previous training state
  if state:
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['epoch']
    losses = state['losses']

  # Fast forward lr schedule through already trained epochs
  for epoch in range(start_epoch):
    if epoch in schedule:
      print ("Learning rate: %f"% schedule[epoch])
      for g in optimizer.param_groups:
        g['lr'] = schedule[epoch]

  for epoch in range(start_epoch, epochs):
    sum_loss = 0.0

    # Update learning rate when scheduled
    if epoch in schedule:
      print ("Learning rate: %f"% schedule[epoch])
      for g in optimizer.param_groups:
        g['lr'] = schedule[epoch]

    for i, batch in enumerate(dataloader, 0):
      inputs, labels = batch[0].to(device), batch[1].to(device)

      optimizer.zero_grad()

      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()  # autograd magic, computes all the partial derivatives
      optimizer.step() # takes a step in gradient direction

      losses.append(loss.item())
      sum_loss += loss.item()
        
      if i % print_every == print_every-1:
        if verbose:
          print('[%d, %5d] loss: %.3f' % (epoch, i + 1, sum_loss / print_every))
        sum_loss = 0.0
    if checkpoint_path:
      state = {'epoch': epoch+1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
      torch.save(state, checkpoint_path + 'checkpoint-%d.pkl'%(epoch+1))
  return losses

def accuracy(net, dataloader):
  net.to(device)
  net.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for batch in dataloader:
      images, labels = batch[0].to(device), batch[1].to(device)
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return correct/total

def smooth(x, size):
  return np.convolve(x, np.ones(size)/size, mode='valid')

if __name__ == '__main__':
  net = DarknetUpdated()
  losses = train(net, trainloader, epochs=25, schedule={0:0.01, 5:0.001, 15:0.0001})
  print("Testing accuracy: %f" % accuracy(net, testloader))

  # Used to display the loss curve
  # plt.plot(smooth(losses,50))
  # plt.show()
