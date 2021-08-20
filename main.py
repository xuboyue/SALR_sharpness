import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import math

n_epochs = 40
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.01
LR_SGA = learning_rate / 10
momentum = 0 # 0.9 for imagenet and finetuning
log_interval = 10
WD = 0.005

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# CIFAR

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=0)



class normalized_SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(normalized_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(normalized_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            total_norm = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # param_norm = p.grad.data.norm(2)
                # total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                d_p = d_p / total_norm

                p.data.add_(-group['lr'], d_p)

        return loss




def forward(data, target, model, criterion, epoch=0, training=True, optimizer=None):

    losses = 0 # AverageMeter()
    grad_vec = None

    if training:
      optimizer_1 = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=WD)
      optimizer_1.zero_grad()  # only zerout at the beginning

    inputs = data
    target = target
    output = model(inputs)
    loss = criterion(output, target)
    losses += loss.item()
    loss.backward()
    # losses += loss.item()
        # optimizer.step() # no step in this case

        # output = model(inputs)
        # loss = criterion(output, target)
        # losses += loss.item()
        # loss.backward()

    # reshape and averaging gradients
    if training:
      for p in model.parameters():
        # p.grad.data.div_(len(data_loader))
        if grad_vec is None:
            grad_vec = p.grad.data.view(-1)
        else:
            grad_vec = torch.cat((grad_vec, p.grad.data.view(-1)))

    #logging.info('{phase} - \t'
    #             'Loss {loss.avg:.4f}\t'
    #             'Prec@1 {top1.avg:.3f}\t'
    #             'Prec@5 {top5.avg:.3f}'.format(
    #              phase='TRAINING' if training else 'EVALUATING',
    #              loss=losses, top1=top1, top5=top5))

    return {'loss': losses}, grad_vec


class projected_SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(projected_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(projected_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            total_norm = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                d_p = np.sign(d_p)
                p.data.add_(-group['lr'], d_p)
                print(d_p)
        return loss



def get_minus_cross_entropy(data, target, model, criterion, training=False):

  model.eval()
  result, grads = forward(data, target, model, criterion, 0,
                 training=training, optimizer=None)
  return (-result['loss'], None if grads is None else grads.cpu().numpy().astype(np.float64))

learning_rate_alpha = 0.001
epss = 5e-4

def get_sharpness_ascent(data, target, model_ori, criterion, f, manifolds=0):

  model = model_ori
  f_x0, _ = get_minus_cross_entropy(data, target, model, criterion)
  f_x0 = -f_x0
  optimizer_sga = normalized_SGD(model.parameters(), lr=LR_SGA)
  inputs = data
  labels = target

  optimizer_PGA = projected_SGD(network.parameters(), lr=learning_rate_alpha, momentum=momentum)  # SGD optimizer
  temp = optimizer.param_groups[0]['params']  # [0]
  for j in range(5):

      i = 0
      optimizer_PGA.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      (loss).backward()
      # with torch.no_grad():
      for p in network.parameters():
          p_1 = p + optimizer_PGA.param_groups[0]['lr'] * p.grad.data
          optimizer_PGA.param_groups[0]['params'][i] = temp[i] + (p_1 - temp[i]).clamp(min=-epss, max=epss)
          i += 1
      i = 0
      with torch.no_grad():
          for name, param in network.named_parameters():
              param.copy_(optimizer_PGA.param_groups[0]['params'][i])
              i += 1

  optimizer.zero_grad()
  output = network(data)
  loss_new = criterion(output, target)
  # print("f_x is:", loss_new.item(), "f_x0 is: ", f_x0)
  # print("sharp os:", loss_new.item() - f_x0)


  f_x = loss_new.item()
  # f_x = -f_x
  sharpness = (f_x - f_x0) #/(1+f_x0)*100

  return sharpness


def get_sharpness_descent(data, target, model_ori, criterion, f, manifolds=0):

  model = model_ori
  f_x0, _ = get_minus_cross_entropy(data, target, model, criterion)
  f_x0 = -f_x0
  optimizer_sga = normalized_SGD(model.parameters(), lr=LR_SGA)
  inputs = data
  labels = target

  optimizer_PGA = projected_SGD(network.parameters(), lr=learning_rate_alpha, momentum=momentum)  # SGD optimizer
  temp = optimizer.param_groups[0]['params']  # [0]

  for i in range(5):
      i = 0
      optimizer_PGA.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      (loss).backward()
      # with torch.no_grad():
      for p in network.parameters():
          p_1 = p - optimizer_PGA.param_groups[0]['lr'] * p.grad.data
          optimizer_PGA.param_groups[0]['params'][i] = temp[i] + (p_1 - temp[i]).clamp(min=-epss, max=epss)
          i += 1
      i = 0
      with torch.no_grad():
          for name, param in network.named_parameters():
              param.copy_(optimizer_PGA.param_groups[0]['params'][i])
              i += 1
  optimizer.zero_grad()
  output = network(data)
  loss_new = criterion(output, target)
  # print("f_x is:", loss_new.item(), "f_x0 is: ", f_x0)
  # print("sharp os:", loss_new.item() - f_x0)

  f_x = loss_new.item()
  # f_x = -f_x
  sharpness = (f_x0 - f_x) #/(1+f_x0)*100

  return sharpness

#################
### replace this network by other network as listed in GitHub
#################

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)






network = DenseNet121()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


criterion = getattr(network, 'criterion', nn.CrossEntropyLoss)()
criterion.type('torch.FloatTensor')
network.type('torch.FloatTensor')

sharp_array = []
LR_array = []
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if epoch <= -1 and batch_idx % 2 == 0:
        sharp_a = get_sharpness_ascent(data, target, network, criterion, loss.item(), manifolds=0)
        sharp_d = get_sharpness_descent(data, target, network, criterion, loss.item(), manifolds=0)
        # print("sharp_ascent: ", sharp_a)
        # print("sharp_descent: ", sharp_d)
        sharp = sharp_a + sharp_d
        sharp_array.append(sharp)
        LR_array.append(optimizer.param_groups[0]['lr'])
        # print([batch_idx, sharp])

    if epoch >= 0 and batch_idx % 2 == 0:


        sharp_a = get_sharpness_ascent(data, target, network, criterion, loss.item(), manifolds=0)
        sharp_d = get_sharpness_descent(data, target, network, criterion, loss.item(), manifolds=0)
        sharp = sharp_a + sharp_d
        sharp_array.append(sharp)
        sharp_array_1 = np.array(sharp_array)
        # sharp_array_1 = reject_outliers(sharp_array_1, 3)
        # sharp = sharp / sharp_array.max()
        if sharp > np.percentile(sharp_array_1, 51) or sharp < np.percentile(sharp_array_1, 49):
            optimizer.param_groups[0]['lr'] = 1 * learning_rate * sharp / np.percentile(sharp_array_1, 50) # ( (np.percentile(sharp_array_1, 85) ) ) # (sharp_array.mean() + sharp_array.std()) # sharp_array.max()
        else:
            optimizer.param_groups[0]['lr'] = 1 * learning_rate
        if optimizer.param_groups[0]['lr'] > learning_rate * 5: optimizer.param_groups[0]['lr'] = learning_rate * 5
        # print("Base is " , learning_rate, "; new LR is ", optimizer.param_groups[0]['lr'], "; sharp is ", sharp)
        LR_array.append(optimizer.param_groups[0]['lr'])


    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      # torch.save(network.state_dict(), '/results/model.pth')
      # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

    # if epoch> 1 and epoch % 30 == 0:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5
    #     # print('LR is decreased')

loss_function = nn.CrossEntropyLoss()

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += loss_function(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  test_acc_array.append(100 - 100. * correct / len(test_loader.dataset))

  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


test_acc_array = []
test()

for epoch in range(1, n_epochs + 2):
  train(epoch)
  test()

