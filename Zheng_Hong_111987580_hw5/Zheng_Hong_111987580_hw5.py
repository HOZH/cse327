# import packages here
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import time

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# my imports
from scipy.ndimage.interpolation import rotate
import torch.optim as optim
from time import time
import sys
from torchvision import models
from sklearn.svm import LinearSVC

# not import the google.colab related stuff in this python file

# ==========================================
#    Load Training Data and Testing Data
# ==========================================
class_names = [name[13:] for name in glob.glob('./data/train/*')]
class_names = dict(zip(range(len(class_names)), class_names))
print("class_names: %s " % class_names)
n_train_samples = 150
n_test_samples = 50


def img_norm(img):
    #
    # Write your code here
    # normalize img pixels to [-1, 1]
    #

    #   Min-Max Normalization
    #    x' = (x - X_min) / (X_max - X_min)

    pixels = img.astype('float64')
    min, max = pixels.min(), pixels.max()
    # print(pixels-min,max-min)
    return (2. * (pixels - min) / (max - min)) - 1


def load_dataset(path, img_size, num_per_class=-1, batch_num=1, shuffle=False, augment=False, is_color=False,
                 rotate_90=False, zero_centered=False):
    data = []
    labels = []

    if is_color:
        channel_num = 3
    else:
        channel_num = 1

    # read images and resizing
    for id, class_name in class_names.items():
        print("Loading images from class: %s" % id)
        img_path_class = glob.glob(path + class_name + '/*.jpg')
        if num_per_class > 0:
            img_path_class = img_path_class[:num_per_class]
        labels.extend([id] * len(img_path_class))
        for filename in img_path_class:
            if is_color:
                img = cv2.imread(filename)
            else:
                img = cv2.imread(filename, 0)

            # resize the image
            img = cv2.resize(img, img_size, cv2.INTER_LINEAR)

            if is_color:
                img = np.transpose(img, [2, 0, 1])

            # norm pixel values to [-1, 1]
            # Write your Data Normalization code here
            data.append(img_norm(img))

    #
    # Write your Data Augmentation code here
    # mirroring

    if augment:
        mrr_data = [np.flip(img, 1) for img in data]
        data.extend(mrr_data)
        labels.extend(labels)

    if rotate_90:
        aug_data = [rotate(img, np.random.randint(0, 360),
                           reshape=False) for img in data]
        data.extend(aug_data)
        labels.extend(labels)

    #
    # Write your Data Normalization code here
    # norm data to zero-centered
    # img already normed above

    if zero_centered:
        for i in range(len(data)):
            #   x' = x - μ
            pixels = np.asarray(data[i]).astype('float64')
            data[i] = pixels - pixels.mean()

    # randomly permute (this step is important for training)
    if shuffle:
        bundle = list(zip(data, labels))
        random.shuffle(bundle)
        data, labels = zip(*bundle)

    # divide data into minibatches of TorchTensors
    if batch_num > 1:
        batch_data = []
        batch_labels = []

        print(len(data))
        print(batch_num)

        for i in range(int(len(data) / batch_num)):
            minibatch_d = data[i * batch_num: (i + 1) * batch_num]
            minibatch_d = np.reshape(
                minibatch_d, (batch_num, channel_num, img_size[0], img_size[1]))
            batch_data.append(torch.from_numpy(minibatch_d))

            minibatch_l = labels[i * batch_num: (i + 1) * batch_num]
            batch_labels.append(torch.LongTensor(minibatch_l))
        data, labels = batch_data, batch_labels

    return zip(batch_data, batch_labels)


# load data into size (64, 64)
img_size = (64, 64)
batch_num = 50  # training sample number per batch

# load training dataset
trainloader_small = list(load_dataset('./data/train/', img_size, batch_num=batch_num, shuffle=True,
                                      augment=True, zero_centered=True))
train_num = len(trainloader_small)
print("Finish loading %d minibatches(=%d) of training samples." %
      (train_num, batch_num))

# load testing dataset
testloader_small = list(load_dataset(
    './data/test/', img_size, num_per_class=50, batch_num=batch_num))
test_num = len(testloader_small)
print("Finish loading %d minibatches(=%d) of testing samples." %
      (test_num, batch_num))


# show some images


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if len(npimg.shape) > 2:
        npimg = np.transpose(img, [1, 2, 0])
    plt.figure
    plt.imshow(npimg, 'gray')
    plt.show()


img, label = trainloader_small[0][0][11][0], trainloader_small[0][1][11]
label = int(np.array(label))
print(class_names[label])
imshow(img)


################################################## q 1 #############################################################


# part1

# ==========================================
#       Define Network Architecture
# ==========================================
class Model(nn.Module):
    # referencing model that introduction in https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
    def __init__(self):
        dropout_rate = 0.5
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)  # output channel =12 kenel size - 5*5
        self.pool1 = nn.MaxPool2d(2)  # 2*2 window
        self.conv2 = nn.Conv2d(12, 16, 5)  # kenel size - 5*5
        # using previous conv layer's output channel as input channel
        self.pool2 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 16)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# ==========================================
#         Optimize/Train Network
# ==========================================
num_epoch = 50
device = torch.device("cuda") if torch.cuda.is_available(
) else torch.device("cpu")  # use cuda if that's available
print("cuda is available", torch.cuda.is_available())
nn_part1 = Model().double().to(device)


def train_optimize_model(dataset, NNmodel, epochs=30):
    optimizer = optim.Adam(NNmodel.parameters(), lr=0.001)
    for i in range(epochs):
        for batch in dataset:
            x, y = batch
            optimizer.zero_grad()
            output = NNmodel((x.to(device)))
            loss = nn.CrossEntropyLoss()(output, y.to(device))
            loss.backward()
            optimizer.step()


init_train_time_p1 = time()
train_optimize_model(trainloader_small, nn_part1, num_epoch)
time_took_4_train_n_optimize_mode = time()


# ==========================================
#            Evaluating Network
# ==========================================

def evaluate_network(dataset, NNModel):
    for data in dataset:
        x, y = data
        return float((torch.max(NNModel(Variable(x.to(device))), 1)[1] == y.to(device)).sum() / float(y.size(0)))


acc = evaluate_network(testloader_small, nn_part1)

print('question 1 part 1\n')
print('Data augmentation:  mirroring')
print(
    ' Data normalization: [-1,1], zero-centered by substract mean from pixels ')
print(' Network Regularization: dropout layer after each conv layer')
print('\n\nnet struct：', nn_part1)
print('\n\nAccuracy is {0:.2f}%'.format(acc * 100), 'after', num_epoch,
      'epoches in {0:.2f}'.format(time_took_4_train_n_optimize_mode - init_train_time_p1), 'secs')
standard_acc = acc


# part2 approach 1: Data augmentation:  mirroring, Data normalization [-1,1], zero-centered, Network Regularization: using dropout layers, activation function sigmoid
class Model_sigmoid(nn.Module):
    # referencing model that introduction in https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
    def __init__(self):
        dropout_rate = 0.5
        super(Model_sigmoid, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)  # output channel =12 kenel size - 5*5
        self.pool1 = nn.MaxPool2d(2)  # 2*2 window
        self.conv2 = nn.Conv2d(12, 16, 5)  # kenel size - 5*5
        # using previous conv layer's output channel as input channel
        self.pool2 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 16)

    def forward(self, x):
        x = self.pool1(torch.sigmoid(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(torch.sigmoid(self.conv2(x)))
        x = self.dropout2(x)

        x = x.view(-1, self.num_flat_features(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# part2 approach 2: Data augmentation:  mirroring, Data normalization [-1,1], zero-centered, batch normlaization, Network Regularization: using dropout layers, activation function relu
class Model_bathch_norm(nn.Module):
    # referencing model that introduction in https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
    def __init__(self):
        dropout_rate = 0.5
        super(Model_bathch_norm, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 5)  # output channel =12 kenel size - 5*5
        self.pool1 = nn.MaxPool2d(2)  # 2*2 window
        self.conv2 = nn.Conv2d(12, 16, 5)  # kenel size - 5*5
        self.norm1 = nn.BatchNorm2d(16)

        # using previous conv layer's output channel as input channel
        self.pool2 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 16)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = self.norm1(x)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# part2 optimaization/tranning phrase
num_epoch = 50

# print("cuda is available",torch.cuda.is_available())
nn_part2_1 = Model_sigmoid().double().to(device)
nn_part2_2 = Model_bathch_norm().double().to(device)
nn_part2_3 = Model().double().to(device)

# nn_part2_3 = Model_softmax().double().to(device)

# nn_part2_3 = Model().double().to(device)
# print(123)


init_time = time()
train_optimize_model(trainloader_small, nn_part2_1, num_epoch)
time_p2_1 = time()
train_optimize_model(trainloader_small, nn_part2_2, num_epoch)
time_p2_2 = time()

trainloader_small_rotated = list(load_dataset(
    'data/train/', img_size, batch_num=batch_num, shuffle=True, augment=True, zero_centered=True, rotate_90=True))
testloader_small_rotated = list(load_dataset(
    'data/test/', img_size, num_per_class=100, batch_num=batch_num))
time_p2_3_ini = time()

# train_optimize_model(trainloader_small_rotated, nn_part2_3,num_epoch)
train_optimize_model(trainloader_small_rotated, nn_part2_3, num_epoch)

time_p2_3_fin = time()

# evaluation for part 2 test 1
acc = evaluate_network(testloader_small, nn_part2_1)

print(
    'test 1\n Data augmentation:  mirroring\n Data normalization [-1,1], zero-centered\n Network Regularization: using dropout layers\n activation function sigmoid')
print('\n\n NN struct：', nn_part2_1)
print('\n\nAccuracy is {0:.2f}%'.format(acc * 100), 'after', num_epoch,
      'epoches in {0:.2f}'.format(time_p2_1 - init_time), 'secs')
print('accuracy increase is{0:.2f}%'.format(acc * 100 - standard_acc * 100))

# evaluation for part 2 test 2
acc = evaluate_network(testloader_small, nn_part2_2)

print(
    'test 2\n Data augmentation:  mirroring\n  Data normalization [-1,1], zero-centered, batch normlaization\n Network Regularization: using dropout layers\n')
print('\n\n NN struct：', nn_part2_2)
print('\n\nAccuracy is {0:.2f}%'.format(acc * 100), 'after', num_epoch,
      'epoches in {0:.2f}'.format(time_p2_2 - time_p2_1), 'secs')
print('accuracy increase is{0:.2f}%'.format(acc * 100 - standard_acc * 100))

# evaluation for part 2 test 3
acc = evaluate_network(testloader_small_rotated, nn_part2_3)

print(
    'test 1\n Data augmentation:  mirroring, retated 90 degrees\n Data normalization [-1,1], zero-centered\n Network Regularization: using dropout layers\n')

print('\n\n NN struct：', nn_part2_3)
print('\n\nAccuracy is {0:.2f}%'.format(acc * 100), 'after', num_epoch,
      'epoches in {0:.2f}'.format(time_p2_3_fin - time_p2_3_ini), 'secs')
print('accuracy increase is{0:.2f}%'.format(acc * 100 - standard_acc * 100))

############################################################ q 2 #################################################

# reload data with a larger size
img_size = (224, 224)
batch_num = 50  # training sample number per batch

# load training dataset
trainloader_large = list(load_dataset('./data/train/', img_size, batch_num=batch_num, shuffle=True,
                                      augment=False, is_color=True, zero_centered=True))
train_num = len(trainloader_large)
print("Finish loading %d minibatches(=%d) of training samples." %
      (train_num, batch_num))

# load testing dataset
testloader_large = list(load_dataset(
    './data/test/', img_size, num_per_class=50, batch_num=batch_num, is_color=True))
test_num = len(testloader_large)
print("Finish loading %d minibatches(=%d) of testing samples." %
      (test_num, batch_num))

# part 1
print("replace final layer with 16 channels")

temp_model = models.alexnet(pretrained=True).double().to(device)
temp_model.classifier[-1] = nn.Linear(
    temp_model.classifier[-1].in_features, 16)
loss_fun = nn.CrossEntropyLoss()
temp_optimizer = torch.optim.SGD(
    temp_model.parameters(), lr=0.001, momentum=0.9)
temp_scheduler = torch.optim.lr_scheduler.StepLR(
    temp_optimizer, step_size=7, gamma=0.1)
temp_model.to(device)
temp_model = temp_model.double()
print("part 1")


# part1
# referencing https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    time1 = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in trainloader_large:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs.to(device))
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.to(device))
        scheduler.step()
        # print(len(trainloader_large))
        epoch_acc = running_corrects.double() / len(trainloader_large)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    time2 = time()
    totalTime = time2 - time1
    print('Training complete in {:.0f}m {:.0f}s'.format(
        totalTime // 60, totalTime % 60))
    #   print('Best Acc: {:4f}'.format(best_acc*100))
    model.load_state_dict(best_model_wts)
    return model


best_model = train_model(temp_model, loss_fun,
                         temp_optimizer, temp_scheduler, num_epochs=20)

time1 = time()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader_large:
        images, labels = data
        outputs = best_model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print("replace final layer with 16 channels")
print('NN struct：', best_model)
print('Accuracy of the network on the test images: %d %%' %
      (100 * correct / total))
time2 = time()
print("total " + str(time2 - time1) + "seconds")

# part 2
time1 = time()
alex = models.alexnet(pretrained=True).double()
temp_model = LinearSVC(penalty="l2", C=1.0, random_state=23333)
# remove last layer
del alex.classifier[-1]
featureSize = 200

X_train, y_train = [], []
for train, label in trainloader_large:
    X_train.append(alex(train)[:, :featureSize].detach().numpy())
    y_train.append(label.detach().numpy())
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)
temp_model.fit(X_train, y_train)
time2 = time()

correct_pred = 0
total_pred = 0
for data, label in testloader_large:
    predicted = temp_model.predict(
        alex(data)[:, :featureSize].detach().numpy())
    correct_pred += np.sum(predicted == label.detach().numpy())
    total_pred += len(label)

acc = correct_pred / total_pred * 100

print('part 2')
print('net struct：', alex)
print('Accuracy is {0:.2f}%'.format(acc),
      ' within {0:.2f}'.format(time2 - time1), 'seconds')

# bonus
print('question 2 bonus')
temp_vgg_model = models.vgg16(pretrained=True).double().to(device)
temp_vgg_model.classifier[-1] = nn.Linear(
    temp_vgg_model.classifier[-1].in_features, 16)
loss_fun = nn.CrossEntropyLoss()
temp_optimizer = torch.optim.SGD(
    temp_vgg_model.parameters(), lr=0.001, momentum=0.9)
temp_scheduler = torch.optim.lr_scheduler.StepLR(
    temp_optimizer, step_size=7, gamma=0.1)
temp_vgg_model.to(device)
temp_vgg_model = temp_vgg_model.double()

vgg_model = train_model(temp_vgg_model, loss_fun,
                        temp_optimizer, temp_scheduler, num_epochs=5)

time1 = time()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader_large:
        images, labels = data
        outputs = vgg_model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print("replace final layer with 16 channels")
print('NN struct：', vgg_model)
print('Accuracy of the network on the test images: %d %%' %
      (100 * correct / total))
time2 = time()
print("total " + str(time2 - time1) + "seconds")

print('reason why vgg is better than alexnet\n')
print(
    'vgg does not use large receptive fields like alexnet (11*11 with stride of 4 compares to 3*3 with stride set to 1 => multiple maller size kernel is better than a large kernel becuase with the number to increasing non-linear layers, the depth of the NN is increasing therefore the nn can learn more complex feattures ')
print('vgg also uses fewer parameters')
