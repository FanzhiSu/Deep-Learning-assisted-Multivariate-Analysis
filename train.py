import torch
import torch.nn.functional as F
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn as nn
from dataset_binary_classification import pytorch_data
from torch.utils.data import DataLoader
from model import CNN
#from model import ResNet, Bottleneck
import time
from torch.autograd import Variable
start_time = time.time()
import tqdm

NUM_EPOCHS = 50
NUM_CLASSES = 2
LEARNING_RATE = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = None
image_path = None
df = None

data_set =  pytorch_data(df, image_path, transform = transforms.ToTensor())
data_loader = DataLoader(dataset = data_set, batch_size = 64)
def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# define transformation that converts a PIL image into PyTorch tensors
data_transformer = transforms.Compose([ transforms.ToTensor(),transforms.Resize((100,100)), transforms.RandomCrop((80,80)), transforms.Normalize(mean=mean,std=std)])



img_dataset = pytorch_data(labels, data_dir, transform = data_transformer) # Histopathalogic images

# load an example tensor
# img,label=img_dataset[11]
# print(img.shape,torch.min(img),torch.max(img))

# Define the following transformations for the training dataset
# split dataset
len_img = len(img_dataset)
len_train = int(0.7*len_img)
len_val = len_img-len_train
train_ts,val_ts = random_split(img_dataset,[len_train,len_val]) # random split 80/20


# Training DataLoader
train_dl = DataLoader(train_ts,
                      batch_size=10, 
                      shuffle=True)

# Validation DataLoader
val_dl = DataLoader(val_ts,
                    batch_size=300,
                    shuffle=False)
val_dl_2 = DataLoader(img_dataset_2,
                    batch_size=1,
                    shuffle=False)


# Create instantiation of Network class
cnn_model = CNN()

# define computation hardware approach (GPU/CPU)
model = cnn_model.to(device)
loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)

# keeping-track-of-losses 
train_losses = []
valid_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0
    # training-the-model
    model.train()

    for data, target in train_dl:
        # move-tensors-to-GPU 
        data = Variable(data)
        data = data.to(device)
        # target = target.unsqueeze(1)
        #y must be one hot key encode
        # target = target.type(torch.LongTensor)
        target = Variable(target)
        target = target.to(device)
        print('target_1.shape', target.shape)
        # clear-the-gradients-of-all-optimized-variables
        optimiser.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        print('output.shape', output.shape)
        # calculate-the-batch-loss
        # loss = loss_func(output, target)
        loss = loss_func(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimiser.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

      

        # validate-the-model
    model.eval()
    for data, target in val_dl:
        data = Variable(data)
        data = data.to(device)
        # target = target.unsqueeze(1)
        # target = target.type(torch.LongTensor)
        target = Variable(target)
        target = target.to(device)
        output = model(data)
        # loss = loss_func(output, target)
        loss = loss_func(output, target)
        print('target',target)
            
        # update-average-validation-loss 
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss/len(train_dl.sampler)
    valid_loss = valid_loss/len(val_dl.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)    

    # print-training/validation-statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))


# test-the-model

model.eval()  # it-disables-dropout

with torch.no_grad():
    beta = 1.0
    correct = 0
    total = 0
    for images, labels in val_dl_2:
        labels = Variable(labels)
        labels = labels.to(device)
        images = Variable(images)
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        prob = F.softmax(outputs.data, dim=1)
        print('prob', prob)
        print('pre',predicted)
        print('pre.shape', predicted.shape)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        labels_2 = labels.unsqueeze(-1)
        labels = labels.unsqueeze(0)
        acc = utils.get_performance_metrics(y = labels.cpu().numpy(), y_2 = labels_2.cpu().numpy(), pred_2 = outputs.cpu().numpy(), pred = outputs, class_labels = ['signal'])

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
print("--- %s seconds ---" % (time.time() - start_time))

