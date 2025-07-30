import torch
import torchvision
import torchvision.transforms as transforms
import sys


cinic_directory = sys.argv[1]
saving_directory = sys.argv[2]
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
cinic_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/train',
        transform=transforms.Compose([transforms.Resize((64, 64)),
            transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=128, shuffle=True)

cinic_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/test',
        transform=transforms.Compose([transforms.Resize((64, 64)),
            transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=128, shuffle=True)

train_images = []
train_labels = []

for images, labels in cinic_train:
    train_images.append(images)
    train_labels.append(labels)

X_train = torch.cat(train_images, dim=0)
y_train = torch.cat(train_labels, dim=0)

torch.save(X_train, saving_directory + "/x_train.pt")
torch.save(y_train, saving_directory + "/y_train.pt")

del train_images
del train_labels
del X_train
del y_train

test_images = []
test_labels = []
for images, labels in cinic_test:
    test_images.append(images)
    test_labels.append(labels)

X_test = torch.cat(test_images, dim=0)
y_test = torch.cat(test_labels, dim=0)

torch.save(X_test, saving_directory + "/x_test.pt")
torch.save(y_test, saving_directory + "/y_test.pt")
