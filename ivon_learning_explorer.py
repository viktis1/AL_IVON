import torch
import torchvision
import numpy as np
import argparse
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from helper_functions import load_datasets
import ivon

# Parse arguments
parser = argparse.ArgumentParser(description="Set seed and test_samples_ivon for active learning")
parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
args = parser.parse_args()

# seed for reproducibility
seed = args.seed
torch.manual_seed(seed)

### Hyperparameters --------------------------------------------------------------------
val_split = 0.1
unlabelled_size = 0.99
lr = 1e0
batch_size = 64
num_epochs = 300
weight_decay = 1e-4
test_samples_list = [1,2,4,8,16,32,64]


# Load data ---------------------------------------------------------------------------------------------------
val_loader, unlabbelled_dataset, train_dataset, _, _ = load_datasets(val_split, unlabelled_size) 

# Setup model -------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
# Modify input layer to accept 1 channel
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = ivon.IVON(model.parameters(), lr=lr, ess=len(train_dataset), weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)


# Create helper functions -----------------------------------------------------------------------------------------

def validate_model_ivon(model, val_loader, test_samples_list, optimizer, device):
    model.eval()
    correct = [0] * len(test_samples_list)
    total = [0] * len(test_samples_list)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            for i, test_samples in enumerate(test_samples_list):
                sampled_probs = []
                for _ in range(test_samples):   
                    with optimizer.sampled_params(): 
                        outputs = model(images).softmax(dim=1)
                        sampled_probs.append(outputs)
                    # print(sampled_probs)
                    prob = torch.mean(torch.stack(sampled_probs), dim=0)
                    _, predicted = torch.max(prob, 1)
                    total[i] += labels.size(0)
                    correct[i] += (predicted == labels).sum().item()
                    # print(total[i])
                    # print(correct[i])

    # Convert lists to tensors for division and return as percentages
    correct = torch.tensor(correct)
    total = torch.tensor(total)
    return 100 * correct / total


def train_model_ivon(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=10, val_interval=100):
    accuracies = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with optimizer.sampled_params(train=True):    
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if (epoch + 1) % val_interval == 0:
            val_accuracy = validate_model_ivon(model, val_loader, test_samples_list, optimizer, device)
            accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}')
            print(val_accuracy)
    
    return accuracies

## Run active learning with IVON optimizer ----------------------------------------------------------------------------------------
datapoint_list_ivon = []
accuracy_list_ivon = []
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
accuracies = train_model_ivon(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=num_epochs, val_interval=100)
datapoint_list_ivon.append(len(train_dataset))
accuracy_list_ivon.append(accuracies)

np.savez(f'data/ivon_learning_explorer/ivon_learning_explorer_{seed}.npz', datapoints=datapoint_list_ivon, accuracies=accuracy_list_ivon)