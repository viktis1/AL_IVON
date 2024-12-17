import torch
import torchvision
import numpy as np
import argparse
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from helper_functions import load_datasets, train_model, label_iteration

# Parse arguments
parser = argparse.ArgumentParser(description="Set seed and test_samples_ivon for active learning")
parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
args = parser.parse_args()

# Set seed for reproducibility
seed = args.seed
torch.manual_seed(seed)

### Hyperparameters --------------------------------------------------------------------
val_split = 0.1
unlabelled_size = 0.95
lr = 1e-2
batch_size = 64
num_epochs = 300
label_iterations = 5
weight_decay = 1e-1


# Load data ---------------------------------------------------------------------------------------------------
val_loader, unlabbelled_dataset, train_dataset, _, _ = load_datasets(val_split, unlabelled_size)

# Setup model -------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
# Modify input layer to accept 1 channel
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 
model_parameters = deepcopy(model.state_dict())
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()

## Run active learning ----------------------------------------------------------------------------------------

datapoint_list = []
accuracy_list = []
for i in range(label_iterations):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Important to reset the model each time
    accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=num_epochs, val_interval=20)
    datapoint_list.append(len(train_dataset))
    accuracy_list.append(accuracies)
    if i < label_iterations - 1:
        train_dataset, unlabbelled_dataset = label_iteration(model, train_dataset, unlabbelled_dataset, device, batch_size, top_frac=0.01)

np.savez(f'data/active_learningW/active_learningW_{seed}.npz', datapoints=datapoint_list, accuracies=accuracy_list)
