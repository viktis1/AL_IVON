import torch
import torchvision
import numpy as np
import argparse
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from helper_functions import load_datasets, train_model_ivon, label_iteration_ivon
import ivon

# Parse arguments
parser = argparse.ArgumentParser(description="Set seed and test_samples_ivon for active learning")
parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
parser.add_argument('--test_samples_ivon', type=int, default=20, help="Number of test samples for IVON")
args = parser.parse_args()

# seed for reproducibility
seed = args.seed
torch.manual_seed(seed)

### Hyperparameters --------------------------------------------------------------------
val_split = 0.1
unlabelled_size = 0.95
lr = 1e0
batch_size = 64
num_epochs = 300
label_iterations = 5
weight_decay = 1e-4
test_samples_ivon = args.test_samples_ivon
 

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


# Print sanity checks so you don't waste time
print(f"{device=}")
print(f"{torch.get_num_threads()=}")

## Run active learning with IVON optimizer ----------------------------------------------------------------------------------------
datapoint_list_ivon = []
accuracy_list_ivon = []
for i in range(label_iterations):
    optimizer = ivon.IVON(model.parameters(), lr=lr, ess=len(train_dataset), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Important to reset the model each time
    accuracies = train_model_ivon(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=num_epochs, val_interval=20, test_samples=test_samples_ivon)
    datapoint_list_ivon.append(len(train_dataset))
    accuracy_list_ivon.append(accuracies)
    if i < label_iterations - 1:
        train_dataset, unlabbelled_dataset = label_iteration_ivon(model, train_dataset, unlabbelled_dataset, device, batch_size, test_samples_ivon, optimizer, top_frac=0.01)

np.savez(f'data/ivon_learningW/ivon_learningW_{seed}_{test_samples_ivon}.npz', datapoints=datapoint_list_ivon, accuracies=accuracy_list_ivon)