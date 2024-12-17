import torch
import torchvision
import numpy as np
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from helper_functions import load_datasets, train_model, label_iteration_random
from helper_functions import train_model_ivon, label_iteration_ivon
import ivon


### Hyperparameters ------------------------------------------------------------------------------------------------
val_split = 0.1
unlabelled_size = 0.99
batch_size = 64
num_epochs = 300 # the number of epochs I want to normally run
label_iterations = 5
weight_decay = 1e-4
test_samples_ivon=20
epochs_eval = 5 # the epoch I want to evaluate to see lr impact

# Set learning rates as a list
lr_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]


# Set seed to account for run2run variance
torch.manual_seed(0)

# Setup model given the seed -----------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
# Modify input layer to accept 1 channel
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model_parameters = deepcopy(model.state_dict())
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()

# Load data ---------------------------------------------------------------------------------------------------
val_loader, unlabbelled_dataset, train_dataset, _, _ = load_datasets(val_split, unlabelled_size)


## Run Random selecter (baseline - ADAM optimizer) -----------------------------------------------------------------------------
accuracy_list_adam = []
loss_list_adam = []
for i, lr in enumerate(lr_list):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.load_state_dict(model_parameters)  # Important to reset the model each time
    accuracy, loss_list = train_model(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=epochs_eval, val_interval=1, return_loss=True)
    print(loss_list)
    loss_list = [loss.detach().numpy() for loss in loss_list]
    accuracy_list_adam.append(accuracy)
    loss_list_adam.append(loss_list)


## Run Random selecter (baseline - IVON optimizer) -----------------------------------------------------------------------------

accuracy_list_ivon = []
loss_list_ivon = []
for i, lr in enumerate(lr_list):
    optimizer = ivon.IVON(model.parameters(), lr=lr, ess=len(train_dataset), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Important to reset the model each time
    accuracy, loss_list = train_model_ivon(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=epochs_eval, val_interval=1, test_samples=test_samples_ivon, return_loss=True)
    loss_list = [loss.detach().numpy() for loss in loss_list]
    accuracy_list_ivon.append(accuracy)
    loss_list_ivon.append(loss_list)


    np.savez(
    f'data/lr_explorer/lr_explorer.npz',
    accuracy_list_ivon=accuracy_list_ivon,
    loss_list_ivon=loss_list_ivon,
    accuracy_list_adam=accuracy_list_adam,
    loss_list_adam=loss_list_adam
)

    
