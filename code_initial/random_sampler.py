import torch
import torchvision
import numpy as np
from copy import deepcopy
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import ivon

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", action='store_true', help="Debug mode")
args = ap.parse_args()
torch.manual_seed(0)

### Hyperparameters --------------------------------------------------------------------
val_split = 0.1
unlabelled_size = 0.99
lr = 0.0005
batch_size = 64
num_epochs = 300
label_iterations = 2
test_samples = 20

### Setup MNIST dataset --------------------------------------------------------------------------------
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
debug = args.debug
if debug:
    train_dataset.data = train_dataset.data[:1000]
    train_dataset.targets = train_dataset.targets[:1000]
    
    torch.set_num_threads(4)
val_dataset = deepcopy(train_dataset)
train_size = int((1 - val_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
# Permute the datasets so the validation and training set are created from the same data and there is no structure in the data.
indexes = torch.randperm(len(train_dataset)).tolist()

indexes_val = indexes[train_size:]
val_dataset.targets = val_dataset.targets[indexes_val]
val_dataset.data = val_dataset.data[indexes_val]
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)

indexes_train = indexes[:train_size]
train_dataset.targets = train_dataset.targets[indexes_train]
train_dataset.data = train_dataset.data[indexes_train] 
# Split training data into labelled and unlabelled
unlabelled_size = int(unlabelled_size * len(train_dataset))
indexes_train = torch.randperm(len(train_dataset)).tolist()  # Kind of seems unneccessary since this has already been randomly permuted to create tran data but for readiability is redone.
unlabbelled_dataset = deepcopy(train_dataset)
unlabbelled_dataset.targets = unlabbelled_dataset.targets[indexes_train[:unlabelled_size]]
unlabbelled_dataset.data = unlabbelled_dataset.data[indexes_train[:unlabelled_size]]
train_dataset.targets = train_dataset.targets[indexes_train[unlabelled_size:]]
train_dataset.data = train_dataset.data[indexes_train[unlabelled_size:]]
start_train_dataset = deepcopy(train_dataset)  # Save for baseline
start_unlabbelled_dataset = deepcopy(unlabbelled_dataset)  # Save for baseline

# Helper functions ------------------------------------------------------------------------------------
def transfer_unlabelled_to_labeled(unlabbelled_dataset, train_dataset, indexes):
    # Convert indexes to boolean mask
    indexes = torch.tensor([i in indexes for i in range(len(unlabbelled_dataset.targets))])
    
    train_dataset.targets = torch.cat([train_dataset.targets, unlabbelled_dataset.targets[indexes]])
    train_dataset.data = torch.cat([train_dataset.data, unlabbelled_dataset.data[indexes]])
    unlabbelled_dataset.targets = unlabbelled_dataset.targets[~indexes]
    unlabbelled_dataset.data = unlabbelled_dataset.data[~indexes]

    return train_dataset, unlabbelled_dataset

def validate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, val_interval=1):
    accuracies = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % val_interval == 0:
            val_accuracy = validate_model(model, val_loader, device)
            accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}, Accuracy: {val_accuracy:.2f}%')
    return accuracies

def train_model_ivon(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, val_interval=1):
    accuracies = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for images, labels in train_loader:
            with optimizer.sampled_params(train=True):    
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
            optimizer.step()

        if (epoch + 1) % val_interval == 0:
            val_accuracy = validate_model(model, val_loader, device)
            accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}, Accuracy: {val_accuracy:.2f}%')
    return accuracies

def label_iteration(model, train_dataset, unlabelled_dataset, device, top_frac=0.01):
    # Use model to label all images in validation set 
    model.eval()
    predictions = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for images, _ in tqdm(unlabelled_loader):
            images = images.to(device)
            outputs = model(images).softmax(dim=1)
            predictions.extend(outputs.detach().cpu().numpy())

    predictions = torch.tensor(predictions)
    # Find top % of images with lowest top-confidence
    top_percent = int(top_frac * len(predictions))
    _, top_indices = predictions.max(-1)[0].topk(top_percent, largest=False)
    print(f"Adding {len(top_indices)} images to training set")
    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labeled(unlabelled_dataset, train_dataset, top_indices)
    
    return train_dataset, unlabelled_dataset

def label_iteration_ivon(model, train_dataset, unlabelled_dataset, device, top_frac=0.01):
    # Use model to label all images in validation set 
    model.eval()
    predictions = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for images, _ in tqdm(unlabelled_loader):
            sampled_probs = []
            for i in range(test_samples):    
                with optimizer.sampled_params():
                    images = images.to(device)
                    outputs = model(images).softmax(dim=1)
                    sampled_probs.append(outputs)
            prob = torch.mean(torch.stack(sampled_probs), dim=0)
            predictions.extend(prob.detach().cpu().numpy())

    predictions = torch.tensor(predictions)
    # Find top % of images with lowest top-confidence
    top_percent = int(top_frac * len(predictions))
    _, top_indices = predictions.max(-1)[0].topk(top_percent, largest=False)
    print(f"Adding {len(top_indices)} images to training set")
    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labeled(unlabelled_dataset, train_dataset, top_indices)
    
    return train_dataset, unlabelled_dataset

def label_iteration_random(train_dataset, unlabelled_dataset, top_frac=0.01):
    # Generate random indices for the unlabeled dataset and take 1% of them and transfer to labeled.
    random_indices = torch.randperm(len(unlabelled_dataset)).tolist()
    num_to_label = int(top_frac * len(unlabelled_dataset))
    unlabel_to_labeled_indices = random_indices[:num_to_label]

    train_dataset, unlabelled_dataset = transfer_unlabelled_to_labeled(unlabelled_dataset, train_dataset, unlabel_to_labeled_indices)

    return train_dataset, unlabelled_dataset

# Setup model --------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
# Modify input layer to accept 1 channel
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model_parameters = deepcopy(model.state_dict())
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
## Run Random selecter (baseline) -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_dataset = deepcopy(start_train_dataset)
unlabbelled_dataset = deepcopy(start_unlabbelled_dataset)
datapoint_list_random = []
accuracy_list_random = []
for i in range(label_iterations):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Important to reset the model each time
    accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, val_interval=100)
    datapoint_list_random.append(len(train_dataset))
    accuracy_list_random.append(accuracies)
    if i < label_iterations - 1:
        train_dataset, unlabbelled_dataset = label_iteration_random(train_dataset, unlabbelled_dataset, top_frac=0.001)
    

## Run active learning ----------------------------------------------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_dataset = deepcopy(start_train_dataset)
unlabbelled_dataset = deepcopy(start_unlabbelled_dataset)
datapoint_list = []
accuracy_list = []
for i in range(label_iterations):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Important to reset the model each time
    accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, val_interval=100)
    datapoint_list.append(len(train_dataset))
    accuracy_list.append(accuracies)
    if i < label_iterations - 1:
        train_dataset, unlabbelled_dataset = label_iteration(model, train_dataset, unlabbelled_dataset, device, top_frac=0.001)

## Run active learning with IVON optimizer ----------------------------------------------------------------------------------------
optimizer = ivon.IVON(model.parameters(), lr=0.1, ess=len(train_dataset))

train_dataset = deepcopy(start_train_dataset)
unlabbelled_dataset = deepcopy(start_unlabbelled_dataset)
datapoint_list_ivon = []
accuracy_list_ivon = []
for i in range(label_iterations):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.load_state_dict(model_parameters)  # Important to reset the model each time
    accuracies = train_model_ivon(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, val_interval=100)
    datapoint_list_ivon.append(len(train_dataset))
    accuracy_list_ivon.append(accuracies)
    if i < label_iterations - 1:
        train_dataset, unlabbelled_dataset = label_iteration_ivon(model, train_dataset, unlabbelled_dataset, device, top_frac=0.001)

# Plot the accuracy
datapoints = np.array(datapoint_list)
accuracies = np.array(accuracy_list).max(-1)
datapoints_ivon = np.array(datapoint_list_ivon)
accuracies_ivon = np.array(accuracy_list_ivon).max(-1)
datapoints_random = np.array(datapoint_list_random)
accuracies_random = np.array(accuracy_list_random).max(-1)
plt.figure(figsize=(10, 5))
plt.plot(datapoints, accuracies, '-bo', label='AL Accuracy')
plt.plot(datapoints_ivon, accuracies_ivon, '-ko', label='IVON AL Accuracy')
plt.plot(datapoints_random, accuracies_random, '-ro', label='Random sampling')
plt.xlabel('Datapoints')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('figs/accuracy_random_sampler.png')

print("Done")
plt.show()
