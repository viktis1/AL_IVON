import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import ivon

### Setup MNIST dataset --------------------------------------------------------------------------------

def load_datasets(val_split, unlabelled_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    val_dataset = deepcopy(train_dataset)
    train_size = int((1 - val_split) * len(train_dataset))

    # Permute the datasets so the validation and training set are created from the same data and there is no structure in the data.
    indexes = torch.randperm(len(train_dataset)).tolist()
    
    indexes_val = indexes[train_size:]
    val_dataset.targets = val_dataset.targets[indexes_val]
    val_dataset.data = val_dataset.data[indexes_val]
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # all the data we can train on, but includes both labeled and unlabeled.
    indexes_train = indexes[:train_size]
    train_dataset.targets = train_dataset.targets[indexes_train]
    train_dataset.data = train_dataset.data[indexes_train] 

    # Split training data into labelled and unlabelled
    unlabelled_size = int(unlabelled_size * len(train_dataset))
    indexes_train = torch.randperm(len(train_dataset)).tolist()  # Kind of seems unneccessary since this has already been randomly permuted to create tran data but for readiability is redone.
    
    # Unlabelled dataset
    unlabbelled_dataset = deepcopy(train_dataset)
    unlabbelled_dataset.targets = unlabbelled_dataset.targets[indexes_train[:unlabelled_size]]
    unlabbelled_dataset.data = unlabbelled_dataset.data[indexes_train[:unlabelled_size]]
    # Labeled dataset
    train_dataset.targets = train_dataset.targets[indexes_train[unlabelled_size:]]
    train_dataset.data = train_dataset.data[indexes_train[unlabelled_size:]]

    # Another copy of labeled and unlabelled so that the model can be re initialized 
    start_train_dataset = deepcopy(train_dataset)  # Save for baseline
    start_unlabbelled_dataset = deepcopy(unlabbelled_dataset)  # Save for baseline

    return val_loader, unlabbelled_dataset, train_dataset, start_train_dataset, start_unlabbelled_dataset



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


def validate_model_ivon(model, val_loader, test_samples, optimizer, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            sampled_probs = []
            for _ in range(test_samples):  
                with optimizer.sampled_params():  
                    outputs = model(images).softmax(dim=1)
                    sampled_probs.append(outputs)
            prob = torch.mean(torch.stack(sampled_probs), dim=0)
            _, predicted = torch.max(prob, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

            



def train_model(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=10, val_interval=1, return_loss=False):
    accuracies = []
    loss_list = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % val_interval == 0:
            val_accuracy = validate_model(model, val_loader, device)
            accuracies.append(val_accuracy)
            loss_list.append(loss)
            print(f'Epoch {epoch + 1}, Accuracy: {val_accuracy:.2f}%')

    if return_loss==True:
        return accuracies, loss_list
    return accuracies



def train_model_ivon(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=10, val_interval=1, test_samples=20, return_loss=False):
    accuracies = []
    loss_list = []
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
            val_accuracy = validate_model_ivon(model, val_loader, test_samples, optimizer, device)
            accuracies.append(val_accuracy)
            loss_list.append(loss)
            print(f'Epoch {epoch + 1}, Accuracy: {val_accuracy:.2f}%')
    
    if return_loss==True:
        return accuracies, loss_list
    return accuracies



def label_iteration(model, train_dataset, unlabelled_dataset, device, batch_size, top_frac=0.01):
    # Use model to label all images in validation set 
    model.eval()
    predictions = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for images, _ in unlabelled_loader:
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



def label_iteration_ivon(model, train_dataset, unlabelled_dataset, device, batch_size, test_samples, optimizer, top_frac=0.01):
    # Use model to label all images in validation set 
    model.eval()
    predictions = []
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        for images, _ in unlabelled_loader:
            sampled_probs = []
            for _ in range(test_samples):    
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
