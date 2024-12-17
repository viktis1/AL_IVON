import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict
from matplotlib.cm import viridis
from matplotlib.cm import get_cmap


# Create function for plotter
def plotter(data_name, num_epochs, path):
    # load the data
    data = np.load(f"data/{path}/{data_name}.npz")
    datapoints = data['datapoints']
    accuracies = data['accuracies']
    print(datapoints)
    print(accuracies)

    print(datapoints)
    print(accuracies)

    # Generate epoch indices (assuming every 20 epochs)
    epochs = np.arange(20, num_epochs+1, 20)

    # Define labels for each line
    labels = [f"{datapoints[i]} labeled datapoints" for i in range(datapoints.shape[0])]

    # Plot each line with corresponding label
    plt.figure(figsize=(10, 6))
    for i in range(accuracies.shape[0]):
        plt.plot(epochs, accuracies[i], marker='o', linestyle='-', label=labels[i])

    # Add plot details
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.xlim([0, num_epochs])
    plt.ylim([0, 100])
    plt.title("Validation Accuracy over Epochs for Different Models")
    plt.legend(title=data_name)
    plt.grid(True)
    plt.savefig(f"figs/{data_name}.png")



def plot_lr_explorer():
    # Load saved accuracy data
    data_path = "data/lr_explorer/lr_explorer.npz"
    data = np.load(data_path)

    accuracy_list_adam = np.array(data['accuracy_list_adam'])  # Adam
    loss_list_ivon=loss_list_adam = np.array(data['loss_list_adam'])
    accuracy_list_ivon = np.array(data['accuracy_list_ivon'])  # IVON
    loss_list_ivon=loss_list_ivon = np.array(data['loss_list_ivon'])

    print(accuracy_list_adam.shape)
    print(loss_list_adam.shape)

    # Define the learning rates used in the experiment
    lr_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    # Epochs or steps (x-axis)
    epochs = np.arange(1, accuracy_list_adam.shape[1] + 1)

    # Figure for accuracy ------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    handles = []
    # Plot for adam
    for i, lr in enumerate(lr_list):
        line, = axes[0].plot(epochs, accuracy_list_adam[i, :], label=f'LR = {lr:.0e}')
        handles.append(line)  # Collect handles for fig.legend()
    axes[0].set_title('Accuracy - Adam')
    axes[0].set_ylabel('Accuracy')
    # axes[0].legend(title='Learning Rates')
    axes[0].grid(True)
    fig.legend(handles=handles, title='Learning Rates', loc='upper left', bbox_to_anchor=(0, 1), ncol=3)
    # Plot for ivon
    for i, lr in enumerate(lr_list):
        axes[1].plot(epochs, accuracy_list_ivon[i, :], label=f'LR = {lr:.0e}')
    axes[1].set_title('Accuracy - Ivon')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    # axes[1].legend(title='Learning Rates', loc='lower left')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("figs/lr_explorer_accuracy.png")
    
    # Figure for loss ------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    handles = []
    # Plot for adam
    for i, lr in enumerate(lr_list):
        line, = axes[0].plot(epochs, loss_list_adam[i, :], label=f'LR = {lr:.0e}')
        handles.append(line)  # Collect handles for fig.legend()
    axes[0].set_title('Loss - Adam')
    axes[0].set_ylabel('loss')
    # axes[0].legend(title='Learning Rates')
    axes[0].grid(True)
    axes[0].set_yscale('log')
    fig.legend(handles=handles, title='Learning Rates', loc='upper right', bbox_to_anchor=(1, 0.5), ncol=3)

    # Plot for ivon
    for i, lr in enumerate(lr_list):
        axes[1].plot(epochs, loss_list_ivon[i, :], label=f'LR = {lr:.0e}')
    axes[1].set_title('Loss - IVON')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('loss')
    # axes[1].legend(title='Learning Rates')
    axes[1].grid(True)
    axes[1].set_yscale('log')
    plt.tight_layout()
    plt.savefig("figs/lr_explorer_loss.png")


def ivon_test_sample_comparer():
    from collections import defaultdict

    # Initialize data storage
    test_sample_data = defaultdict(lambda: defaultdict(list))
    folder = Path(f"data/ivon_learning/")

    # Load and organize data
    for item in folder.iterdir():
        parts = item.stem.split("_")  # 'ivon_learning_seed_testsample'
        data = np.load(item)
        accuracies = data['accuracies']

        # Group accuracies by test_sample and dataset index
        for dataset_idx in range(accuracies.shape[0]):
            test_sample_data[test_sample][dataset_idx].append(accuracies[dataset_idx])

    # Compute mean and SD of accuracies
    results = {}
    for test_sample, datasets in test_sample_data.items():
        results[test_sample] = {}
        for dataset_idx, accuracy_list in datasets.items():
            stacked = np.vstack(accuracy_list)  # Stack all seeds' accuracies for this dataset
            mean_accuracies = np.mean(stacked, axis=0)  # Mean across seeds
            sd_accuracies = np.std(stacked, axis=0) / np.sqrt(stacked.shape[0])  # SEM across seeds
            results[test_sample][dataset_idx] = {
                "mean_accuracies": mean_accuracies,
                "sd_accuracies": sd_accuracies,
            }

    # -------- Plot All Datasets Together ------------
    plt.figure(figsize=(12, 8))
    test_samples = sorted(results.keys())
    markers = ['o-', 's-', '^-', 'd-', 'v-']  # Different markers for datasets
    colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for datasets

    # Prepare data for plotting
    epochs = np.arange(0, 300, 20) + 20

    for dataset_idx in range(5):  # Assuming 5 datasets
        for test_sample in test_samples:
            stats = results[test_sample][dataset_idx]
            mean_accuracies = stats['mean_accuracies']/100
            sd_accuracies = stats['sd_accuracies']/100
            
            # # Transform data
            # log_mean_accuracies = np.log(mean_accuracies/100)
            # log_sd_accuracies = sd_accuracies / mean_accuracies
    
            # Plot with error bars
            plt.errorbar(
            epochs, mean_accuracies, yerr=sd_accuracies,
            label=f'Dataset {dataset_idx + 1}',
            capsize=5, fmt=markers[dataset_idx % len(markers)],
            color=colors[dataset_idx % len(colors)]
            )
    
    # Add labels, title, and legend
    plt.yscale('logit')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy Comparison Across Datasets and Test Samples', fontsize=14)
    plt.legend(title="Legend", loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.yticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [20,30,40,50,60,70,80,90])
    plt.minorticks_off()
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    plt.savefig("figs/ivon_test_sample_comparer.png")


def ivon_test_explorer():
    # Define the directory containing the datasets
    data_dir = Path("data/ivon_learning_explorer")
    
    # Collect all dataset files in the directory
    data_files = sorted(data_dir.glob("ivon_learning_explorer_*.npz"))
    
    # Initialize a list to store accuracies
    all_accuracies = []
    
    # Loop through all dataset files and collect accuracies
    for file in data_files:
        data = np.load(file)
        accuracies = np.array(data['accuracies'])
        all_accuracies.append(accuracies)
    
    # Convert the list of accuracies to a NumPy array for easier computation
    all_accuracies = np.array(all_accuracies)  # Shape: (num_datasets, num_epochs, num_indices)
    
    N = all_accuracies.shape[0]

    # Compute mean and standard deviation across datasets
    mean_accuracies = np.mean(all_accuracies, axis=0)[0]  # Shape: (num_epochs, num_indices)
    std_accuracies = np.std(all_accuracies, axis=0)[0] / np.sqrt(N)    # Shape: (num_epochs, num_indices)
    
    # Define x-axis values
    x = np.array([1, 2, 4, 8, 16, 32, 64])
    
    # Plot mean with error bars
    plt.figure(figsize=(8, 6))
    for epoch in range(mean_accuracies.shape[0]):
        plt.errorbar(np.arange(x.shape[0]), mean_accuracies[epoch], yerr=std_accuracies[epoch], 
                     marker='o', label=f'Epoch {100 * (epoch + 1)}', capsize=3)
    
    # Labels and legend
    plt.xlabel('Number of Test Samples', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Mean Accuracies with Error Bars Across Datasets', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(np.arange(x.shape[0]), x)

    
    # Save plot
    plt.savefig("figs/ivon_test_explorer_mean_error.png")
    plt.show()

def compare_methods():

    def mean_and_sem(folder):
        data = defaultdict(list)
        for item in folder.iterdir():
            file_data = np.load(item)
            accuracies = file_data['accuracies']
            datapoints = file_data['datapoints']
            for dataset_idx in range(accuracies.shape[0]):
                data[dataset_idx].append(accuracies[dataset_idx])
        # Compute mean and SD of accuracies
        mean_accuracies_list = []
        sem_accuracies_list = []

        for dataset_idx, accuracy_list in data.items():
            stacked = np.vstack(accuracy_list)  # Stack all seeds' accuracies for this dataset
            mean_accuracies = np.mean(stacked, axis=0)  # Mean across seeds
            sem_accuracies = 1.96*np.std(stacked, axis=0) / np.sqrt(stacked.shape[0])  # SEM across seeds

            mean_accuracies_list.append(mean_accuracies)
            sem_accuracies_list.append(sem_accuracies)

        # Convert lists to arrays for final storage
        mean_accuracies_array = np.array(mean_accuracies_list)
        sem_accuracies_array = np.array(sem_accuracies_list)
        return mean_accuracies_array, sem_accuracies_array, datapoints
    
    
    # Find the data for ivon, baseline and normal
    folder = Path(f"data/ivon_learningW/")
    mean_accuracies_ivon, sem_accuracies_ivon, datapoints = mean_and_sem(folder)
    folder = Path(f"data/active_learningW/")
    mean_accuracies_active, sem_accuracies_active, datapoints = mean_and_sem(folder)
    folder = Path(f"data/baselineW/")
    mean_accuracies_base, sem_accuracies_base, datapoints = mean_and_sem(folder)
    folder = Path(f"data/random_ivon/")
    mean_accuracies_bivon, sem_accuracies_bivon, datapoints = mean_and_sem(folder)
    # folder = Path(f"data/baselineW/")
    # mean_accuracies_base, sem_accuracies_base = mean_and_sem(folder)
    # We want the learning rate for the selected epochs 100, 200, 300
    epochs = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
    
    AL_it = np.array([1,2,3,4,5])
    for index in range(epochs.shape[0]):
        selected_epochs = [index]
        # Extract data for selected epochs
        mean_ivon = mean_accuracies_ivon[:, selected_epochs].flatten()
        sem_ivon = sem_accuracies_ivon[:, selected_epochs].flatten()
        mean_active = mean_accuracies_active[:, selected_epochs].flatten()
        sem_active = sem_accuracies_active[:, selected_epochs].flatten()
        mean_base = mean_accuracies_base[:, selected_epochs].flatten()
        sem_base = sem_accuracies_base[:, selected_epochs].flatten()
        mean_bivon = mean_accuracies_bivon[:, selected_epochs].flatten()
        sem_bivon = sem_accuracies_bivon[:, selected_epochs].flatten()

        # Plotting
        bivon_color = '#b40000'  # Slightly different shade of blue
        ivon_color = '#fe5d01'  # Base color for IVON (blue)
        base_color = '#000081'  # Slightly different shade of orange
        adam_color = '#015dfd'  # Base color for AdamW (orange)
        plt.figure(figsize=(10, 6))
        plt.errorbar(AL_it, mean_bivon, yerr=sem_bivon, label="IVON Baseline", fmt='o-', capsize=5, color=bivon_color)
        plt.errorbar(AL_it, mean_ivon, yerr=sem_ivon, label="IVON Learning", fmt='o-', capsize=5, color=ivon_color)
        plt.errorbar(AL_it, mean_base, yerr=sem_base, label="AdamW Baseline", fmt='s-', capsize=5, color=base_color)
        plt.errorbar(AL_it, mean_active, yerr=sem_active, label="ADAM Learning", fmt='s-', capsize=5, color=adam_color)

        plt.xlabel("Number of labeled data")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Comparison at epoch {epochs[selected_epochs][0]}")
        plt.legend()
        plt.xticks(AL_it, datapoints)
        plt.grid(True)
        plt.savefig(f'figs/compare_methods_{epochs[selected_epochs][0]}.png')


def generate_accuracy_plots(data_folder, output_path="figs/"):
    """
    Generate plots for mean accuracies with error bars for a given data folder.

    Args:
        data_folder (str): Path to the directory containing the .npz files.
        output_path (str): Path to save the generated plot.
    """
    # Initialize data storage
    accuracy_data = defaultdict(list)
    folder = Path(data_folder)

    # Load and organize data
    for item in folder.iterdir():
        if item.suffix == ".npz":
            data = np.load(item)
            accuracies = data['accuracies']
            datapoints = data['datapoints']

            # Group accuracies by dataset index
            for dataset_idx in range(accuracies.shape[0]):
                accuracy_data[dataset_idx].append(accuracies[dataset_idx])

    # Compute mean and SD of accuracies
    results = {}
    for dataset_idx, accuracy_list in accuracy_data.items():
        stacked = np.vstack(accuracy_list)  # Stack all seeds' accuracies for this dataset
        mean_accuracies = np.mean(stacked, axis=0)  # Mean across seeds
        sd_accuracies = 1.96*np.std(stacked, axis=0) / np.sqrt(stacked.shape[0])  # SEM across seeds
        results[dataset_idx] = {
            "mean_accuracies": mean_accuracies,
            "sd_accuracies": sd_accuracies,
        }

    # -------- Plot All Datasets Together ------------
    plt.figure(figsize=(12, 8))
    colormap = get_cmap('cividis')  
    num_datasets = len(results)
    colors = colormap(np.linspace(0, 1, num_datasets))  # Generate colors for each dataset
    
    # Prepare data for plotting
    epochs = np.arange(0, 300, 20) + 20

    for dataset_idx, (stats, color) in enumerate(zip(results.values(), colors)):
        mean_accuracies = stats['mean_accuracies'] / 100
        sd_accuracies = stats['sd_accuracies'] / 100

        # Plot with error bars
        plt.errorbar(
            epochs, mean_accuracies, yerr=sd_accuracies,
            label=f'{datapoints[dataset_idx]} datapoints',
            capsize=5, fmt='o-', color=color
        )

    # Add labels, title, and legend
    plt.yscale('logit')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy Comparison for {data_folder}', fontsize=14)
    plt.legend(title="Legend", loc='best', fontsize=12, frameon=True, borderpad=1, edgecolor='black')
    plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999], [20, 30, 40, 50, 60, 70, 80, 90, 99, 99.9])
    plt.ylim([0, 0.999])
    plt.minorticks_off()
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / f"accuracy_plot_{data_folder.split('/')[-2]}.png")
    print(output_path / f"accuracy_plot_{data_folder.split('/')[-2]}.png")
    plt.close()

# Example Usage
# Generate plots for different directories
generate_accuracy_plots("data/active_learningW/")
generate_accuracy_plots("data/baselineW/")
generate_accuracy_plots("data/ivon_learningW/")
generate_accuracy_plots("data/random_ivon/")
compare_methods()




# Call the function
# ivon_test_explorer()
# ivon_test_sample_comparer()


        
# ivon_test_sample_comparer()

# Learning Rate expolorer
# plot_lr_explorer()

# plotter("baseline_0", num_epochs=300, path="baseline")
# plotter("baseline_1", num_epochs=300, path="baseline")
# plotter("active_learning_0", num_epochs=300, path="active_learning")
# plotter("ivon_learning_0_20", num_epochs=300, path="ivon_learning")
# plotter("ivon_learning_0_20_old", num_epochs=300, path="ivon_learning")



# plotter("ivon_learning_1_1", num_epochs=300, path="ivon_learning")
# plotter("ivon_learning_1_2", num_epochs=300, path="ivon_learning")
# plotter("ivon_learning_1_5", num_epochs=300, path="ivon_learning")
# plotter("ivon_learning_1_10", num_epochs=300, path="ivon_learning")
# plotter("ivon_learning_1_20", num_epochs=300, path="ivon_learning")

