# %%

# import modules
import argparse
import torch

import aiimc.models as models 
from aiimc.xbar import xLinear

from aiimc.utils import load_data
from aiimc.utils import train
from aiimc.utils import quantize
from aiimc.utils import replace_layer
from aiimc.utils import evaluate_accuracy

# %%
# argument parsors
parser = argparse.ArgumentParser(description='PyTorch AI in-memory compute simulator')
parser.add_argument('-D', '--dataset', default='MNIST', help='MNIST|fashionMNIST|cifar10')
parser.add_argument('-M', '--model', default= 'MLP', help='MLP|ResNet18')
args = parser.parse_args()



# %%
# load data
[train_loader, test_loader] = load_data(args.dataset, root='data')

# %%
# load and display model

if args.model == 'MLP':
    model_1 = models.MLP()

learning_rate = 0.001
num_epochs = 1
model_1_location = train(model_1, train_loader, learning_rate, num_epochs)


# %%
# Create a second model by loading the saved weights
if args.model == 'MLP':
    model_2 = models.MLP()

# model_2.load_state_dict(torch.load(model_1_location))
model_2.load_state_dict(model_1.state_dict())

# retrain the second model with noise and clamp
learning_rate2 = 0.001
num_epochs2 = 1
noise_std = 0.01
clamp_std = 2
model_2_location = train(model_2, train_loader, learning_rate2, num_epochs2, noise_std=noise_std, clamp_std=clamp_std)


# %%
# create a third model, quantize its weights

if args.model == 'MLP':
    model_3 = models.MLP()

# intialize the model with a previous model parameters
model_3.load_state_dict(model_2.state_dict())

# quantize the model parameters
quantize(model_3, 5) 
# map the weights to a crossbar of given size (default = 256)
replace_layer(model_3, torch.nn.Linear, xLinear)

# %%
# display model evaluation result 
# Calculate accuracy for all the models
first_model_accuracy = evaluate_accuracy(model_1, test_loader)
second_model_accuracy = evaluate_accuracy(model_2, test_loader)
third_model_accuracy = evaluate_accuracy(model_3, test_loader)

print(f'First model accuracy: {first_model_accuracy:.4f}')
print(f'Second model accuracy: {second_model_accuracy:.4f}')
print(f'Third model accuracy (quantized): {third_model_accuracy:.4f}')

# %%

# custom code: integrate with the module!
# plot the weight distributions
import matplotlib.pyplot as plt

def plot_weight_distribution(model, model_name):
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Weight Distribution - {model_name}', fontsize=16)

    for i, layer in enumerate([model.L1, model.L2, model.L3], start=1):
        weights = layer.weight.data.view(-1).numpy()
        plt.subplot(1, 3, i)
        plt.hist(weights, bins=50, color='skyblue', edgecolor='black')
        plt.title(f'Layer {i}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Plot weight distributions
plot_weight_distribution(model_1, 'First Model')
plt.savefig("images/model_1_saved_image.jpg")
plot_weight_distribution(model_2, 'Second Model (with Noise)')
plt.savefig("images/model_2_saved_image.jpg")
plot_weight_distribution(model_3, 'Third Model (Quantized)')
plt.savefig("images/model_3_saved_image.jpg")



