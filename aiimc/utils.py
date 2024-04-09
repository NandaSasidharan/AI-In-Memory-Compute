import torch
import torch.nn as nn
import torch.fx as fx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

__all__ = [
    'load_data',
    'train',
    'quantize',
    'evaluate_accuracy',
    'replace_layer',
]


def load_data(dataset_name, root='data'):
    if dataset_name == 'MNIST':
        # Load MNIST dataset
        transform = transforms.ToTensor()
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return [train_loader, test_loader]


def train(model, train_loader, learning_rate, num_epoch, save='True', loss='crossentropy', optim='adam', noise_std=0, clamp_std=0 ):
    
    """
    Train the model for a given number of epochs. Optionally you can add noise to the weights 

    and/or clamp the weights for a given standard deviation multiplier.

    """
    
    # Define loss function and optimizer
    if loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for _ in range(num_epoch):  # 
        for images, labels in train_loader:

            if noise_std:
                # Add random noise to layer weights
                with torch.no_grad():
                    for param in model.parameters():
                        noise = torch.randn_like(param) * noise_std * param.std()
                        param.add_(noise)    

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if clamp_std:
            # Clip the weights to twice their standard deviation
                with torch.no_grad():
                    for param in model.parameters():
                        mean = param.mean()
                        std = param.std()
                        param.clamp_(min=mean - clamp_std * std, max=mean + clamp_std * std)
    # end of training loop

    print('Training finished.')        
    # Save the trained model
    if save:
        saved_name = f'saved_models/{model.name}_lr{learning_rate}_numEpoch{num_epoch}'
        if noise_std:
            saved_name += f'_noise_std{noise_std}'
        if clamp_std:
            saved_name += f'_clamp_std{clamp_std}'

        saved_name += f'.pth'
        torch.save(model.state_dict(), saved_name)
        print(f'Model state_dict saved as {saved_name}.')

    return saved_name


# Quantize the model parameters into a given number of levels within its value range. 
def quantize(model, num_levels):
    with torch.no_grad():
        for param in model.parameters():
            quantization_step = torch.max(torch.abs(param)) *2 / (num_levels-1)
            param.div_(quantization_step).round_().mul_(quantization_step)


# Evaluate accuracy on test dataset
def evaluate_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def replace_layer(model, original_layer, new_layer):
    # Create a symbolic trace of the model
    traced = fx.symbolic_trace(model)

    # Iterate over all nodes in the graph
    for node in traced.graph.nodes:
        # If the node is a call_module node (i.e., it corresponds to a nn.Module in the original model)
        if node.op == 'call_module':
            # If the module is an instance of original_layer
            if isinstance(getattr(traced, node.target), original_layer):
                # Get the module
                original_layer_module = getattr(traced, node.target)
                # Create a new new_layer module with the same parameters
                new_layer_module = new_layer(original_layer_module.in_features, original_layer_module.out_features)
                new_layer_module.weight = original_layer_module.weight
                new_layer_module.bias = original_layer_module.bias
                # Replace the old module with the new one
                setattr(traced, node.target, new_layer_module)

    # Recompile the graph to propagate the changes we made
    traced.recompile()

    return traced