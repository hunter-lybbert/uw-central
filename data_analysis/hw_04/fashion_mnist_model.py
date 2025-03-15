"""The model class for training FashionMNIST model."""
from torch import nn
from torch.nn import functional as F

FASHION_MNIST_INPUT_DIM = 784
FASHION_MNIST_OUTPUT_DIM = 10


#Define your (As Cool As It Gets) Fully Connected Neural Network 
class ACAIGFCN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_dims: list[int],
        activation_function: callable = F.relu,
        dropout: list[float] = [0.5],
        batch_norm: bool = False,
    ): 
        super(ACAIGFCN, self).__init__()

        #Define the network layer(s) and activation function(s)
        if not hidden_layer_dims:
            raise ValueError(f"No middle layers provided, expected at least one.")
        
        if len(dropout) > len(hidden_layer_dims):
            raise ValueError(f"Expected less dropout layers than hidden layers, got {len(dropout)} dropout layers and {len(hidden_layer_dims)} hidden layers.")
        
        self.dropout_rates = dropout
        self.do_batch_norm = batch_norm
        
        # Setup layers and batch normalization
        layers = []
        batch_norms = []
        prev_dim = input_dim

        for dim in hidden_layer_dims:

            layers.append(nn.Linear(prev_dim, dim))

            if batch_norm:
                batch_norms.append(nn.BatchNorm1d(dim))

            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        
        if self.do_batch_norm:
            self.batch_norms = nn.ModuleList(batch_norms)

        if self.dropout_rates:
            self.dropout_layers = nn.ModuleList([nn.Dropout(p) for p in dropout])

        self.activation_function = activation_function
 
    def forward(self, x):
        #Define how your model propagates the input through the network
        num_droput_layers = len(self.dropout_rates)
        for j, layer in enumerate(self.layers[:-1]):
            # Apply layer
            layer_output = layer(x)

            # Do batch normalization if needed
            if self.do_batch_norm:
                layer_output = self.batch_norms[j](layer_output)

            # Apply activation function
            x = self.activation_function(layer_output)

            # Apply dropout if needed
            if j < num_droput_layers - 1:
                x = self.dropout_layers[j](x)
        
        final_layer = self.layers[-1]
        return final_layer(x)
