import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 action_size,
                 hidden_size=400,
                 activation=F.relu,  
    ):
        super(VAE, self).__init__()
    
        self.input_size = input_size
        self.output_size = output_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.embedding_size = input_size
        self.activation = activation

        # Set encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Linear(self.input_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size)
        ])

        # Set embedding layers
        self.embedding_layer1 = nn.Linear(self.hidden_size, self.embedding_size)
        self.embedding_layer2 = nn.Linear(self.hidden_size, self.embedding_size)
        
        # Set decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Linear(self.embedding_size + self.action_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size)
        ])

        # Set output layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def encode(self, x):
        for encoder_layer in self.encoder_layers:
            x = self.activation(encoder_layer(x))
        return self.embedding_layer1(x), self.embedding_layer2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, a):
        x = torch.cat([z,a], dim=-1).to(device)
        for decoder_layer in self.decoder_layers:
            x = self.activation(decoder_layer(x))
        return self.output_layer(x)

    def forward(self, s, a):
        mu, logvar = self.encode(s)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, a), mu, logvar