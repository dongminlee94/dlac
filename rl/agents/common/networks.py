import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def identity(x):
    """Return input without any change."""
    return x


"""
PPO critic
"""
class MLP(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(64,64), 
                 activation=F.relu, 
                 output_activation=identity,
                 use_output_layer=True,
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer

        # Set hidden layers
        self.hidden_layers = nn.ModuleList()
        in_size = self.input_size
        for next_size in self.hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.hidden_layers.append(fc)

        # Set output layers
        if self.use_output_layer:
            self.output_layer = nn.Linear(in_size, self.output_size)
        else:
            self.output_layer = identity

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))
        return x


"""
SAC qf
"""
class FlattenMLP(MLP):
    def forward(self, x, a):
        q = torch.cat([x,a], dim=-1)
        return super(FlattenMLP, self).forward(q)


"""
PPO actor
"""
class GaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(64,64),
                 activation=torch.tanh,
    ):
        super(GaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, x):
        mu = super(GaussianPolicy, self).forward(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        pi = dist.sample()
        return mu, std, dist, pi


"""
SAC actor
"""
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ReparamGaussianPolicy(MLP):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 hidden_sizes=(64,64),
                 activation=F.relu,
                 action_scale=1.0,
                 log_type='log',
                 q=1.5,
    ):
        super(ReparamGaussianPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]

        # Set output layers
        self.mu_layer = nn.Linear(in_size, output_size)
        self.log_std_layer = nn.Linear(in_size, output_size)
        self.action_scale = action_scale
        self.log_type = log_type
        self.q = 2.0 - q

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        clip_value = (u - x)*clip_up + (l - x)*clip_low
        return x + clip_value.detach()

    def apply_squashing_func(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        if self.log_type == 'log':
            log_pi -= torch.sum(torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6), dim=-1)
        elif self.log_type == 'log-q':
            log_pi -= torch.log(self.clip_but_pass_gradient(1 - pi.pow(2), l=0., u=1.) + 1e-6)
        return mu, pi, log_pi

    def tsallis_entropy_log_q(self, x, q):
        safe_x = torch.max(x, torch.Tensor([1e-6]).to(device))
        log_q_x = torch.log(safe_x) if q==1. else (safe_x.pow(1-q)-1)/(1-q)
        return log_q_x.sum(dim=-1)
        
    def forward(self, x):
        x = super(ReparamGaussianPolicy, self).forward(x)
        
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        
        # https://pytorch.org/docs/stable/distributions.html#normal
        dist = Normal(mu, std)
        pi = dist.rsample() # reparameterization trick (mean + std * N(0,1))

        if self.log_type == 'log':
            log_pi = dist.log_prob(pi).sum(dim=-1)
            mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
        elif self.log_type == 'log-q':
            log_pi = dist.log_prob(pi)
            mu, pi, log_pi = self.apply_squashing_func(mu, pi, log_pi)
            exp_log_pi = torch.exp(log_pi)
            log_pi = self.tsallis_entropy_log_q(exp_log_pi, self.q)
        
        # make sure actions are in correct range
        mu = mu * self.action_scale
        pi = pi * self.action_scale
        return mu, pi, log_pi


"""
Embedding Model
"""
class DynamicsEmbedding(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 action_size,
                 hidden_size=400,
                 activation=F.relu,  
    ):
        super(DynamicsEmbedding, self).__init__()
    
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
        x = torch.cat([z,a], dim=-1)
        for decoder_layer in self.decoder_layers:
            x = self.activation(decoder_layer(x))
        return self.output_layer(x)

    def forward(self, s, a):
        mu, logvar = self.encode(s)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, a), mu, logvar