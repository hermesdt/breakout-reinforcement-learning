import torch
from torch import nn

class ParamNoiseLayer(nn.Linear):
    def __init__(self, in_features, out_features, sigma=0, bias=True):
        super().__init__(in_features, out_features, bias)

        self.sigma = sigma
        self.register_buffer("weight_noise", torch.zeros(self.weight.size()))
        if self.bias is not None:
            self.register_buffer("bias_noise", torch.zeros(self.bias.size()))
        self.reset_noise()

    def forward(self, input):
        weight = self.weight + self.weight_noise
        bias = None
        if self.bias is not None:
            bias = self.bias + self.bias_noise

        return nn.functional.linear(input, weight, bias)
    
    def reset_noise(self):
        normal_dist = torch.distributions.Normal(0, self.sigma**2)

        noise = normal_dist.sample(self.weight.size())
        self.weight_noise.data = noise.data

        if self.bias is not None:
            noise = normal_dist.sample(self.bias.size())
            self.bias_noise.data = noise.data