import torch
from math import pi, sqrt

def norm_pdf(xs, mu, sigma):
    min_mean = xs - mu
    return torch.exp(-min_mean*min_mean/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma)
