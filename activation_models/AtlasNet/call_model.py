import torch
p = '/home/wtownle1/dimensionality_powerlaw/activation_models/AtlasNet'
import sys
sys.path.append(p)
from activation_models.AtlasNet.call_old import get_activation_models
import numpy as np

c2_dict = {}
key = 0
for model,layers in get_activation_models():
    #** if not PytorchWrapper
    
    key = key+1
    c2_dict[key] = model.c2.weight

    #c2_dict['get_parameter'] = model.get_parameter(target='c2.weight')
    #c2_dict['c2.weight'] = model.c2.weight

print(torch.allclose(c2_dict[1], c2_dict[2]))
print(torch.allclose(c2_dict[1], c2_dict[3]))
print(torch.allclose(c2_dict[2], c2_dict[3]))