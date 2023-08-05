p = '/home/wtownle1/dimensionality_powerlaw/activation_models/AtlasNet/'
import sys
sys.path.append(p)
import torch
import numpy as np
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from functools import partial
from utils import properties_to_id
from activation_models.AtlasNet.model_2L_eig import EngineeredModel2L_Eig
from activation_models.AtlasNet.model_2L_SVD import EngineeredModel2L_SVD

atlasnet_layers = ['c2', 'mp2']

def get_activation_models(atlasnet=True):
    
    if atlasnet:
        for model, layers in atlas_net():
            yield model, layers
            

def atlas_net():
    eig = True
    SVD = False
        
    if eig:
        inits = [1]
        for i in inits:
            task = f'Eig_test_{i}'
            
            model = EngineeredModel2L_Eig(filters_2=1000, k_size=9, exponent=-2).Build()
            #identifier = properties_to_id('AtlasNet', task, 'Untrained', 'PyTorch')
            #model = wrap_atlasnet(model, identifier)
            yield model, atlasnet_layers
            
    if SVD:
        inits = [1,2]
        for i in inits:
            task = f'SVD_test_{i}'
            
            model = EngineeredModel2L_SVD(filters_2=1000, k_size=9, exponent=-1).Build()
            identifier = properties_to_id('AtlasNet', task, 'Untrained', 'PyTorch')
            model = wrap_atlasnet(model, identifier)
            yield model, atlasnet_layers


def wrap_atlasnet(model, identifier, res=96, norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
    preprocess = partial(load_preprocess_images, image_size=res,
                         normalize_mean=norm[0], normalize_std=norm[1])
    return PytorchWrapper(model=model,
                          preprocessing=preprocess,
                          identifier=identifier)