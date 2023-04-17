import torch
from torchvision.models import resnet18, resnet50
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from functools import partial
from typing import List
#from utils import properties_to_id
import logging
from model_tools.utils import fullname
import numpy as np
import os
import h5py



def properties_to_id(architecture, task, kind, source):
    identifier = f'architecture:{architecture}|task:{task}|kind:{kind}|source:{source}'
    return identifier

resnet18_pt_layers = [f'layer1.{i}.relu' for i in range(2)] + \
                     [f'layer2.{i}.relu' for i in range(2)] + \
                     [f'layer3.{i}.relu' for i in range(2)] + \
                     [f'layer4.{i}.relu' for i in range(2)]

resnet50_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                     [f'layer2.{i}.relu' for i in range(4)] + \
                     [f'layer3.{i}.relu' for i in range(6)] + \
                     [f'layer4.{i}.relu' for i in range(3)]

def test_models():
    model = resnet18(pretrained=False)
    identifier = properties_to_id('ResNet18', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = resnet50(pretrained=False)
    identifier = properties_to_id('ResNet50', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers
    
def wrap_pt(model, identifier, res=224, norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
    preprocess = partial(load_preprocess_images, image_size=res,
                         normalize_mean=norm[0], normalize_std=norm[1])
    return PytorchWrapper(model=model,
                          preprocessing=preprocess,
                          identifier=identifier)



def get_imagenet_val(num_classes=1000, num_per_class=1, separate_classes=False):
    _logger = logging.getLogger(fullname(get_imagenet_val))
    base_indices = np.arange(num_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)

    framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
    imagenet_filepath = os.getenv('MT_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    if not os.path.isfile(imagenet_filepath):
        os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
        _logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
        s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)

    filepaths = []
    with h5py.File(imagenet_filepath, 'r') as f:
        for index in indices:
            imagepath = os.path.join(imagenet_dir, f"{index}.png")
            if not os.path.isfile(imagepath):
                image = np.array(f['val/images'][index])
                Image.fromarray(image).save(imagepath)
            filepaths.append(imagepath)

    if separate_classes:
        filepaths = [filepaths[i * num_per_class:(i + 1) * num_per_class]
                     for i in range(num_classes)]

    return filepaths


num_classes = num_classes = 1000
num_per_class = 10
image_paths = get_imagenet_val(num_classes=num_classes, num_per_class=num_per_class)
def get_image_paths(self) -> List[str]:
    return self.image_paths


def get_acts(activations_extractor, layers, hooks=None):
    extractor = activations_extractor
    
    for layer in layers:
    
        #handles = []
        #if hooks is not None:
        #    handles = [cls.hook(extractor) for cls in hooks]

        activations = extractor(image_paths, layers=[layer])
        activations = activations.sel(layer=layer).values
        
        print(layer)
        print(activations.shape)
        return activations

layer_dims = {}
for model, layers in test_models():
    activations = get_acts(model,layers)
    layer_dims[model,layer] = activations.shape
    
print(layer_dims)


    #if pooling = 'max':
    #    handle = 1
    
    #handles = []
    #if self._hooks is not None:
    #    handles = [cls.hook(extractor) for cls in hooks]

    #self._logger.debug('Retrieving stimulus activations')
    #activations = self._extractor(image_paths, layers=[layer])
    #activations = activations.sel(layer=layer).values



        if pooling == 'spatial_pca':
            last_ls = [l for l in layers if l > 'layer3.6.relu']
            positions = [0,1,2,3,4,5,6]
            lays = []
            for l in last_ls:
                for x in positions:
                    for y in positions:
                        l_new = l + f'.position{x}x{y}'
                        lays.append(l_new)
        else:
            lays = layers