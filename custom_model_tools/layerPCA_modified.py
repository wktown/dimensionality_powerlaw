import logging
import os

import h5py
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from scipy import stats
from model_tools.activations.core import flatten, change_dict
from model_tools.utils import fullname, s3
from result_caching import store_dict
from custom_model_tools.hooks import GlobalMaxPool2d


class LayerPCA_Modified:
    def __init__(self, activations_extractor, n_components, mod='max_pool'):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._n_components = n_components
        self._mod = mod
        self._layer_pcas = {}
    #2
    def __call__(self, batch_activations):
        #max pool here??
        #if self._max_pooling == True:
        #    self._max_pool(batch_activations)
        self._ensure_initialized(batch_activations.keys())
        
        #9
        def apply_pca(layer, activations):
            pca = self._layer_pcas[layer]
            if self._mod == 'max_pool':
                activations = torch.from_numpy(activations)
                activations = F.adaptive_max_pool2d(activations, 1)
                activations = activations.numpy()
                activations = flatten(activations)
            elif self._mod == 'z_score':
                activations = flatten(activations)
                activations = stats.zscore(activations, axis=0)
            elif self._mod == 'none':
                activations = flatten(activations)
           
            if pca is None:
                return activations
            return pca.transform(activations)
        #8
        return change_dict(batch_activations, apply_pca, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
    #3
    def _max_pool(self, batch_activations):
        def apply(layer, activations):
            if activations.ndim != 4:
                return activations
            activations = torch.from_numpy(activations)
            activations = F.adaptive_max_pool2d(activations, 1)
            activations = activations.numpy()
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        
    #4
    def _ensure_initialized(self, layers):
        missing_layers = [layer for layer in layers if layer not in self._layer_pcas]
        if len(missing_layers) == 0:
            return
        layer_pcas = self._pcas(identifier=self._extractor.identifier, layers=missing_layers,
                                n_components=self._n_components)
        self._layer_pcas = {**self._layer_pcas, **layer_pcas}

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    #5
    def _pcas(self, identifier, layers, n_components):
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = _get_imagenet_val(num_images=n_components)
        self.handle.disable()
        
        imagenet_activations = self._extractor(imagenet_paths, layers=layers)
        #print('imagenet activations shape (extractor):')
        #print(imagenet_activations.shape)
        imagenet_activations = imagenet_activations.reset_index("neuroid")
        imagenet_activations = imagenet_activations.set_index({"neuroid": ["channel", "channel_x", "channel_y"]}).unstack(dim="neuroid")
        imagenet_activations = {layer: imagenet_activations.where(imagenet_activations.layer == layer, drop=True).values 
                                for layer in np.unique(imagenet_activations.layer.values)}
        #imagenet_activations = {layer: imagenet_activations.sel(layer=layer).values
                                #for layer in np.unique(imagenet_activations['layer'])}
        print('imagenet dict [0] shape')
        print(list(imagenet_activations.values())[0].shape)
        
        assert len(set(activations.shape[0] for activations in imagenet_activations.values())) == 1, "stimuli differ"
        
        #handle_2.remove()
        #for h in handles_2:
        #    h.remove()
        self.handle.enable()

        self._logger.debug('Computing ImageNet principal components')
        progress = tqdm(total=len(imagenet_activations), desc="layer principal components")
        #7
        def init_and_progress(layer, activations):
            if self._mod == 'max_pool':
                print('max pool')
                print('activations post imagenet dict (one layer?)')
                print(activations.shape)
                activations = torch.from_numpy(activations)
                activations = F.adaptive_max_pool2d(activations, 1)
                activations = activations.numpy()
                print('activations post pool')
                print(activations.shape)
                activations = flatten(activations)
                print('shape of activations post flatten:')
                print(activations.shape)
            elif self._mod == 'z_score':
                print('zscore')
                activations = flatten(activations)
                activations = stats.zscore(activations, axis=0)
                print('activations zcore shape:')
                print(activations.shape)
            elif self._mod == 'none':
                print('not modified')
                activations = flatten(activations)
                
            if activations.shape[1] < n_components:
                self._logger.debug(f"Not computing principal components for {layer} "
                                   f"activations {activations.shape} as shape is small enough already")
                pca = None
                print('no pca because n_features less than requested n_components')
            else:
                pca = PCA(n_components=n_components, random_state=0)
                pca.fit(activations)
                print('pca eigenvalues:')
                print(np.shape(pca.components_))
            progress.update(1)
            return pca
        #6
        from model_tools.activations.core import change_dict
        layer_pcas = change_dict(imagenet_activations, init_and_progress, keep_name=True,
                                 multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        progress.close()
        return layer_pcas

    @classmethod
    def hook(cls, activations_extractor, n_components, mod='max_pool'):
        #1
        hook = LayerPCA_Modified(activations_extractor=activations_extractor, n_components=n_components, mod=mod)
        assert not cls.is_hooked(activations_extractor), "PCA already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())


def _get_imagenet_val(num_images):
    _logger = logging.getLogger(fullname(_get_imagenet_val))
    num_classes = 1000
    num_images_per_class = (num_images - 1) // num_classes
    base_indices = np.arange(num_images_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)
    for i in range((num_images - 1) % num_classes + 1):
        indices.extend(50 * i + np.array([num_images_per_class]).astype(int))

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

    return filepaths