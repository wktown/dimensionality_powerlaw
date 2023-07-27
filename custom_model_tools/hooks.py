import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from model_tools.activations.core import change_dict, flatten
from model_tools.activations.pca import _get_imagenet_val
from model_tools.utils import fullname
from result_caching import store_dict
from custom_model_tools.zca import ZCA
from sklearn.decomposition import PCA


class GlobalMaxPool2d:
    def __init__(self, activations_extractor):
        self._extractor = activations_extractor

    def __call__(self, batch_activations):
        def apply(layer, activations):
            if activations.ndim != 4:
                return activations
            activations = torch.from_numpy(activations)
            activations = F.adaptive_max_pool2d(activations, 1)
            activations = activations.numpy()
            
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor):
        hook = GlobalMaxPool2d(activations_extractor)
        assert not cls.is_hooked(activations_extractor), "GlobalMaxPool2d already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())

#add av pooling
class GlobalAvgPool2d:
    def __init__(self, activations_extractor):
        self._extractor = activations_extractor

    def __call__(self, batch_activations):
        def apply(layer, activations):
            if activations.ndim != 4:
                return activations
            activations = torch.from_numpy(activations)
            activations = F.adaptive_avg_pool2d(activations, 1)
            activations = activations.numpy()
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor):
        hook = GlobalAvgPool2d(activations_extractor)
        assert not cls.is_hooked(activations_extractor), "GlobalAvgPool2d already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())


class RandomProjection:
    def __init__(self, activations_extractor, n_components=1024):
        self._extractor = activations_extractor
        self.n_components = n_components
        self.layer_ws = {}
        np.random.seed(27)

    def __call__(self, batch_activations):
        def apply(layer, activations):
            activations = activations.reshape(activations.shape[0], -1)
            if layer not in self.layer_ws:
                w = np.random.normal(size=(activations.shape[-1], self.n_components)) / np.sqrt(self.n_components)
                self.layer_ws[layer] = w
            else:
                w = self.layer_ws[layer]
            activations = activations @ w
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor, n_components=1024):
        hook = RandomProjection(activations_extractor=activations_extractor, n_components=n_components)
        assert not cls.is_hooked(activations_extractor), "RandomProjection already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())


#add spatial PCA
class SpatialPCA:
    def __init__(self, activations_extractor):
        self._extractor = activations_extractor
        
    def __call__(self, batch_activations):
        def apply(layer, activations):
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        
    @classmethod
    def hook(cls, activations_extractor):
        hook = SpatialPCA(activations_extractor)
        assert not cls.is_hooked(activations_extractor), "SpatialPCA already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
        
        
#one random spatial position (instead of max pool)
class RandomSpatial:
    def __init__(self, activations_extractor):
        self._extractor = activations_extractor
        np.random.seed(27)

    def __call__(self, batch_activations):
        def apply(layer, activations):
            if activations.ndim != 4:
                return activations
            w = activations.shape[2]
            h = activations.shape[3]
            x = np.random.randint(low=0, high=w)
            y = np.random.randint(low=0, high=h)
            activations = activations[:,:,x,y]
            
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor):
        hook = RandomSpatial(activations_extractor)
        assert not cls.is_hooked(activations_extractor), "RandomSpatial already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
        

class MaxPool_PCA:
    def __init__(self, activations_extractor, n_components):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._n_components = n_components
        self._layer_pcas = {}
        
        def __call__(self, batch_activations):
            
            @store_dict(dict_key='layers', identifier_ignore=['layers'])
            def apply(layers, activations): #or something
                
                activations = torch.from_numpy(activations)
                print('activations pre-pooling:')
                print(activations.size)
                activations = F.adaptive_max_pool2d(activations, 1)
                activations = activations.numpy()
                activations = flatten(activations)
                print('activations post-pooling:')
                print(activations.shape)
                
                #def _ensure_initialized(self, layers):
                missing_layers = [layer for layer in layers if layer not in self._layer_pcas]
                if len(missing_layers) == 0:
                    return
                
                layer_pcas = self._pcas(identifier=self._extractor.identifier, layers=missing_layers,
                                n_components=self._n_components)
                self._layer_pcas = {**self._layer_pcas, **layer_pcas}
                
                if activations.shape[1] <= n_components:
                    print(f"Not computing principal components for {layer} "
                                    f"activations {activations.shape} as shape is small enough already")
                    pca = None
                else:
                    pca = PCA(n_components=self._n_components, random_state=0)
                    pca.fit(activations)
                    
                if pca is None:
                    print('no pca shape:')
                    print(activations.shape)
                    return activations
                else:
                    print('pca shape:')
                    print(np.shape(pca.transform(activations)))
                    return pca.transform(activations)
                
            return change_dict(batch_activations, apply, keep_name=True,
                               multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
            
    @classmethod
    def hook(cls, activations_extractor, n_components):
        hook = MaxPool_PCA(activations_extractor, n_components)
        assert not cls.is_hooked(activations_extractor), "MaxPool_PCA already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
        


class LayerZCA:
    def __init__(self, activations_extractor):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._layer_zcas = {}

    def __call__(self, batch_activations):
        self._ensure_initialized(batch_activations.keys())

        def apply_zca(layer, activations):
            zca = self._layer_zcas[layer]
            activations = flatten(activations)
            if zca is None:
                return activations
            return zca.transform(activations)

        return change_dict(batch_activations, apply_zca, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    def _ensure_initialized(self, layers):
        missing_layers = [layer for layer in layers if layer not in self._layer_zcas]
        if len(missing_layers) == 0:
            return
        layer_zcas = self._zcas(identifier=self._extractor.identifier, layers=missing_layers)
        self._layer_zcas = {**self._layer_zcas, **layer_zcas}

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _zcas(self, identifier, layers):
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = _get_imagenet_val(num_images=1000)
        self.handle.disable()
        imagenet_activations = self._extractor(imagenet_paths, layers=layers)
        imagenet_activations = {layer: imagenet_activations.sel(layer=layer).values
                                for layer in np.unique(imagenet_activations['layer'])}
        assert len(set(activations.shape[0] for activations in imagenet_activations.values())) == 1, "stimuli differ"
        self.handle.enable()

        self._logger.debug('Computing ImageNet ZCA whitening matrix')
        progress = tqdm(total=len(imagenet_activations), desc="zca whitening matrix")

        def init_and_progress(layer, activations):
            activations = flatten(activations)
            zca = ZCA()
            zca.fit(activations)
            progress.update(1)
            return zca

        from model_tools.activations.core import change_dict
        layer_zcas = change_dict(imagenet_activations, init_and_progress, keep_name=True,
                                 multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        progress.close()
        return layer_zcas

    @classmethod
    def hook(cls, activations_extractor):
        hook = LayerZCA(activations_extractor=activations_extractor)
        assert not cls.is_hooked(activations_extractor), "ZCA already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
