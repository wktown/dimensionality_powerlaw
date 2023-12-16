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
    def __init__(self, activations_extractor, n_components, mod='none', ret='transformed', new_alpha = None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._n_components = n_components
        self._mod = mod
        self._return = ret
        self._newalpha = new_alpha
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
            #decomp_dict = self._layer_pcas[layer]
            #pca = decomp_dict['pca']
            
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
                print('activations shape')
                print(activations.shape)
           
            if pca is None:
                return activations
            
            if self._return == 'transformed':
                return pca.transform(activations)
            elif self._return == 'components':
                print('components shape')
                print(pca.components_.shape)
                return pca.components_
            elif self._return == 'trans_zscored':
                activations = pca.transform(activations)
                activations = stats.zscore(activations, axis=0)
                #n_pcs_shape = activations.shape[1]
                #print('n_components = activations.shape[1]')
                #print(n_pcs_shape)
                #new_eigenvalues = np.power(np.arange(1, n_pcs_shape+1, dtype=float), self._newalpha)
                #activations = activations * new_eigenvalues[np.newaxis, :]
                #print('final activations.shape')
                #print(activations.shape)
                return activations
            
            elif self._return == 'trans_reshape':
                if self._newalpha is None:
                    print('*did not specify new alpha*')
                    
                activations = pca.transform(activations)
                U, S, V_T = np.linalg.svd(activations, full_matrices=False)
                
                n_new_components = min(activations.shape) # = len(S)
                print('n_new_components')
                print(n_new_components)
                
                new_eigenvalues = np.power(np.arange(1, n_new_components+1, dtype=float), self._newalpha)
                
                S_new = np.zeros((n_new_components, n_new_components))
                np.fill_diagonal(S_new, val=np.sqrt( (new_eigenvalues*(n_new_components-1)) ))
                #S_mat = np.diag(S_new)
                
                activations_new = U @ S_new @ V_T
                print('activations_new shape')
                print(activations_new.shape)
                
                Uc, Sc, VTc = np.linalg.svd(activations_new, full_matrices=False)
                print('singular value all close:')
                print(np.allclose(Sc, np.diag(S_new)))
                print( np.dot(Sc, np.diag(S_new)) / (np.linalg.norm(Sc)*np.linalg.norm(np.diag(S_new))) )
                
                eigenvalues_check = Sc**2 / (n_new_components-1)
                print('SVD eigenvalues all close:')
                print(np.allclose(eigenvalues_check, new_eigenvalues))
                print( np.dot(eigenvalues_check, new_eigenvalues) / (np.linalg.norm(eigenvalues_check)*np.linalg.norm(new_eigenvalues)) )
                
                pca_check = PCA(random_state=0)
                pca_check.fit(activations_new)
                print('PCA eigenvalues all close')
                print(np.allclose(pca_check.explained_variance_, new_eigenvalues))
                print( np.dot(pca_check.explained_variance_, new_eigenvalues) / (np.linalg.norm(pca_check.explained_variance_)*np.linalg.norm(new_eigenvalues)) )
                
                return activations_new
                
            elif self._return == 'trans_reshape_2':
                if self._newalpha is None:
                    print('*did not specify new alpha*')
                decomp_dict = self._layer_pcas[layer]
                pca = decomp_dict['pca']
                U_learned = decomp_dict['U']
                VT_learned = decomp_dict['V_T']
                
                ###activations = pca.transform(activations)
                ###U, S, V_T = np.linalg.svd(activations, full_matrices=False)
                
                #n_new_components = min(activations.shape) # = len(S)
                n_new_components = U_learned.shape[0]
                print('n_new_components')
                print(n_new_components)
                
                new_eigenvalues = np.power(np.arange(1, n_new_components+1, dtype=float), self._newalpha)
                
                S_new = np.zeros((n_new_components, n_new_components))
                np.fill_diagonal(S_new, val=np.sqrt( (new_eigenvalues*(n_new_components-1)) ))
                #S_mat = np.diag(S_new)
                
                
                #**
                activations_new = U @ S_new @ V_T
                activations_projected = X @ VT_learned.T
                #**
                
                
                activations_new = U_learned @ S_new @ VT_learned
                activations_new = activations_new @ VT_learned.T
                
                print('activations_new shape')
                print(activations_new.shape)
                
                Uc, Sc, VTc = np.linalg.svd(activations_new, full_matrices=False)
                print('singular value all close:')
                print(np.allclose(Sc, np.diag(S_new)))
                print( np.dot(Sc, np.diag(S_new)) / (np.linalg.norm(Sc)*np.linalg.norm(np.diag(S_new))) )
                
                eigenvalues_check = Sc**2 / (n_new_components-1)
                print('SVD eigenvalues all close:')
                print(np.allclose(eigenvalues_check, new_eigenvalues))
                print( np.dot(eigenvalues_check, new_eigenvalues) / (np.linalg.norm(eigenvalues_check)*np.linalg.norm(new_eigenvalues)) )
                
                pca_check = PCA(random_state=0)
                pca_check.fit(activations_new)
                print('PCA eigenvalues all close')
                print(np.allclose(pca_check.explained_variance_, new_eigenvalues))
                print( np.dot(pca_check.explained_variance_, new_eigenvalues) / (np.linalg.norm(pca_check.explained_variance_)*np.linalg.norm(new_eigenvalues)) )
                
                return activations_new
            
            elif self._return == 'trans_reshape_3':
                if self._newalpha is None:
                    print('*did not specify new alpha*')
                    
                activations = pca.transform(activations)
                activations = stats.zscore(activations, axis=1)
                
                n_new_components = activations.shape[0] #**1
                new_eigenvalues = np.power(np.arange(1, n_new_components+1, dtype=float), self._newalpha)
                new_eigenvalues = np.sqrt(new_eigenvalues)

                #**activations_new = activations * new_eigenvalues[np.newaxis, :]
                activations_new = activations * new_eigenvalues[:, np.newaxis]
                return activations_new
                
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
        #layer_pcas, = self._pcas(identifier=self._extractor.identifier, layers=missing_layers,
        #                         n_components=self._n_components)
        layer_pcas = self._pcas(identifier=self._extractor.identifier, layers=missing_layers,
                                n_components=self._n_components)
        self._layer_pcas = {**self._layer_pcas, **layer_pcas}

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    #5
    def _pcas(self, identifier, layers, n_components):
        self._logger.debug('Retrieving ImageNet activations')
        #**imagenet_paths = _get_imagenet_val(num_images=n_components)
        imagenet_paths = _get_imagenet_val(num_images=1000)
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
                U = None
                S = None
                V_T = None
                decomp_dict = {'pca':pca, 'U':U, 'S':S, 'V_T':V_T}
                print('no pca because n_features less than requested n_components')
            else:
                pca = PCA(n_components=n_components, random_state=0)
                pca.fit(activations)
                print('pca eigenvalues:')
                print(np.shape(pca.components_))
                
                #U, S, V_T = np.linalg.svd(activations, full_matrices=False)
                #decomp_dict = {'pca':pca, 'U':U, 'S':S, 'V_T':V_T}
                
            progress.update(1)
            return pca
        #6
        from model_tools.activations.core import change_dict
        layer_pcas = change_dict(imagenet_activations, init_and_progress, keep_name=True,
                                 multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        #layer_decomp_dict = change_dict(imagenet_activations, init_and_progress, keep_name=True,
        #                                multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        progress.close()
        return layer_pcas

    @classmethod
    def hook(cls, activations_extractor, n_components, mod='none', ret='transformed', new_alpha=None):
        #1
        hook = LayerPCA_Modified(activations_extractor=activations_extractor, n_components=n_components, mod=mod, ret=ret, new_alpha=new_alpha)
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

def _majaj_images(num_images):
    #3200 images
    data_dir = os.getenv('BRAINIO_HOME', os.path.expanduser('~/.brainio'))
    data_dir = os.path.join(data_dir, 'image_dicarlo_hvm-public')
    assert os.path.exists(data_dir)

    image_paths = os.listdir(data_dir)

    if num_images is not None:
        assert len(image_paths) >= num_images
        image_paths = image_paths[:num_images]

    image_paths = [os.path.join(data_dir, file) for file in image_paths]
    image_paths = [p for p in image_paths if p[-4:] == '.png']
    return image_paths