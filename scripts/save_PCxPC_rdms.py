import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
#import torchlens as tl
from utils import timed#, id_to_properties
from torchvision.transforms import Grayscale
from custom_model_tools.hooks import GlobalMaxPool2d,  RandomProjection, RawActivations
from model_tools.activations.pca import LayerPCA
from custom_model_tools.layerPCA_modified import LayerPCA_Modified
from activation_models.generators import get_activation_models
from custom_model_tools.image_transform import ImageDatasetTransformer
from typing import Optional, List
from model_tools.utils import fullname
from tqdm import tqdm
from model_tools.activations.core import flatten
from scipy.spatial.distance import cdist
#from scipy.spatial.distance import squareform
from result_caching import store_dict
from utils import id_to_properties, get_imagenet_val
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import h5py
from PIL import Image
from sklearn.decomposition import PCA
from model_tools.activations.core import flatten, change_dict
from model_tools.utils import fullname, s3
from result_caching import store_dict
from torch.nn import functional as F
from scipy import stats

import logging
logging.basicConfig(level=logging.INFO)



#get model activations from majaj images (multiple powerlaw slopes in eigenspectrum)
#get brain data
# - create/plot RDMs

# - project model activations onto PCs (transformed)
#     - do RSA & encoding model predictions with various numbers of PCs (top 10, 50, 100, etc)
#     - plot rsa/encoding performance vs. number of PCs
# - same as above but z-score pc data first (could be similar to variance scale in eig script)

@timed
def main(data_dir, seed, pooling, grayscale, debug=False): #dataset
    image_transform = ImageDatasetTransformer('grayscale', Grayscale()) if grayscale else None
    
    if pooling=='max' or pooling=='none': #projections, spatial_PCA, random_spatial
        n_pcs = 'NA'
    elif pooling=='layerPCA' or pooling=='maxpool_PCA' or pooling=='zscore_PCA' or pooling=='PCA' or pooling=='PCAtrans_zscore' or pooling=='PCA_components':
        n_pcs = 1000
    
    inlcude_euclidean = False
    model_pcs = {}
    model_keys = []
    #euclidean_rdms = {}
    for model, layers in get_activation_models(seed, n_pcs):
        properties = id_to_properties(model.identifier)
        print(properties)
        
        pc = get_pcs(data_dir, model, pooling, n_pcs, image_transform, d_metric='correlation') #dataset
        pc.get(layers)
        if inlcude_euclidean:
            euclidean = get_pcs(data_dir, model, pooling, n_pcs, image_transform, d_metric='euclidean') #dataset
            euclidean.get(layers)
        
        key = properties['architecture']+'|'+properties['kind']
        print(key)
        model_keys.append(key)
        print(model_keys)
        model_pcs[key] = pc._pcs
        if inlcude_euclidean:
            euclidean_pcs[key] = euclidean._pcs
        
        
        #for layer in layers:
        #    key = properties['task']+'|'+properties['kind']+'|'+properties['source']+'|'+'layer:'+layer
        #    pearson_rdms[key] = pearson._rdms[layer]
        #    euclidean_rdms[key] = euclidean._rdms[layer]
        
        if debug:
            break

    #if not debug:
    
    print('--exit--')
    sns.set(style = 'white', context='paper',
    rc={'axes.titlesize': 12,
        'lines.linewidth': 2.5,
        'axes.linewidth': 0.4})
    sns.set(palette="colorblind")
    
    #seed = task.split('_')[1]
    #seed = seed.split('=')[1]
    
    #method = properties['task'].split('_')[0]
    #if method == 'Eig':
    #    m = 'X_transformed'
    #elif method == 'SVD':
    #    m = 'SVD'
    
    if pooling == 'layerPCA':
        pool = f'layerPCA:{n_pcs}'
    elif pooling == 'max':
        pool = f'pooling:{pooling}'
    elif pooling == 'maxpool_PCA':
        pool = f'maxpool_PCA:{n_pcs}'
    elif pooling == 'zscore_PCA':
        pool = f'z-scored_PCA:{n_pcs}'
    elif pooling == 'PCA':
        pool = f'PCA:{n_pcs}'
    elif pooling == 'PCAtrans_zscore':
        pool = f'transformed_z-scored'
    elif pooling == 'PCA_components':
        pool = pooling

    
    #alphas = ['-0.2', '-0.6', '-1.0', '-2', '-3']
    alphas = ['-0.2', '-3']
    
    layer_dict_02 = model_pcs[model_keys[0]]
    layer_dict_3 = model_pcs[model_keys[1]]
    
    pcs_02 = []
    for layer_pc_dicts02 in layer_dict_02.items():
        print('un-needed dict key, try returning layer_pcas instead of pcs[layer]')
        #try returning layer_pcas instead of pcs[layer]
        print(layer_pc_dicts02[0])
        print(layer_pc_dicts02[1])
        for layer_pcs in layer_pc_dicts02[1].items():
            print('layer_pcs 0.02')
            print(layer_pcs[0])
            print(layer_pcs[1])
            pcs_02.append(layer_pcs[1].components_)
            #if returning pca.components_ works later: pcs_02.append(layer_pcs[1]))
            # print(layer_pcs[1] will return components instead of PCA (perhaps with new models))
            
    pcs_3 = []
    for layer_pc_dicts3 in layer_dict_3.items():
        print('assumed layer pc dict key')
        print(layer_pc_dicts3[0])
        print(layer_pc_dicts3[1])
        #pcs_3.append(layer_pc_dicts3[1])
        for layer_pcs in layer_pc_dicts3[1].items():
            print('layer_pcs 3 key')
            print(layer_pcs[0])
            print(layer_pcs[1])
            pcs_3.append(layer_pcs[1].components_)
        
    print(pcs_02[0].shape)
    print(isinstance(pcs_02[0], np.ndarray))
    print(pcs_3[0].shape)
    print(isinstance(pcs_3[0], np.ndarray))
    
    distance = cdist(pcs_02[0], pcs_3[0], metric='correlation')
    print('distance shape:')
    print(distance.shape)
    #distance_mat = squareform(distance_vec)
    
    #architecture = model.split('|')[0]
    #architecture = architecture.split('_')[0]
    path = f'/home/wtownle1/dimensionality_powerlaw/figures/keaton/RDMs/PCxPC/AtlasNet'
    if not os.path.exists(path):
        os.makedirs(path)
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(distance, square=True, cbar_kws={"shrink": .8})
    ax.set(title=f'alphas=0.2,3.0 PCxPC RDM (1-pearson)')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(f'{path}/PCrdmMH|alphas=0.2,3|seed:{seed}|{pool}|Eig.png')#, dpi=300)
        #plt.savefig(f'{path}/PearsonRDM|alpha={a}|{pool}|.png')
        
        
        #*np.save(f'results/RDM_{method}_pearson|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', pearson_rdms)
        #*np.save(f'results/RDM_{method}_euclidean|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', euclidean_rdms)
        
        
def get_pcs(data_dir, activations_extractor, pooling, n_pcs, image_transform, d_metric): #dataset
    return EigenspectrumBase(activations_extractor=activations_extractor,
                             d_metric=d_metric,
                             pooling=pooling,
                             n_pcs=n_pcs,
                             image_transform=image_transform) #dataset=dataset,
    
    #if dataset == 'imagenet':
    #    return EigenspectrumImageNet(activations_extractor=activations_extractor,
    #                                 d_metric=d_metric,
    #                                 dataset=dataset,
    #                                 pooling=pooling,
    #                                 n_pcs=n_pcs,
    #                                 image_transform=image_transform)
        
    #elif dataset == 'majajhong2015':
    #    return EigenspectrumMajajHong2015(activations_extractor=activations_extractor,
    #                                      d_metric=d_metric,
    #                                      dataset=dataset,
    #                                      pooling=pooling,
    #                                      n_pcs=n_pcs,
    #                                      image_transform=image_transform)
        
    #else:
    #    raise ValueError(f'Unknown eigenspectrum dataset: {dataset}')
    
        
class EigenspectrumBase:

    def __init__(self, activations_extractor, d_metric, pooling, n_pcs, stimuli_identifier=None, #dataset
                 image_transform: Optional[ImageDatasetTransformer] = None,
                 hooks: Optional[List] = None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._n_pcs = n_pcs
        self._hooks = hooks
        self._stimuli_identifier = stimuli_identifier
        self._image_transform = image_transform
        self._metric = d_metric
        #self._dataset = dataset
        self._pcs = {}
        
    def get(self, layers):
        transform_name = None if self._image_transform is None else self._image_transform.name
        
        self._pcs = self.extract_pcs(identifier=self._extractor.identifier,
                                                        stimuli_identifier=self._stimuli_identifier,
                                                        layers=layers,
                                                        pooling=self._pooling,
                                                        n_pcs = self._n_pcs,
                                                        image_transform_name=transform_name)
        
        #np.save(f'results/modelRDM_{self._metric}|dataset:{self._dataset}|pooling:{self._pooling}|grayscale:{self._image_transform}', self._rdms)

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def extract_pcs(self, identifier, stimuli_identifier, layers, pooling, n_pcs, image_transform_name):
        
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
        
        #image_paths = self.get_image_paths()
        #if self._image_transform is not None:
        #    image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)
        

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        
        pcs = {}

        for layer in layers:
            if pooling == 'PCA_components':
                handle = RawActivations.hook(self._extractor)
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            #if pooling == 'max':
            #    handle = GlobalMaxPool2d.hook(self._extractor)
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:max'
            #elif pooling == 'none':
            #    handle = RandomProjection.hook(self._extractor)
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:none'
            #elif pooling == 'layerPCA':
            #    handle = LayerPCA.hook(self._extractor, n_components=n_pcs)
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:layerPCA|n_components:{n_pcs}'
            #elif pooling == 'maxpool_PCA':
            #    handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='max_pool')
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            #elif pooling == 'zscore_PCA':
            #    handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='z_score')
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            #elif pooling == 'PCA':
            #    handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='none')
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:PCA|n_components:{n_pcs}'
            #elif pooling == 'PCAtrans_zscore':
            #    handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='none', ret='trans_zscored')
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            #elif pooling == 'PCA_components':
            #    handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='none', ret='components')
            #    self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            
                
            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]
                
            logging.info(identifier)
            logging.info(layer)
            
            self._logger.debug('Retrieving ImageNet activations')
            imagenet_paths = _get_imagenet_val(num_images=n_pcs)
            handle.disable()
            
            imagenet_activations = self._extractor(imagenet_paths, layers=layers)
            imagenet_activations = imagenet_activations.reset_index("neuroid")
            imagenet_activations = imagenet_activations.set_index({"neuroid": ["channel", "channel_x", "channel_y"]}).unstack(dim="neuroid")
            imagenet_activations = {layer: imagenet_activations.where(imagenet_activations.layer == layer, drop=True).values 
                                    for layer in np.unique(imagenet_activations.layer.values)}
            
            print('imagenet dict [0] shape')
            print(list(imagenet_activations.values())[0].shape)
            
            assert len(set(activations.shape[0] for activations in imagenet_activations.values())) == 1, "stimuli differ"
            
            handle.enable()

            self._logger.debug('Computing ImageNet principal components')
            progress = tqdm(total=len(imagenet_activations), desc="layer principal components")
            

            def init_and_progress(layer, activations):
                
                activations = flatten(activations)
                    
                if activations.shape[1] < n_pcs:
                    self._logger.debug(f"Not computing principal components for {layer} "
                                        f"activations {activations.shape} as shape is small enough already")
                    pca = None
                    print('no pca because n_features less than requested n_components')
                else:
                    pca = PCA(n_components=n_pcs, random_state=0)
                    pca.fit(activations)
                    print('pca components:')
                    print(np.shape(pca.components_))
                progress.update(1)
                return pca.components_
                
            from model_tools.activations.core import change_dict
            layer_pcas = change_dict(imagenet_activations, init_and_progress, keep_name=True,
                                        multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
            
            progress.close()
            #return layer_pcas
            pcs[layer] = layer_pcas
            
            handle.remove()
            for h in handles:
                h.remove()

        return pcs
            
            
            #self._logger.debug('Retrieving stimulus activations')
            #activations = self._extractor(image_paths, layers=[layer])
            #print(activations.coords)
            
            #activations = activations.sel(layer=layer).values
            #print(activations.shape)

            #self._logger.debug('Computing RDMs')
            #progress = tqdm(total=1, desc="RDMs")
            
            ##  *- if computing similarity between two rdms
            ##corr, p = spearmanr(triu1, triu2)
            
            #progress.update(1)
            #progress.close()
            
            ##key = identifier+layer
            #rdms[layer] = activations
            ##euclidean_rdms[layer] = edistance_modele

           #handle.remove()

            #for h in handles:
            #    h.remove()

        #return rdms
    
    
    def as_df(self):
        df = pd.DataFrame()
        for layer, eigspec in self._pcs.items():
            layer_df = pd.DataFrame({'n': range(1, len(eigspec) + 1), 'variance': eigspec})
            layer_df = layer_df.assign(layer=layer)
            df = df.append(layer_df)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    #def get_image_paths(self) -> List[str]:
    #        raise NotImplementedError()
    
    
class EigenspectrumImageNet(EigenspectrumBase):
    
    # *num_classes=1000, num_per_class=10*
    def __init__(self, *args, num_classes=1000, num_per_class=1, **kwargs):
        super(EigenspectrumImageNet, self).__init__(*args, **kwargs,
                                                    stimuli_identifier='imagenet')
        assert 1 <= num_classes <= 1000 and 1 <= num_per_class <= 100
        self.num_classes = num_classes
        self.num_per_class = num_per_class
        self.image_paths = get_imagenet_val(num_classes=num_classes, num_per_class=num_per_class)

    def get_image_paths(self) -> List[str]:
        return self.image_paths
    
class EigenspectrumImageFolder(EigenspectrumBase):

    def __init__(self, data_dir, *args, num_images=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_images = num_images

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        image_paths = os.listdir(data_dir)
        if num_images is not None:
            assert len(image_paths) >= num_images
            image_paths = image_paths[:num_images]
        image_paths = [os.path.join(data_dir, file) for file in image_paths]
        self.image_paths = image_paths

    def get_image_paths(self) -> List[str]:
        return self.image_paths
    
class EigenspectrumMajajHong2015(EigenspectrumImageFolder):

    def __init__(self, *args, **kwargs):
        data_dir = os.getenv('BRAINIO_HOME', os.path.expanduser('~/.brainio'))
        data_dir = os.path.join(data_dir, 'image_dicarlo_hvm-public')
        assert os.path.exists(data_dir)
        super().__init__(data_dir, *args, **kwargs,
                         stimuli_identifier='dicarlo.hvm-public')

        self.image_paths = [p for p in self.image_paths if p[-4:] == '.png']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store eigenspectra of models')
    #parser.add_argument('--dataset', type=str,
    #                    choices=['imagenet', 'imagenet21k', 'object2vec', 'majajhong2015'],
    #                    help='Dataset of concepts for which to compute the eigenspectrum')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--pooling', dest='pooling', type=str, default=None,
                        choices=['max', 'none', 'layerPCA', 'zscore_PCA', 'maxpool_PCA', 'PCA', 'PCAtrans_zscore', 'PCA_components'],
                        help='Choose global max pooling, random projection, or layer PCA prior to computing RDMs')
    parser.add_argument('--seed', dest='seed', type=int, default=0,
                        help='Choose a random seed for analysis (torch and numpy)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Compute the eigenspectrum on grayscale inputs')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(data_dir=args.data_dir, seed=args.seed, pooling=args.pooling, grayscale=args.grayscale, debug=args.debug) #dataset=args.dataset,