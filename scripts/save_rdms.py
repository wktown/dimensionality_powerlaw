import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
#import torchlens as tl
from utils import timed#, id_to_properties
from torchvision.transforms import Grayscale
from custom_model_tools.hooks import GlobalMaxPool2d,  RandomProjection
from model_tools.activations.pca import LayerPCA
from custom_model_tools.layerPCA_modified import LayerPCA_Modified
from activation_models.generators import get_activation_models
from custom_model_tools.image_transform import ImageDatasetTransformer
from typing import Optional, List
from model_tools.utils import fullname
from tqdm import tqdm
from model_tools.activations.core import flatten
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from result_caching import store_dict
from utils import id_to_properties, get_imagenet_val
import matplotlib.pyplot as plt
import seaborn as sns

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
def main(dataset, data_dir, seed, pooling, grayscale, debug=False):
    image_transform = ImageDatasetTransformer('grayscale', Grayscale()) if grayscale else None
    
    if pooling=='max' or pooling=='none': #projections, spatial_PCA, random_spatial
        n_pcs = 'NA'
    elif pooling=='layerPCA' or pooling=='PCA_maxpool' or pooling=='PCA_zscore' or pooling == 'PCA':
        n_pcs = 1000
    
    inlcude_euclidean = False
    pearson_rdms = {}
    euclidean_rdms = {}
    for model, layers in get_activation_models(seed, n_pcs):
        properties = id_to_properties(model.identifier)
        
        pearson = get_rdms(dataset, data_dir, model, pooling, n_pcs, image_transform, d_metric='correlation')
        pearson.get(layers)
        if inlcude_euclidean:
            euclidean = get_rdms(dataset, data_dir, model, pooling, n_pcs, image_transform, d_metric='euclidean')
            euclidean.get(layers)
        
        key = properties['architecture']+'|'+properties['kind']
        pearson_rdms[key] = pearson._rdms
        if inlcude_euclidean:
            euclidean_rdms[key] = euclidean._rdms
        
        
        #for layer in layers:
        #    key = properties['task']+'|'+properties['kind']+'|'+properties['source']+'|'+'layer:'+layer
        #    pearson_rdms[key] = pearson._rdms[layer]
        #    euclidean_rdms[key] = euclidean._rdms[layer]
        
        if debug:
            break

    if not debug:
        
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
        elif pooling == 'PCA_maxpool':
           pool = f'maxpool_PCA:{n_pcs}'
        elif pooling == 'PCA_zscore':
            pool = f'Zscore_PCA:{n_pcs}'
        elif pooling == 'PCA':
            pool = f'PCA:{n_pcs}'
    
        
        #alphas = ['-0.2', '-0.6', '-1.0', '-2', '-3']
        alphas = ['-1.0']
        for model in pearson_rdms.keys():
            a = model.split('|')[1].split('_')[1]
            if a in alphas:
                architecture = model.split('|')[0]
                path = f'/home/wtownle1/dimensionality_powerlaw/figures/keaton/RDMs/PCxPC/{architecture}'
                if not os.path.exists(path):
                    os.makedirs(path)
                
                
                for layer_rdms in pearson_rdms[model].items():
                    layer = layer_rdms[0]
                    rdm = layer_rdms[1]
                    
                    fig, ax = plt.subplots(figsize=(10,10))
                    ax = sns.heatmap(rdm, square=True, cbar_kws={"shrink": .8})
                    #ax.set(title=f'Pearson RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
                    ax.set(title=f'RDM (1-pearson r|alpha:{a}|{pool})')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    plt.savefig(f'{path}/PearsonRDM|seed:{seed}|alpha={a}|layer:{layer}|{pool}|Eig.png')#, dpi=300)
                    #plt.savefig(f'{path}/PearsonRDM|alpha={a}|{pool}|.png')
                
                if inlcude_euclidean:
                    for layer_rdms in euclidean_rdms[model].items():
                        layer = layer_rdms[0]
                        rdm = layer_rdms[1]
                        
                        fig, ax = plt.subplots(figsize=(10,10))
                        ax = sns.heatmap(rdm, square=True, cbar_kws={"shrink": .8})
                        #ax.set(title=f'Euclidean RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
                        ax.set(title=f'RDM (euclidean|alpha:{a}|{pool})')
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        plt.savefig(f'{path}/EuclideanRDM|seed:{seed}|alpha={a}|layer:{layer}|{pool}|Eig.png')#, dpi=300)
                        #plt.savefig(f'{path}/EuclideanRDM|alpha={a}|{pool}|.png')
            
            
        #*np.save(f'results/RDM_{method}_pearson|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', pearson_rdms)
        #*np.save(f'results/RDM_{method}_euclidean|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', euclidean_rdms)
        
        
def get_rdms(dataset, data_dir, activations_extractor, pooling, n_pcs, image_transform, d_metric):
    if dataset == 'imagenet':
        return EigenspectrumImageNet(activations_extractor=activations_extractor,
                                     d_metric=d_metric,
                                     dataset=dataset,
                                     pooling=pooling,
                                     n_pcs=n_pcs,
                                     image_transform=image_transform)
        
    elif dataset == 'majajhong2015':
        return EigenspectrumMajajHong2015(activations_extractor=activations_extractor,
                                          d_metric=d_metric,
                                          dataset=dataset,
                                          pooling=pooling,
                                          n_pcs=n_pcs,
                                          image_transform=image_transform)
       
    elif dataset == 'imagenet21k':
        data_dir = data_dir
        #return EigenspectrumImageNet21k(data_dir=data_dir,
        #                                activations_extractor=activations_extractor,
        #                                pooling=pooling,
        #                                image_transform=image_transform)
        
    else:
        raise ValueError(f'Unknown eigenspectrum dataset: {dataset}')
    
        
class EigenspectrumBase:

    def __init__(self, activations_extractor, d_metric, dataset, pooling, n_pcs, stimuli_identifier=None,
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
        self._dataset = dataset
        self._rdms = {}
        
    def get(self, layers):
        transform_name = None if self._image_transform is None else self._image_transform.name
        
        self._rdms = self.rdms(identifier=self._extractor.identifier,
                                                        stimuli_identifier=self._stimuli_identifier,
                                                        layers=layers,
                                                        pooling=self._pooling,
                                                        n_pcs = self._n_pcs,
                                                        image_transform_name=transform_name)
        
        #np.save(f'results/modelRDM_{self._metric}|dataset:{self._dataset}|pooling:{self._pooling}|grayscale:{self._image_transform}', self._rdms)

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def rdms(self, identifier, stimuli_identifier, layers, pooling, n_pcs, image_transform_name):
        image_paths = self.get_image_paths()
        if self._image_transform is not None:
            image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        
        rdms = {}

        for layer in layers:
            if pooling == 'max':
                handle = GlobalMaxPool2d.hook(self._extractor)
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:max'
            elif pooling == 'none':
                handle = RandomProjection.hook(self._extractor)
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:none'
            elif pooling == 'layerPCA':
                handle = LayerPCA.hook(self._extractor, n_components=n_pcs)
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:layerPCA|n_components:{n_pcs}'
            elif pooling == 'PCA_maxpool':
                handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='max_pool')
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            elif pooling == 'PCA_zscore':
                handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='z_score')
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            elif pooling == 'PCA':
                handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='none')
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:PCA|n_components:{n_pcs}'
                
            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]
                
            logging.info(identifier)
            logging.info(layer)
                
            self._logger.debug('Retrieving stimulus activations')
            activations = self._extractor(image_paths, layers=[layer])
            print(activations.dims)
            print(activations.coords)
            print(activations.attrs)
            
            activations = activations.sel(layer=layer).values
            print(activations.shape)

            self._logger.debug('Computing RDMs')
            progress = tqdm(total=1, desc="RDMs")
            
            print('---activations---')
            print('nimages x npcs (x nfeatures?)')
            print(activations.shape)
            
            print('---flattened activations---')
            flat_activations = flatten(activations)
            print(flat_activations.shape)
            
            print('---correlations---')
            distance_vec = pdist(activations, metric=self._metric)
            distance_mat = squareform(distance_vec)
            print(distance_mat.shape)
            
            print('---flattened correlations---')
            flat_distance_vec = pdist(flat_activations, metric=self._metric)
            flat_distance_mat = squareform(flat_distance_vec)
            print(flat_distance_mat.shape)
            
            #if self._distance == 'pearson':
            #    p_model = np.corrcoef(activations)
            #    pdistance_model = 1 - p_model
            
            #if self._distance == 'euclidean':
            #    evec_model = pdist(activations, metric='euclidean')
            #    edistance_model = squareform(evec_model)
            
            
            
            #  *- if computing similarity between two rdms
            #corr, p = spearmanr(triu1, triu2)
            
            progress.update(1)
            progress.close()
            
            #key = identifier+layer
            rdms[layer] = distance_mat
            #euclidean_rdms[layer] = edistance_modele

            handle.remove()

            for h in handles:
                h.remove()

        return rdms
    
    
    def as_df(self):
        df = pd.DataFrame()
        for layer, eigspec in self._rdms.items():
            layer_df = pd.DataFrame({'n': range(1, len(eigspec) + 1), 'variance': eigspec})
            layer_df = layer_df.assign(layer=layer)
            df = df.append(layer_df)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    def get_image_paths(self) -> List[str]:
            raise NotImplementedError()
    
    
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
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec', 'majajhong2015'],
                        help='Dataset of concepts for which to compute the eigenspectrum')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--pooling', dest='pooling', type=str, default=None,
                        choices=['max', 'none', 'layerPCA', 'PCA_zscore', 'PCA_maxpool', 'PCA'],
                        help='Choose global max pooling, random projection, or layer PCA prior to computing RDMs')
    parser.add_argument('--seed', dest='seed', type=int, default=0,
                        help='Choose a random seed for analysis (torch and numpy)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Compute the eigenspectrum on grayscale inputs')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, seed=args.seed, pooling=args.pooling, grayscale=args.grayscale, debug=args.debug)