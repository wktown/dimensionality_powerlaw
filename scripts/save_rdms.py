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
def main(dataset, data_dir, seed, pooling, grayscale, n_pcs=1000, debug=False):
    image_transform = ImageDatasetTransformer('grayscale', Grayscale()) if grayscale else None
    
    pearson_rdms = {}
    euclidean_rdms = {}
    for model, layers in get_activation_models(seed, pooling, n_pcs=n_pcs):
        properties = id_to_properties(model.identifier)
        
        pearson = get_rdms(dataset, data_dir, model, pooling, image_transform, d_metric='correlation')
        pearson.get(layers)
        #pearson_rdms = {**pearson._rdms}
        euclidean = get_rdms(dataset, data_dir, model, pooling, image_transform, d_metric='euclidean')
        euclidean.get(layers)
        #euclidean_rdms = {**euclidean._rdms}   
        
        #task = properties['task']
        #alpha = properties['kind']
        #n_pcs = properties['source']
        
        #key_pearson = properties['architecture']+'_pearson'
        #key_euclidean = properties['architecture']+'_euclidean'
        #key = properties['architecture']
        #model_rdms[key] = pearson_rdms
        #model_rdms[key] = euclidean_rdms
        
        key = properties['architecture']+'|'+properties['kind']
        pearson_rdms[key] = pearson._rdms
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
        method = properties['task'].split('_')[0]
        if method == 'Eig':
            m = 'X_transformed'
        elif method == 'SVD':
            m = 'SVD'
        
        if pooling == 'layerPCA':
            pool = f'layerPCA:{n_pcs}'
        elif pooling == 'max':
            pool = f'pooling:{pooling}'
        elif pooling == 'max_PCA':
            pool = f'maxpool_PCA:{n_pcs}'
        
        alphas = ['-0.3', '-0.6', '-1.0', '-2', '-3']
        for model in pearson_rdms.keys():
            a = model.split('|')[1].split('_')[1]
            if a in alphas:
                architecture = model.split('|')[0]
                path = f'/home/wtownle1/dimensionality_powerlaw/figures/keaton/RDMs/{architecture}'
                if not os.path.exists(path):
                    os.makedirs(path)
                
                for layer_rdms in pearson_rdms[model].items():
                    layer = layer_rdms[0]
                    rdm = layer_rdms[1]
                    
                    fig, ax = plt.subplots(figsize=(10,10))
                    ax = sns.heatmap(rdm, square=True, cbar_kws={"shrink": .8})
                    ax.set(title=f'Pearson RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    plt.savefig(f'{path}/PearsonRDM|seed:{seed}|alpha={a}|layer:{layer}|{pool}|{m}.png')#, dpi=300)
                    
                for layer_rdms in euclidean_rdms[model].items():
                    layer = layer_rdms[0]
                    rdm = layer_rdms[1]
                    
                    fig, ax = plt.subplots(figsize=(10,10))
                    ax = sns.heatmap(rdm, square=True, cbar_kws={"shrink": .8})
                    ax.set(title=f'Euclidean RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    plt.savefig(f'{path}/EuclideanRDM|seed:{seed}|alpha={a}|layer:{layer}|{pool}|{m}.png')#, dpi=300)
            
            
            #key_plot = 'AtlasNet|'+properties['task']+f'|a_{a}|pcs_{pcs}|layer:c2'
            #rdm_pearson = pearson_rdms[key_plot]
            #rdm_euclidean = euclidean_rdms[key_plot]
            #fig ...
            
        #*np.save(f'results/RDM_{method}_pearson|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', pearson_rdms)
        #*np.save(f'results/RDM_{method}_euclidean|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', euclidean_rdms)
        
        
def get_rdms(dataset, data_dir, activations_extractor, pooling, image_transform, d_metric):
    if dataset == 'imagenet':
        return EigenspectrumImageNet(activations_extractor=activations_extractor,
                                     d_metric=d_metric,
                                     dataset=dataset,
                                     pooling=pooling,
                                     image_transform=image_transform)
        
    elif dataset == 'majajhong2015':
        return EigenspectrumMajajHong2015(activations_extractor=activations_extractor,
                                          d_metric=d_metric,
                                          dataset=dataset,
                                          pooling=pooling,
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

    def __init__(self, activations_extractor, d_metric, dataset, pooling=True, stimuli_identifier=None,
                 image_transform: Optional[ImageDatasetTransformer] = None,
                 hooks: Optional[List] = None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
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
                                                        image_transform_name=transform_name)
        
        #np.save(f'results/modelRDM_{self._metric}|dataset:{self._dataset}|pooling:{self._pooling}|grayscale:{self._image_transform}', self._rdms)

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def rdms(self, identifier, stimuli_identifier, layers, pooling, image_transform_name):
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
            elif pooling == 'none':
                handle = RandomProjection.hook(self._extractor)
            elif pooling == 'layerPCA':
                n_pcs = 1000
                handle = LayerPCA.hook(self._extractor, n_components=n_pcs)
                
            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]
                
            logging.info(identifier)
            logging.info(layer)
                
            self._logger.debug('Retrieving stimulus activations')
            activations = self._extractor(image_paths, layers=[layer])
            activations = activations.sel(layer=layer).values
            print(activations.shape)

            self._logger.debug('Computing RDMs')
            progress = tqdm(total=1, desc="RDMs")
            
            activations = flatten(activations)
            
            dvec_model = pdist(activations, metric=self._metric)
            dsquare_model = squareform(dvec_model)
            
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
            rdms[layer] = dsquare_model
            #euclidean_rdms[layer] = edistance_model

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

    def __init__(self, *args, num_classes=1000, num_per_class=10, **kwargs):
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
                        choices=['max', 'none', 'layerPCA'],
                        help='Choose global max pooling, random projection, or layer PCA prior to computing RDMs')
    parser.add_argument('--seed', dest='seed', type=int, default=0,
                        help='Choose a random seed for analysis (torch and numpy)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Compute the eigenspectrum on grayscale inputs')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, seed=args.seed, pooling=args.pooling, grayscale=args.grayscale, debug=args.debug)