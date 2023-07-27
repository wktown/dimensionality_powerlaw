import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import torch
import torchvision
import torchlens as tl
from utils import timed, id_to_properties
from torchvision.transforms import Grayscale
from custom_model_tools.hooks import GlobalMaxPool2d,  RandomProjection
from activation_models.generators import get_activation_models
from custom_model_tools.image_transform import ImageDatasetTransformer
from typing import Optional, List
from model_tools.utils import fullname
from tqdm import tqdm
from model_tools.activations.core import flatten
from scipy.stats import spearmanr

import logging
logging.basicConfig(level=logging.INFO)



###save_rdms with functions instead of classes###
def get_rdms(dataset, data_dir, activations_extractor, layers, pooling, image_transform):
    if dataset == 'majajhong2015':
        
        data_dir = os.getenv('BRAINIO_HOME', os.path.expanduser('~/.brainio'))
        data_dir = os.path.join(data_dir, 'image_dicarlo_hvm-public')
        assert os.path.exists(data_dir)
        stimuli_identifier='dicarlo.hvm-public'
        identifier = activations_extractor.identifier
        
        def image_folder(data_dir, num_images=None):
            
            #self.num_images = num_images

            assert os.path.exists(data_dir)
            #self.data_dir = data_dir

            image_paths = os.listdir(data_dir)
            if num_images is not None:
                assert len(image_paths) >= num_images
                image_paths = image_paths[:num_images]
            image_paths = [os.path.join(data_dir, file) for file in image_paths]
            return image_paths

        #def get_image_paths(self) -> List[str]:
        #    return self.image_paths
        
        im_pths = image_folder(data_dir)
        
        image_paths = [p for p in im_pths if p[-4:] == '.png']
        
        #seems like this is just to set self.imagepaths?
        pearson_rdm, euclidean_rdm = rdms(activations_extractor, identifier, stimuli_identifier, layers, pooling, image_transform)
        
    return pearson_rdm, euclidean_rdm






#get model activations from majaj images (multiple powerlaw slopes in eigenspectrum)
#get brain data
# - create/plot RDMs

# - project model activations onto PCs (transformed)
#     - do RSA & encoding model predictions with various numbers of PCs (top 10, 50, 100, etc)
#     - plot rsa/encoding performance vs. number of PCs
# - same as above but z-score pc data first (could be similar to variance scale in eig script)

@timed
def main(dataset, data_dir, pooling, grayscale, debug=False):
    image_transform = ImageDatasetTransformer('grayscale', Grayscale()) if grayscale else None
    #pearson_rdm_df = pd.DataFrame()
    #euclidean_rdm_df = pd.DataFrame()
    for model, layers in get_activation_models():
        pearson_rdm, euclidean_rdm = get_rdms(dataset, data_dir, model, pooling, image_transform)
        #pearson_rdm_df = pearson_rdm_df.append(pearson_rdm.as_df())
        #euclidean_rdm_df = euclidean_rdm_df.append(euclidean_rdm.as_df())
        if debug:
            break

    if not debug:
        np.save(f'results/modelRDM_pearson|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}.csv', pearson_rdm)
        np.save(f'results/modelRDM_euclidean|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}.csv', euclidean_rdm)
        
        
        
@store_dict(dict_key='layers', identifier_ignore=['layers'])
def get_rdms(self, identifier, stimuli_identifier, layers, pooling, image_transform_name):
    image_paths = self.get_image_paths()
    if self._image_transform is not None:
        image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)

    # Compute activations and PCA for every layer individually to save on memory.
    # This is more inefficient because we run images through the network several times,
    # but it is a more scalable approach when using many images and large layers.
    
    pearson_rdms = {}
    euclidean_rdms = {}

    for layer in layers:
        if pooling == 'max':
            handle = GlobalMaxPool2d.hook(self._extractor)
        elif pooling == 'none':
            handle = RandomProjection.hook(self._extractor)
            
        handles = []
        if self._hooks is not None:
            handles = [cls.hook(self._extractor) for cls in self._hooks]
            
        self._logger.debug('Retrieving stimulus activations')
        activations = self._extractor(image_paths, layers=[layer])
        activations = activations.sel(layer=layer).values
        print(activations.shape)

        self._logger.debug('Computing principal components')
        progress = tqdm(total=1, desc="layer principal components")
        
        activations = flatten(activations)
        p_model = np.corrcoef(activations)
        pdistance_model = 1 - p_model
        
        evec_model = pdist(activations, metric='euclidean')
        edistance_model = squareform(evec_model)
        
        #corr, p = spearmanr(triu1, triu2)
        
        progress.update(1)
        progress.close()

        pearson_rdms[layer] = pdistance_model
        euclidean_rdms[layer] = edistance_model

        handle.remove()

        for h in handles:
            h.remove()

    return pearson_rdms, euclidean_rdms
    
    
    
    
    
    
        
        
class EigenspectrumBase:

    def __init__(self, activations_extractor, pooling=True, stimuli_identifier=None,
                 image_transform: Optional[ImageDatasetTransformer] = None,
                 hooks: Optional[List] = None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._hooks = hooks
        self._stimuli_identifier = stimuli_identifier
        self._image_transform = image_transform
        self._layer_eigenspectra = {}

    def fit(self, layers):
        transform_name = None if self._image_transform is None else self._image_transform.name
        self._layer_eigenspectra = self._fit(identifier=self._extractor.identifier,
                                             stimuli_identifier=self._stimuli_identifier,
                                             layers=layers,
                                             pooling=self._pooling,
                                             image_transform_name=transform_name)
        
    # **save eigenspectra for plotting RDMs**
    # - just the max number of PCs (i.e. normal spectra?)
    # - just for majaj?
    np.save(self._layer_eigenspectra)
    
    ###
    
    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, stimuli_identifier, layers, pooling, image_transform_name):
        image_paths = self.get_image_paths()
        if self._image_transform is not None:
            image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        
        layer_eigenspectra = {}

        for layer in layers:
            if pooling == 'max':
                handle = GlobalMaxPool2d.hook(self._extractor)
            elif pooling == 'none':
                handle = RandomProjection.hook(self._extractor)
                
            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]
                
            self._logger.debug('Retrieving stimulus activations')
            activations = self._extractor(image_paths, layers=[layer])
            activations = activations.sel(layer=layer).values

            self._logger.debug('Computing principal components')
            progress = tqdm(total=1, desc="layer principal components")
            activations = flatten(activations)
            pca = PCA(random_state=0)
            pca.fit(activations)
            eigenspectrum = pca.explained_variance_
            progress.update(1)
            progress.close()

            layer_eigenspectra[layer] = eigenspectrum

            handle.remove()

            for h in handles:
                h.remove()

        return layer_eigenspectra

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


class EigenspectrumNestedImageFolder(EigenspectrumBase):

    def __init__(self, data_dir, *args, num_folders=None, num_per_folder=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_folders = num_folders
        self.num_per_folder = num_per_folder

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        image_paths = []
        cats = os.listdir(self.data_dir)
        if num_folders is not None:
            assert len(cats) >= num_folders
            cats = cats[:num_folders]
        for cat in cats:
            cat_dir = os.path.join(data_dir, cat)
            files = os.listdir(cat_dir)
            if num_per_folder is not None:
                assert len(files) >= num_per_folder
                files = files[:num_per_folder]
            paths = [os.path.join(cat_dir, file) for file in files]
            image_paths += paths
        self.image_paths = image_paths

    def get_image_paths(self) -> List[str]:
        return self.image_paths


class EigenspectrumImageNet21k(EigenspectrumNestedImageFolder):

    def __init__(self, data_dir, *args, num_classes=996, num_per_class=10, **kwargs):
        super().__init__(data_dir, *args, num_folders=num_classes, num_per_folder=num_per_class, **kwargs,
                         stimuli_identifier='imagenet21k')

class EigenspectrumMajajHong2015(EigenspectrumImageFolder):

    def __init__(self, *args, **kwargs):
        data_dir = os.getenv('BRAINIO_HOME', os.path.expanduser('~/.brainio'))
        data_dir = os.path.join(data_dir, 'image_dicarlo_hvm-public')
        assert os.path.exists(data_dir)
        super().__init__(data_dir, *args, **kwargs,
                         stimuli_identifier='dicarlo.hvm-public')

        self.image_paths = [p for p in self.image_paths if p[-4:] == '.png']
        
        


def get_eigenspectrum(dataset, data_dir, activations_extractor, pooling, image_transform):
    if dataset == 'imagenet':
        return EigenspectrumImageNet(activations_extractor=activations_extractor,
                                     pooling=pooling,
                                     image_transform=image_transform)
    elif dataset == 'imagenet21k':
        return EigenspectrumImageNet21k(data_dir=data_dir,
                                        activations_extractor=activations_extractor,
                                        pooling=pooling,
                                        image_transform=image_transform)
    elif dataset == 'majajhong2015':
        return EigenspectrumMajajHong2015(activations_extractor=activations_extractor,
                                          pooling=pooling,
                                          image_transform=image_transform)
    else:
        raise ValueError(f'Unknown eigenspectrum dataset: {dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store eigenspectra of models')
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec', 'majajhong2015'],
                        help='Dataset of concepts for which to compute the eigenspectrum')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--pooling', dest='pooling', type=str, default=None,
                        choices=['max', 'avg', 'none', 'spatial_pca', 'random_spatial'],
                        help='Choose global max pooling, avg pooling, no pooling, to select one random spatial position, or to compute the eigenspectrum at each spatial position in the final layer(s) of the model prior to computing the eigenspectrum')
    parser.add_argument('--grayscale', action='store_true',
                        help='Compute the eigenspectrum on grayscale inputs')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, pooling=args.pooling, grayscale=args.grayscale, debug=args.debug)
    
        
        
        
        
        
        
        
        



@timed
def main(benchmark, pooling, debug=False):
    save_path = f'results/rsa_Eig-Vscale|benchmark:{benchmark._identifier}|pooling:{pooling}.csv'
    if os.path.exists(save_path):
        print(f'Results already exists: {save_path}')
        return
    
    scores = pd.DataFrame()
    for model, layers in get_activation_models():
        layer_scores = fit_rsa(benchmark, model, layers, pooling)
        scores = scores.append(layer_scores)
        if debug:
            break
    if not debug:
        scores.to_csv(save_path, index=False)












from brainscore.metrics.rdm import RDM

rdm = RDM(neuroid_dim='neuroid')

assembly1 = 1 #network activations
assembly2 = 2 #brain data

rdm1 = rdm(assembly1)
rdm2 = rdm(assembly2)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
#from sklearn.manifold import MDS
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#%matplotlib inline
sns.set(style = 'white', context='poster', rc={'lines.linewidth': 2.5})
sns.set(palette="colorblind")

data = np.load('data.npy',allow_pickle=True).item() # A data dictonary containing all the data needed


#pearson correlation rdms
p_mri = np.corrcoef(data['fMRI'])
d_mri = 1 - p_mri
p_model = np.corrcoef(data['model1'])
d_model = 1 - p_model
#sns.heatmap(d_mri)

#euclidean distance rdms
evec_mri = pdist(data['fMRI'], metric='euclidean')
e_mri = squareform(evec_mri)
evec_model = pdist(data['model1'], metric='euclidean')
e_model = squareform(evec_model)
#plt.figure()
#sns.heatmap(rsm_fmri)
#plt.show()


#vectorize np correalation RDMs (use evec_mri/model for scipy euclidean distance)
pl_mri = np.tril(d_mri)
pl_mri = np.asarray(pl_mri)
pl_mri = pl_mri[pl_mri > 0.000000002]
pvec_mri = np.ndarray.flatten(pl_mri)
pl_model = np.tril(d_model)
pl_model = np.asarray(pl_model)
pl_model = pl_model[pl_model > 0.000000002]
pvec_model = np.ndarray.flatten(pl_model)

#Correlation between RDMs
p_mri_model = np.corrcoef(pvec_mri, pvec_model)
p_mri_model = p_mri_model[0,1]
print(p_mri_model)

e_mri_model = np.corrcoef(evec_mri, evec_model)
e_mri_model = e_mri_model[0,1]
print(e_mri_model)

