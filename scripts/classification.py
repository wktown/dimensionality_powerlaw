# classification at bottom (after local libraries 'imported')

#***
from model_features.models.expansion_3_layers import Expansion
#***

#libraries
import os
import pandas as pd
import sys
import pickle
import random
random.seed(42)
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import top_k_accuracy_score as top_k
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import svm
from sklearn.model_selection import cross_val_predict
import random
import numpy as np
from scipy.special import softmax
import torchvision
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
import functools
import gc

from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import warnings
warnings.warn('my warning')
warnings.filterwarnings('ignore')
from collections import OrderedDict
import xarray as xr
SUBMODULE_SEPARATOR = '.'
from torch.autograd import Variable
from torch import nn


#vars
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
#***
PLACES_IMAGES = '/home/atlask/data/atlas/datasets/places' #***
#***
CACHE_DIR = '/home/atlask/data/atlas'
CACHE = os.path.join(CACHE_DIR,'.cache')
PATH_TO_PCA = os.path.join(CACHE,'pca')



# -- image loading --

import os
import pandas as pd
import sys
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
#***
PLACES_IMAGES = '/home/atlask/data/atlas/datasets/places' #***
#***


def load_places_images():
    #***
    #PLACES_IMAGES = '/home/atlask/data/atlas/datasets/places'
    #***
    """
    Loads the file paths of validation images from the PLACES_IMAGES directory.

    Returns:
        list: A sorted list of full paths to the validation images.
    """
    val_images = os.listdir(os.path.join(PLACES_IMAGES,'val_images/val_256'))
    val_images_paths = [f'{PLACES_IMAGES}/val_images/val_256/{i}' for i in val_images]
    
    return sorted(val_images_paths)


def load_image_paths(name): 
    
    """
    Load image file paths based on a specified dataset name.

    Args:
        name (str): Name of the dataset ('naturalscenes', 'majajhong', or 'places').

    Returns:
        list: A sorted list of full paths to the images for the specified dataset.
    """
    
    match name:
        
        case 'naturalscenes':
            return load_nsd_images()

        case 'majajhong':
            return load_majaj_images()

        case 'places':
            return load_places_images()
    

def get_image_labels(dataset, images):
    
    
    """
    Get image labels based on a specified dataset.

    Args:
        dataset (str): Name of the dataset ('naturalscenes', 'majajhong', or 'places').
        images (list): List of image file paths for which to obtain labels.

    Returns:
        list: List of labels corresponding to the provided images.
    """
    
    match dataset:
        
        case 'naturalscenes':
            return [os.path.basename(i).strip('.png') for i in images]

        
        case 'majajhong':
            from config import MAJAJ_NAME_DICT 
            name_dict = pd.read_csv(MAJAJ_NAME_DICT).set_index('image_file_name')['image_id'].to_dict()
            return [name_dict[os.path.basename(i)] for i in images]


        case 'places':
            return [os.path.basename(i) for i in images]

def load_places_cat_labels():
    #from config import PLACES_IMAGES
    """
    Load category labels for placees dataset.

    Returns:
        dict: Dictionary where keys are image filenames and values are category labels.
    """    
    with open(os.path.join(PLACES_IMAGES,'places365_val.txt'), "r") as file:
        content = file.read()
    annotations = content.split('\n')
    cat_dict = {}
    for annotation in annotations:
        image = annotation.split(' ')[0]
        cat = annotation.split(' ')[1]
        cat_dict[image] = int(cat)
    return cat_dict
    
def load_places_cat_names():
    """
    Load category names for the places dataset.

    Returns:
        list: List of category names for the validation images in the PLACES_IMAGES dataset.
    """    
    val_image_paths = load_places_images()
    val_image_names = [os.path.basename(i) for i in val_image_paths]
    cat_dict = load_places_cat_labels()

    return [cat_dict[i] for i in val_image_names]   



# -- image loading --
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
#import os
#import sys
#import os 
#import sys
import functools
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle

#sys.path.append(os.getenv('BONNER_ROOT_PATH'))
#from config import CACHE


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                print('loading processed images...')
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result, f)
            return result

        return wrapper
    return decorator


class ImageProcessor:
    """
    A utility class to preprocess and transform images. It includes caching functionalities to avoid
    repetitive computations.

    Attributes:
        device (torch.device): The device to which tensors should be sent.
        batch_size (int, optional): Number of samples per batch of computation. Defaults to 100.
    """
    
    def __init__(self, device, batch_size = 100):
                
        self.device = device
        self.batch_size = batch_size
        self.im_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
  

        
        if not os.path.exists(os.path.join(CACHE,'preprocessed_images')):
            os.mkdir(os.path.join(CACHE,'preprocessed_images'))
        
        
    @staticmethod
    def cache_file(image_paths, dataset):
        name = f'{dataset}_num_images={len(image_paths)}'
        return os.path.join('preprocessed_images',name)

    
    @cache(cache_file)
    def process(self, image_paths, dataset):        
        """
        Process and transform a list of images.

        Args:
            image_paths (list): List of image file paths.
            dataset (str): Dataset name.

        Returns:
            torch.Tensor: Tensor containing the processed images.
        """
        print('processing images...')
        dataset = TransformDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return torch.cat([batch for batch in tqdm(dataloader)],dim=0)
    

    def process_batch(self, image_paths, dataset):
        """
        Process a batch of images without using cache.

        Args:
            image_paths (list): List of image file paths.
            dataset (str): Dataset name.

        Returns:
            torch.Tensor: Tensor containing the processed images.
        """
        dataset = TransformDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        return torch.cat([batch for batch in dataloader],dim=0)


class TransformDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Convert image to RGB
        
        if self.transform:
            img = self.transform(img)
        
        return img


#-- _config --

import pickle
#import os
#import sys
#from image_tools.loading import load_places_cat_names, load_places_cat_labels
#from config import PLACES_IMAGES
#PLACES_IMAGES = '/home/atlask/data/atlas/datasets/places' # places 

import random
random.seed(42)


CAT_SUBSET_PATH = os.path.join(PLACES_IMAGES,'categories_subset_100')
if not os.path.exists(PLACES_IMAGES):
    print('generating a subset of 100 categories')
    with open(os.path.join(PLACES_IMAGES,'categories_places365.txt'),'r') as f:
        categories = f.read().split('\n/') 
    
    CAT_SUBSET = random.sample(range(0, len(num_categories)), 100)
    with open(CAT_SUBSET_PATH,'wb') as f:
        pickle.dump(CAT_SUBSET,f)
                            
else:
    with open(CAT_SUBSET_PATH,'rb') as f:
        CAT_SUBSET = pickle.load(f)


CAT_LABELS = load_places_cat_labels()
CAT_LABELS_SUBSET = {k: v for k, v in CAT_LABELS.items() if v in CAT_SUBSET}
VAL_IMAGES_SUBSET = list(CAT_LABELS_SUBSET.keys())
CAT_NAMES = load_places_cat_names()



#-- tools --

#libraries
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import top_k_accuracy_score as top_k
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestCentroid
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm
import random
import numpy as np
from scipy.special import softmax
import torchvision
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
#import pickle
#import sys
#import os
import functools

# local vars
#sys.path.append(os.getenv('BONNER_ROOT_PATH'))
#from image_tools.loading import load_places_cat_labels
#from model_evaluation.image_classification._config import CAT_SUBSET
#from config import CACHE


def create_splits(n: int, num_folds: int = 5, shuffle: bool = True): 
    
    random.seed(0)
    if shuffle:
        indices = np.arange(0,n)
        random.shuffle(indices)
    else:
        indices = np.arange(0,n)

    x = np.array_split(indices, num_folds)
    return x


class NearestCentroidDistances(NearestCentroid):
    def predict_distances(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        distances = pairwise_distances(X, self.centroids_, metric=self.metric)
        return distances
    

def prototype_performance(X_train, y_train, X_test, y_test):
        model = NearestCentroidDistances()
        model.fit(X_train, y_train)
        y_pred = model.predict_distances(X_test)
        y_pred = softmax(-y_pred, axis=1)   
        y_pred = np.argmax(y_pred, axis=1)

        return accuracy_score(y_test, y_pred)


def get_Xy(data, categories):
    
    cat_labels = load_places_cat_labels()
    cat_labels_subset = {k: v for k, v in cat_labels.items() if v in categories}
    images_subset = list(cat_labels_subset.keys())

    data_subset = data.sel(stimulus_id = images_subset).x.values
    labels_subset = np.array([cat_labels_subset[i] for i in images_subset])
    
    encoder = LabelEncoder()
    labels_subset = encoder.fit_transform(labels_subset)
    
    return data_subset, labels_subset


def cv_performance(X, y, num_folds=5):
    
    splits = create_splits(n = len(X), shuffle = True, num_folds=num_folds)
    accuracy = []
    
    for indices_test in splits:

        indices_train = np.setdiff1d(np.arange(0, len(X)), np.array(indices_test))
        
        X_train, y_train = X[indices_train,...], y[indices_train,...]
        X_test, y_test = X[indices_test,...], y[indices_test,...]

        accuracy_score = prototype_performance(X_train, y_train, X_test, y_test)
        accuracy.append(accuracy_score)
    
    return sum(accuracy)/len(accuracy) 


def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                return 
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result,f)
            print('classification results are saved in cache')
            return 

        return wrapper
    return decorator


class PairwiseClassification():
    
    def __init__(self):
        
        if not os.path.exists(os.path.join(CACHE,'classification')):
            os.mkdir(os.path.join(CACHE,'classification'))

    @staticmethod
    def cache_file(iden, data):
        return os.path.join('classification',iden)

    
    @cache(cache_file)
    def get_performance(self, iden, data):
    
        performance_dict = {}
        pairs = []

        for cat_1 in tqdm(CAT_SUBSET):
            for cat_2 in CAT_SUBSET:

                if {cat_1, cat_2} in pairs:
                    pass

                elif cat_1 == cat_2:
                    performance_dict[(cat_1,cat_2)] = 1

                else:
                    X, y = get_Xy(data, [cat_1,cat_2])
                    performance_dict[(cat_1,cat_2)] = cv_performance(X, y)
                    pairs.append({cat_1, cat_2})

        return performance_dict


def normalize(X):
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X



# -- utils --
def get_activations_iden(model_info, dataset, hook=None):
    
        model_name = model_info['iden'] 
        
        activations_identifier = model_name + '_' + f'{model_info["num_layers"]}_layers' + '_' + f'{model_info["num_features"]}_features' 

        try:
            if model_info['hook'] == 'pca':
                return activations_identifier + '_' + dataset + '_principal_components'
            else:
                print('invalid hook')
                return
            
        except KeyError:
                return activations_identifier + '_' + dataset 



# -- model feature utils --

#import sys
#import os
#ROOT = os.getenv('BONNER_ROOT_PATH')
#sys.path.append(ROOT)
#from config import CACHE 
#import functools
#import pickle
#import torch
import gc

def register_pca_hook(x, PCA_FILE_NAME, n_components=256, device='cuda'):
    
    with open(PCA_FILE_NAME, 'rb') as file:
        _pca = pickle.load(file)
    _mean = torch.Tensor(_pca.mean_).to(device)
    _eig_vec = torch.Tensor(_pca.components_.transpose()).to(device)
    x = x.squeeze()
    x -= _mean
    
    return x @ _eig_vec[:, :n_components]


def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                return 
            
            result = func(self, *args, **kwargs)
            result.to_netcdf(cache_path)
            gc.collect()
            return 

        return wrapper
    return decorator



# -- activations_extractor --

import warnings
warnings.warn('my warning')
from collections import OrderedDict
import xarray as xr
import numpy as np
SUBMODULE_SEPARATOR = '.'
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
import pickle
import sys
import functools
#ROOT = os.getenv('BONNER_ROOT_PATH')
#sys.path.append(ROOT)
#from image_tools.loading import load_image_paths, get_image_labels
#from image_tools.processing import ImageProcessor
#from config import CACHE 
#from model_features.utils import cache, register_pca_hook

PATH_TO_PCA = os.path.join(CACHE,'pca')

class PytorchWrapper:
    def __init__(self, model, identifier, device, forward_kwargs=None): 
        self._device = device
        self._model = model
        self._model = self._model.to(self._device)
        self._forward_kwargs = forward_kwargs or {}
        self.identifier = identifier

    def get_activations(self, images, layer_names, _hook):

        images = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results, _hook=_hook)
            hooks.append(hook)

        with torch.no_grad():
            self._model(images, **self._forward_kwargs)
        for hook in hooks:
            hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        if layer_name == 'logits':
            return self._output_layer()
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def _output_layer(self):
        module = self._model
        while module._modules:
            module = module._modules[next(reversed(module._modules))]
        return module

    @classmethod
    def _tensor_to_numpy(cls, output):
        try:
            return output.cpu().data.numpy()
        except AttributeError:
            return output
            

    def register_hook(self, layer, layer_name, target_dict, _hook):
        def hook_function(_layer, _input, output, _hook = _hook, name=layer_name):
            
            if _hook is None:
                target_dict[name] = output
                
            elif _hook == 'pca':
                target_dict[name] = register_pca_hook(output, os.path.join(PATH_TO_PCA, f'{self.identifier}'))

        hook = layer.register_forward_hook(hook_function)
        return hook 


    def __repr__(self):
        return repr(self._model)    
    

def batch_activations(model: nn.Module, 
                      image_labels: list,
                      layer_names:list, 
                      _hook: str,
                      device=str,
                      dataset=None,
                      image_paths: list=None,
                      images: torch.Tensor=None,
                      batch_size:int=None) -> xr.Dataset:

            
        if image_paths is not None:
            images = ImageProcessor(device=device, batch_size=batch_size).process_batch(image_paths=image_paths, 
                                                                 dataset=dataset,
                                                                 )

            
            
        activations_dict = model.get_activations(images = images, 
                                                 layer_names = layer_names, 
                                                 _hook = _hook)
        activations_final = []
    
                             
        for layer in layer_names:
            activations_b = activations_dict[layer]
            activations_b = activations_b.reshape(activations_dict[layer].shape[0],-1)
            ds = xr.Dataset(
            data_vars=dict(x=(["presentation", "features"], activations_b.cpu())),
            coords={'stimulus_id': (['presentation'], image_labels)})

            activations_final.append(ds)     
        
        
        activations_final_all = xr.concat(activations_final,dim='presentation') 
        
        return activations_final_all


class Activations:
    
    def __init__(self,
                 model: nn.Module,
                 layer_names: list,
                 dataset: str,
                 hook:str = None,
                 device:str= 'cuda',
                 batch_size: int = 64,
                 compute_mode:str='fast'):
        
        
        self.model = model
        self.layer_names = layer_names
        self.dataset = dataset
        self.batch_size = batch_size
        self.hook = hook
        self.device = device
        self.compute_mode = compute_mode
        
        assert self.compute_mode in ['fast','slow'], "invalid compute mode, please choose one of: 'fast', 'slow'"

        if not os.path.exists(os.path.join(CACHE,'activations')):
            os.mkdir(os.path.join(CACHE,'activations'))
     
        
    @staticmethod
    def cache_file(iden):
        return os.path.join('activations',iden)

    
    @cache(cache_file)
    def get_array(self,iden):       
                
        wrapped_model = PytorchWrapper(model = self.model, identifier = iden, device=self.device)
        image_paths = load_image_paths(name = self.dataset)
        labels = get_image_labels(self.dataset, image_paths)
        
        if self.compute_mode=='fast':
                
                images = ImageProcessor(device=self.device).process(image_paths=image_paths, 
                                                                    dataset=self.dataset)

                print('extracting activations...')

                i = 0   
                ds_list = []
                pbar = tqdm(total = len(image_paths)//self.batch_size)

                while i < len(image_paths):

                    batch_data_final = batch_activations(model=wrapped_model,
                                                        images=images[i:i+self.batch_size, :],
                                                        image_labels=labels[i:i+self.batch_size],
                                                        layer_names = self.layer_names,
                                                        _hook = self.hook,
                                                        device=self.device,
                                                        )

                    ds_list.append(batch_data_final)    
                    i += self.batch_size
                    pbar.update(1)

                pbar.close()

        else:
            
                print('processing images and extracting activations ...')

                i = 0   
                ds_list = []
                pbar = tqdm(total = len(image_paths)//self.batch_size)

                while i < len(image_paths):

                    batch_data_final = batch_activations(model=wrapped_model,
                                                        image_paths=image_paths[i:i+self.batch_size],
                                                        image_labels=labels[i:i+self.batch_size],
                                                        layer_names = self.layer_names,
                                                        dataset=self.dataset,
                                                        batch_size=self.batch_size,
                                                        _hook = self.hook,
                                                        device=self.device)

                    ds_list.append(batch_data_final)    
                    i += self.batch_size
                    pbar.update(1)

                pbar.close()        
        
        data = xr.concat(ds_list,dim='presentation')
        
        print('model activations are saved in cache')
        return data


# -- model dict --

from model_features.models.expansion_3_layers import Expansion
#import torchvision
#ROOT = os.getenv('BONNER_ROOT_PATH')
#warnings.filterwarnings('ignore')

def load_model_dict(name):
    
    match name:

        case 'expansion_10':
            return {
                    'iden':'expansion_model',
                    'model':Expansion(filters_3=10).Build(),
                    'layers': ['last'], 
                    'num_layers':3,
                    'num_features':10}
            
        case 'expansion_1000': 
            return {
                    'iden':'expansion_model',
                    'model':Expansion(filters_3=1000).Build(),
                    'layers': ['last'], 
                    'num_layers':3,
                    'num_features':1000
                    }



# -- classification --
#import warnings
#warnings.filterwarnings('ignore')

# libraries
#import sys
#import os
#from tqdm import tqdm
#import pickle
#import xarray as xr

# local libraries
#sys.path.append(os.getenv('BONNER_ROOT_PATH'))
#from model_evaluation.image_classification._config import VAL_IMAGES_SUBSET
#from model_evaluation.image_classification.tools import PairwiseClassification, normalize
#from model_evaluation.utils import get_activations_iden
#from model_features.activation_extractor import Activations

#from config import CACHE
#**CACHE_DIR = '/home/' **
CACHE = os.path.join(CACHE_DIR,'.cache')

# models
#from model_features.models.models import load_model_dict

# local vars
DATASET = 'places'
HOOK = None



# define models in a dict
models = ['expansion_10000'] #, 'alexnet_conv5']

for model_name in models:
    
    print(model_name)
    model_info = load_model_dict(model_name)

    activations_iden = get_activations_iden(model_info=model_info, dataset=DATASET)
    
    activations = Activations(model=model_info['model'],
                            layer_names=model_info['layers'],
                            dataset=DATASET,
                            hook = HOOK,
                            batch_size = 50,
                            compute_mode = 'slow').get_array(activations_iden) 
    
    data = xr.open_dataset(os.path.join(CACHE,'activations',activations_iden))
    
    # normalize activations for image classification
    data.x.values = normalize(data.x.values)
    
    # take the subset of activations belonging to the 100 categories of images
    data = data.set_xindex('stimulus_id')
    data_subset = data.sel(stimulus_id = VAL_IMAGES_SUBSET)

    # get pairwise classification performance
    PairwiseClassification().get_performance(iden = activations_iden, 
                                            data = data_subset)