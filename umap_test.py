import os
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from umap import UMAP
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

sns.set_theme(
    context="notebook",
    style="white",
    palette="deep",
)

#majaj images: 
# /home/wtownle1/data-mbonner5/shared/brainscore/brainio/image_dicarlo_hvm-public
# /home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/image_dicarlo_hvm-public
#      - face0007_rx-20.231_ry-35.402_rz+14.046_tx-00.066_ty+00.240_s+00.786_9981a8e22be2765690022943e887a529900ae4d1_256x256.png
#      - image_dicarlo_hvm-public.csv & .zip
#majaj dataset:
# '/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/assy_dicarlo_MajajHong2015_public/assy_dicarlo_MajajHong2015_public.nc'
#NSD:
# /home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/
#imagenet21k_downsampled: /home/wtownle1/data-mbonner5/shared/datasets/imagenet21k_sorscher2021_downsampled


#***/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets instead of my home bonner-datasets***



#won't need to load a dataset for umap, just stimuli
majaj_data_path = '/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/assy_dicarlo_MajajHong2015_public/assy_dicarlo_MajajHong2015_public.nc'
def load_mjh_dataset(region, subject) -> xr.DataArray:
    da = xr.open_dataarray(majaj_data_path)
    da = da.where(da.animal.isin(subject),drop=True)
    da = da.where((da.region == region), drop=True)
    #get rid of unneeded coordinates
    #l = list(da.coords)
    # remove all other regions
    #l.remove('image_id') # keep stimulus id
    #da = da.drop(l) # drop other coords
    da = da.groupby('image_id').mean()
    da = da.squeeze(dim='time_bin', drop=True) #only keep for umap? (dim is size 1 anyway)
    return da.transpose()
    #return  #/ f"subject={subject}.nc"
    
majaj_stim_folder = '/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/image_dicarlo_hvm-public'
def get_mjh_image_paths():
    file_names = pd.read_csv('/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/image_dicarlo_hvm-public/image_dicarlo_hvm-public.csv')['image_file_name'].values.tolist()
    image_paths = [os.path.join(majaj_stim_folder,file) for file in file_names]
    return image_paths

def get_image_labels(dataset,image_paths,*args,**kwargs):
    if 'majajhong' in dataset:
        name_dict = pd.read_csv('/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/image_dicarlo_hvm-public/image_dicarlo_hvm-public.csv').set_index('image_file_name')['image_id'].to_dict()
        return [name_dict[os.path.basename(i)] for i in image_paths]

mjh_im_paths = get_mjh_image_paths()

def load_mjh_stimuli(image_paths):
    mjh_images = np.stack(Image.open(i) for i in image_paths)
    im_da = xr.DataArray(data=mjh_images)
    return im_da #return xr.DataArray(mjh_images) or xr.DataArray(np.stack(Image.open(i) for i in image_paths))
    #return xr.open_dataarray(majaj_stim_path #/  "stimuli.nc")


data = load_mjh_dataset(region='IT', subject='Tito')
stimuli = load_mjh_stimuli(mjh_im_paths)

umap = UMAP()
data_umap = umap.fit_transform(data)
#ValueError: Found array with dim 3. Estimator expected <= 2.

fig, ax = plt.subplots(figsize=(20, 20))

for i_stimulus in range(len(stimuli)):
    image_box = OffsetImage(stimuli[i_stimulus].values, zoom=0.5)
    image_box.image.axes = ax

    ab = AnnotationBbox(
        image_box,
        xy=(data_umap[i_stimulus, 0], data_umap[i_stimulus, 1]),
        xycoords="data",
        frameon=False,
        pad=0,
    )
    ax.add_artist(ab)

ax.set_xlim([data_umap[:, 0].min(), data_umap[:, 0].max()])
ax.set_ylim([data_umap[:, 1].min(), data_umap[:, 1].max()])
ax.axis("off")

fig.suptitle("UMAP projection of neural data")

fig_path = '/home/wtownle1/dimensionality_powerlaw/figures/keaton'
plt.savefig(f'{fig_path}/umap_test1.png')#, dpi=300)
#fig.show()



#da (both monkeys): 256 neuroids, 148480 presentations, 1 time bin
#da (Tito): 128 neuroids, 148480 presentations, 1 time bin
#da (IT, Tito): 110 neuroids, 148480 presentations, 1 time bin
#da (av repeats, above): 110 neuroids, 3200 image_ids, 1 time bin

#data_umap: 110,2
#**keeping neuroids, we want to keep presentations/images
# - in NSD data dims are presentation, neuroid. Here they are reversed
#***fix*** return da.transpose()

#stim (both np and xr): 3200, 256, 256, 3 (images, height, width, RGD channels)
