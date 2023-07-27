import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style = 'white', context='paper',
rc={'axes.titlesize': 12,
    'lines.linewidth': 2.5,
    'axes.linewidth': 0.4})
sns.set(palette="colorblind")

seed = 0
method = 'Eig'
if method == 'Eig':
    m = 'X_transformed'
elif method == 'SVD':
    m = 'SVD'

n_pcs = 1000
pooling = ['max', 'layerPCA']
for p in pooling:
    pearson_seed = np.load(f'/home/wtownle1/dimensionality_powerlaw/results/RDM_{method}_pearson|dataset:majajhong2015|pooling:{p}|grayscale:False.npy', allow_pickle=True)
    pearson_seed = pearson_seed.flat[0]
    euclidean_seed = np.load(f'/home/wtownle1/dimensionality_powerlaw/results/RDM_{method}_euclidean|dataset:majajhong2015|pooling:{p}|grayscale:False.npy', allow_pickle=True)
    euclidean_seed = euclidean_seed.flat[0]
    #'AtlasNet|Eig_seed=0|a_-0.2|pcs_max|layer:c2'
    pearson_old = np.load(f'/home/wtownle1/dimensionality_powerlaw/results/RDMeig_pearson|dataset:majajhong2015|pooling:{p}|grayscale:False.npy', allow_pickle=True)
    pearson_old = pearson_old.flat[0]
    euclidean_old = np.load(f'/home/wtownle1/dimensionality_powerlaw/results/RDMeig_euclidean|dataset:majajhong2015|pooling:{p}|grayscale:False.npy', allow_pickle=True)
    euclidean_old = euclidean_old.flat[0]
    #'AtlasNet|Eig|a_-0.2|pcs_max|layer:c2'

    if p == 'layerPCA':
        pool = f'pcs_{n_pcs}'
    elif p == 'max':
        pool = f'pcs_{p}'

    alphas = ['-1.5']

    architecture = 'AtlasNet'
    path = f'/home/wtownle1/dimensionality_powerlaw/figures/keaton/RDMs/{architecture}'
    if not os.path.exists(path):
        os.makedirs(path)
    
    alphas = [-1.4]
    for a in alphas:
        key_seed = f'AtlasNet|Eig_seed={seed}|a_{a}|{pool}|layer:c2'
        key_old = f'AtlasNet|Eig|a_{a}|{pool}|layer:c2'
        #pearson_rdm_seed = pearson_seed[key_seed]
        #pearson_rdm_old = pearson_old[key_old]
        #euclidean_rdm_seed = euclidean_seed[key_seed]
        #euclidean_rdm_old = euclidean_old[key_old]
    
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.heatmap(pearson_seed[key_seed], square=True, cbar_kws={"shrink": .8})
        ax.set(title=f'Pearson RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.savefig(f'{path}/PearsonRDM|seed:{seed}|alpha={a}|layer:c2|{pool}|{m}.png')#, dpi=300)
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.heatmap(pearson_old[key_old], square=True, cbar_kws={"shrink": .8})
        ax.set(title=f'Pearson RDM (seed:Unk|alpha:{a}|{pool}|{m})')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.savefig(f'{path}/PearsonRDM|seed:Unk|alpha={a}|layer:c2|{pool}|{m}.png')#, dpi=300)
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.heatmap(euclidean_seed[key_seed], square=True, cbar_kws={"shrink": .8})
        ax.set(title=f'Pearson RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.savefig(f'{path}/EuclideanRDM|seed:{seed}|alpha={a}|layer:c2|{pool}|{m}.png')#, dpi=300)
        
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.heatmap(euclidean_old[key_old], square=True, cbar_kws={"shrink": .8})
        ax.set(title=f'Euclidean RDM (seed:Unk|alpha:{a}|{pool}|{m})')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.savefig(f'{path}/EuclideanRDM|seed:Unk|alpha={a}|layer:c2|{pool}|{m}.png')#, dpi=300)
    
    
    
    
    
    
    #alphas = ['-1.5']
    #for model in pearson_rdms.keys():
    #    a = model.split('|')[1].split('_')[1]
    #    if a in alphas:
    #        architecture = model.split('|')[0]
    #        path = f'/home/wtownle1/dimensionality_powerlaw/figures/keaton/RDMs/{architecture}'
    #        if not os.path.exists(path):
    #            os.makedirs(path)
    
    #        for layer_rdms in pearson_rdms[model].items():
    #            layer = layer_rdms[0]
    #            rdm = layer_rdms[1]
                
    #            fig, ax = plt.subplots(figsize=(10,10))
    #            ax = sns.heatmap(rdm, square=True, cbar_kws={"shrink": .8})
    #            ax.set(title=f'Pearson RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
    #            ax.set_xticklabels([])
    #            ax.set_yticklabels([])
    #            plt.savefig(f'{path}/PearsonRDM|seed:{seed}|alpha={a}|layer:{layer}|{pool}|{m}.png')#, dpi=300)
                
    #        for layer_rdms in euclidean_rdms[model].items():
    #            layer = layer_rdms[0]
    #            rdm = layer_rdms[1]
                
    #            fig, ax = plt.subplots(figsize=(10,10))
    #            ax = sns.heatmap(rdm, square=True, cbar_kws={"shrink": .8})
    #            ax.set(title=f'Euclidean RDM (seed:{seed}|alpha:{a}|{pool}|{m})')
    #            ax.set_xticklabels([])
    #            ax.set_yticklabels([])
    #            plt.savefig(f'{path}/EuclideanRDM|seed:{seed}|alpha={a}|layer:{layer}|{pool}|{m}.png')#, dpi=300)




#seed=0
#'/home/wtownle1/dimensionality_powerlaw/results/RDM_Eig_euclidean|dataset:majajhong2015|pooling:layerPCA|grayscale:False.npy'
#'/home/wtownle1/dimensionality_powerlaw/results/RDM_Eig_euclidean|dataset:majajhong2015|pooling:max|grayscale:False.npy'
#'/home/wtownle1/dimensionality_powerlaw/results/RDM_Eig_pearson|dataset:majajhong2015|pooling:layerPCA|grayscale:False.npy'
#'/home/wtownle1/dimensionality_powerlaw/results/RDM_Eig_pearson|dataset:majajhong2015|pooling:max|grayscale:False.npy'
#seed=unk
#'/home/wtownle1/dimensionality_powerlaw/results/RDMeig_euclidean|dataset:majajhong2015|pooling:layerPCA|grayscale:False.npy'
#'/home/wtownle1/dimensionality_powerlaw/results/RDMeig_euclidean|dataset:majajhong2015|pooling:max|grayscale:False.npy'
#'/home/wtownle1/dimensionality_powerlaw/results/RDMeig_pearson|dataset:majajhong2015|pooling:layerPCA|grayscale:False.npy'
#'/home/wtownle1/dimensionality_powerlaw/results/RDMeig_pearson|dataset:majajhong2015|pooling:max|grayscale:False.npy'