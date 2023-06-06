import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

np.random.seed(seed=0)

# Set parameters
num_samples = 5000  # Number of samples
num_features = 1000  # Number of features
power_law_exponent = -2  # Exponent of power law decay
variance_scale = 1  # Scaling factor for eigenvectors' variances

# Generate principal components
eigenvalues = np.power(np.arange(1, num_features+1, dtype=float), power_law_exponent)
eigenvectors = np.random.randn(num_features, num_features)
eigenvectors = eigenvectors / np.sqrt(np.sum(np.square(eigenvectors), axis=0))  # Normalize columns

# Scale eigenvectors' variances
eigenvectors = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :] * variance_scale

# Generate random data
X = np.random.randn(num_samples, num_features) @ eigenvectors

pca = PCA()
pca.fit(X)
#plt.loglog(pca.explained_variance_)

n_pc = len(pca.explained_variance_)
end = np.log10(n_pc)
eignum = np.logspace(0, end, num=1000).round().astype(int)
eigspec = pca.explained_variance_[eignum - 1]
logeignum = np.log10(eignum)
logeigspec = np.log10(pca.explained_variance_)
linear_fit = LinearRegression().fit(logeignum.reshape(-1,1), logeigspec)
alpha = -linear_fit.coef_.item()
print(alpha)


pcs = np.arange(1, n_pc+1)
logeignum = np.log10(pcs)
logeigspec = np.log10(pca.explained_variance_)
linear_fit = LinearRegression().fit(logeignum.reshape(-1,1), logeigspec)
alpha_wrong = -linear_fit.coef_.item()
print(alpha_wrong)

