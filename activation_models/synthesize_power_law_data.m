%% Synthesize data matrices with power law decay eigenspectra

% Set parameters
num_samples = 5000; % Number of samples
num_features = 1000; % Number of features
power_law_exponent = -1; % Exponent of power law decay
variance_scale = 1; % Scaling factor for eigenvectors' variances

% Generate principal components
eigenvalues = (1:num_features).^power_law_exponent;
eigenvectors = randn(num_features);
eigenvectors = eigenvectors ./ sqrt(sum(eigenvectors.^2, 1)); % Normalize columns

% Scale eigenvectors' variances
eigenvectors = eigenvectors .* sqrt(eigenvalues) * variance_scale;

% Generate random data
X = randn(num_samples, num_features) * eigenvectors;

% Check eigenspectrum
[~, ~, eigenvalues] = pca(X);
figure
loglog(eigenvalues)
