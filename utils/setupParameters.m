function ParameterSetup = setupParameters()
% Setup the learning parameters for Optimizations
ParameterSetup = {};

% Train-Test Data----------------------------------------------------------
ParameterSetup.NSplits     = 800;
ParameterSetup.domainNames = {'amazon','webcam','dslr','caltech'};

% Regularizers-------------------------------------------------------------
ParameterSetup.lamda = 1;   % Covariance and Mean Descripency

%Manifold Parameters-------------------------------------------------------
ParameterSetup.n = 4096; % Original Dimensions
ParameterSetup.p = 20;  % Projected dimensions
ParameterSetup.manifold = 'stiefel' ; % 'stiefel' 'euclidean'

%Optimization parameters---------------------------------------------------
ParameterSetup.iters = 30;

end

