
init_script;


model_name = 'cnn_fisher_gmm_vd16_K128_250px';
numClusters = 128;


% Very deep model-16
% layer{29}
% res(30).x

% UPSCALING 
cnn_model = 'imagenet-vgg-verydeep-16';


net = load(['./cnn_models/' cnn_model]);
load('./data/all_img_lfw_funneled.mat', 'imagesLFW');

% get central 150x150 crops from LFW
% faceImg = cropLFW(imagesLFW);

faceImg = imagesLFW;

% keep only till conv5 layer of network
% net.layers = {net.layers{1:13}};
net.layers = {net.layers{1:29}};

% sample 3K images from total 13233 LFW images
rnd_idx_img = randperm(length(faceImg), 1000);
faceImgRnd = {faceImg{rnd_idx_img}};

clear faceImg imagesLFW rnd_idx_img;

denseFeatures = cell(1, length(faceImgRnd));

% compute dense CNN features for images - 512 dimension - no PCA
params = struct;
params.scale_factor = 1 ; %2 ^ (1/2);
params.scale = 1;
params.num_scales = 1;
params.aug_frames = false;
params.sqrt_map = false;
params.normalize = true;

feat = cell(1, 4);

%%
disp('computing dense CNN features');
tic
for i = 1:length(faceImgRnd)
    % img = imresize(faceImgRnd{i}, 2); % upscaled 150x150 crops by factor 2
    % img = imresize(faceImgRnd{i}, [300 300]); % 150x150 crops resized to 250x250
    % img = faceImgRnd{i};
    
    img = faceImgRnd{i};
    feat{1} = denseCNNFeat(img, net, params);
    feat{2} = denseCNNFeat(imresize(img, 2), net, params);
    feat{3} = denseCNNFeat(imresize(img, 3), net, params);
    feat{4} = denseCNNFeat(imresize(img, 0.5), net, params); 
    denseFeatures{i} = cell2mat(feat);
end
toc
clear faceImgRnd;
denseFeatures = cell2mat(denseFeatures);

disp('Done');


%% learning GMM parameters

dimension = size( denseFeatures, 1);

disp('Initializing using KMeans');

% Run KMeans to pre-cluster the data
[initMeans, assignments] = vl_kmeans(denseFeatures, numClusters, ...
    'Algorithm', 'Elkan', ...
    'MaxNumIterations', 100);

initCovariances = zeros(dimension,numClusters);
initPriors = zeros(1,numClusters);

% Find the initial means, covariances and priors
for i=1:numClusters
    data_k = denseFeatures(:,assignments==i);
    initPriors(i) = size(data_k,2) / numClusters;

    if size(data_k,1) == 0 || size(data_k,2) == 0
        initCovariances(:,i) = diag(cov(data'));
    else
        initCovariances(:,i) = diag(cov(data_k'));
    end
end

%%
% Run EM starting from the given parameters
disp('training gmm');
tic

fisher_gmm = struct;

[fisher_gmm.means, fisher_gmm.covars, fisher_gmm.priors] = vl_gmm(denseFeatures, numClusters, ...
    'initialization','custom', ...
    'InitMeans', single(initMeans), ...
    'InitCovariances',single(initCovariances), ...
    'InitPriors', single(initPriors));

fisher_gmm.K = numClusters;
fisher_gmm.dataDim = size(denseFeatures, 1);

toc



% save to disk
if ~exist('models', 'dir')
    mkdir('models');
end
save(fullfile('models', [model_name '.mat']), 'fisher_gmm', '-mat');

disp('Done.');


