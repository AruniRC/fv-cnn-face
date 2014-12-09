%% CNN Fisher Vector encoding for LFW dataset
%
% All the LFW funneled images are assumed to be in a cell array saved at
% data/all_img.mat
% VLFeat library needs to be installed first.
%
% The FV models are expected to be pre-computed using train_gmm_script (or
% available at location models/fisher_model.mat)
%
% This calculates FVs for each image in the LFW dataset. Funneled LFW
% images center-cropped to 150x150 are used here.

% CURRENT - upscaled-image model. 

init_script;

disp('Loading data and models');

% load Fisher Vector model - GMM and PCA projection
% load('models/cnn_fisher_gmm_upscaled.mat'); % fisher_gmm, pca_proj, pca_mu
% load('models/cnn_fisher_gmm_rawupscale.mat');
% model_name = 'cnn_fisher_gmm_MultiUpscaled';
model_name = 'cnn_fisher_gmm_vd16_K128_250px';

load(['models/' model_name '.mat']);


net = load('cnn_models/imagenet-vgg-verydeep-16.mat');
% net.layers = {net.layers{1:13}};
net.layers = {net.layers{1:29}};

% load in all LFW funneled images
if ~exist('imagesLFW', 'var')
    load('./data/all_img_lfw_funneled.mat', 'imagesLFW'); % faceImg
end

disp('Done loading');

% get central 150x150 crops from LFW
faceImg = cropLFW(imagesLFW);
% faceImg = imagesLFW; % raw 250x250 images

clear imagesLFW;
%%
% Preallocate for Fisher Vector encoding of each image 
fv = zeros(2 * fisher_gmm.K * fisher_gmm.dataDim, length(faceImg));

% compute dense CNN features for images - 512 dimension - no PCA
params = struct;
params.scale_factor = 1 ; %2 ^ (1/2);
params.scale = 1;
params.num_scales = 1;
params.aug_frames = false;
params.sqrt_map = false;
params.normalize = true;

disp('Computing DCNN Fisher Vectors for all LFW images');
tic
for i = 1:length(faceImg)
    feat = cell(1, 4);
    % img = imresize(faceImg{i}, 2); % upscaled 150x150 crops by factor 2
    img = faceImg{i}; 
    
    feat{1} = denseCNNFeat(img, net, params);
    feat{2} = denseCNNFeat(imresize(img, 2), net, params);
    feat{3} = denseCNNFeat(imresize(img, 3), net, params);
    feat{4} = denseCNNFeat(imresize(img, 0.5), net, params); 
    feat = cell2mat(feat);
    
    fv(:,i) = vl_fisher(feat, fisher_gmm.means, fisher_gmm.covars, ...
                    fisher_gmm.priors, 'Improved');
end
toc



% save to disk
save('./data/fv_cnn_vd16_K128_250px.mat', 'fv', '-v7.3');

disp('Done');

