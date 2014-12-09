%% On View 1 LFW 

init_script;
addpath('./diagMetricLearn');


% Load pre-computed Fisher Vectors  - MODIFIED FOR CNN-FV
load('./data/fv_cnn_lfw_MultiUpscaledCrop.mat', 'fv');
fv_lfw = fv;
clear fv;

if ~exist('imagePathsLFW', 'var')
    load('./data/all_img_lfw_funneled.mat', 'imagePathsLFW');
end


%%

% parameters
params.lambda = 1e-5;
params.rngSeed = 6756;
params.numIter = 1e6;


% Training and validation data

% Training set pairs
disp('Reading in training set - DevTrain');
[ imgIdx1_same, imgIdx2_same ] = ...
    readSplitLFW( './data/lists/dv_train_same.txt', imagePathsLFW );
[ imgIdx1_diff, imgIdx2_diff ] = ...
    readSplitLFW( './data/lists/dv_train_diff.txt', imagePathsLFW );
trainData = struct;
trainData.feats = fv_lfw;
trainData.posPairs = [ imgIdx1_same ; imgIdx2_same ]; % 2xN
trainData.negPairs = [ imgIdx1_diff ; imgIdx2_diff ];


% Validation set for early stopping
disp('Reading in a validation set - DevTest');
[ imgIdx1_same, imgIdx2_same ] = ...
    readSplitLFW( './data/lists/dv_test_same.txt', imagePathsLFW );
[ imgIdx1_diff, imgIdx2_diff ] = ...
    readSplitLFW( './data/lists/dv_test_diff.txt', imagePathsLFW );
valData = struct;
valData.valPair1 = [imgIdx1_same, imgIdx1_diff];
valData.valPair2 = [imgIdx2_same, imgIdx2_diff];
valData.valGT = [ ones(1, length(imgIdx1_same)) -ones(1, length(imgIdx1_diff))];

clear fv_lfw imagePathsLFW imgIdx1_same imgIdx2_same imgIdx1_diff imgIdx2_diff;

disp('Training DiagMetric model . . . ');

tic
model = diagTrain(trainData, valData, params);
toc

disp('Done (not saving to disk)');




