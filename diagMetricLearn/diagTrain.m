%  Copyright (c) 2014, Karen Simonyan
%  All rights reserved.
%  This code is made available under the terms of the BSD license (see COPYING file).

% Modifed at UMass Amherst by Aruni Roy Chowdhury

function model = diagTrain(trainData, valData, params)


    % trainFeatsPos = ( trainData.feats(:, trainData.posPairs(1, :)) - trainData.feats(:, trainData.posPairs(2, :)) ) .^ 2;
    % trainFeatsNeg = ( trainData.feats(:, trainData.negPairs(1, :)) - trainData.feats(:, trainData.negPairs(2, :)) ) .^ 2;
    
    featDim = size(trainData.feats, 1);
    lambda = params.lambda;
        
    w = zeros(featDim, 1, 'single');
    
    % sets of positives & negatives
    nPos = size(trainData.posPairs, 2);    
    nNeg = size(trainData.negPairs, 2);    
        
    rng(params.rngSeed);
    
    % positive & negative pairs for each iteration
    idxPos = randi([1 nPos], params.numIter, 1);
    idxNeg = randi([1 nNeg], params.numIter, 1);
    
    
    % validation set initializations
    if ~isempty(valData)
        feat1 = trainData.feats(:, valData.valPair1);
        feat2 = trainData.feats(:, valData.valPair2);
        valPairFeat = feat1 - feat2;
        k = 1;
        max_val_accu = 0;
        best_t = 0;
        accu_set = zeros(1, 1000);
    end
        
    for t = 1:params.numIter
        
        % learning rate
        gamma = 1 / (lambda * t);
        
        % feature vector
	trainFeatsPos = ( trainData.feats(:, trainData.posPairs(1, idxPos(t))) - trainData.feats(:, trainData.posPairs(2, idxPos(t))) ) .^ 2;

	trainFeatsNeg = ( trainData.feats(:, trainData.negPairs(1, idxNeg(t))) - trainData.feats(:, trainData.negPairs(2, idxNeg(t))) ) .^ 2;

	feat = trainFeatsPos - trainFeatsNeg;


        % feat = trainFeatsPos(:, idxPos(t)) - trainFeatsNeg(:, idxNeg(t));
        
        % update w
        if w' * feat > -1
            w = w * (1 - gamma * lambda) - gamma * feat;
        else
            w = w * (1 - gamma * lambda);
        end
        
        
        % validation set
        if ~isempty(valData)
            if mod(t, 1e3) == 0 %% EDIT- 1e3
                val_accu = get_val_accu(w, valPairFeat, valData.valGT);
                 fprintf('%d %f\n', t, val_accu)

                accu_set(k) = val_accu;
                k = k + 1;
                if val_accu > max_val_accu
                    max_val_accu = val_accu;
                    best_t = t;
                end
            end
        end
           
    end
    
    % save learnt model     
    model = struct;
    
    % current state
    model.w = w;
    
    if ~isempty(valData)
        model.t = best_t;
        model.accu_plot = accu_set;
    end

        
    % params
    model.lambda = lambda;
           
end

function accu = get_val_accu(w, valPairFeat, gt)
    
     % squared difference feature
    testFeats = valPairFeat .^ 2;
    
    % compute test scores
    scores = -w' * testFeats;
    
    [~,~,info] = vl_roc(gt, scores);
    accu = 1 - info.eer;
end

