%  Copyright (c) 2014, Karen Simonyan
%  All rights reserved.
%  This code is made available under the terms of the BSD license (see COPYING file).

% Modified at UMass by Aruni Roy Chowdhury.

function [tpr, tnr, res, extra] = evalScores(scores, gt)
    
    [tpr, tnr, info] = vl_roc(gt, scores);
    
    % the accuracy at the ROC operating point where the error rates are equal (as in [Guillaumin et al., ICCV '09])
    res = (1 - info.eer) * 100;
    extra = info;
end


