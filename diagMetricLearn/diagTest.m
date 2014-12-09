function [ scores ] = diagTest( model, feat1, feat2 )
%DIAGTEST Summary of this function goes here
%   Detailed explanation goes here

    % squared difference feature
    testFeats = (feat1 - feat2) .^ 2;
    
    % compute test scores
    scores = -model.w' * testFeats;
    
end

