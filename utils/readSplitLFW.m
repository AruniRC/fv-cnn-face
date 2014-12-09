function [ imgIdx1, imgIdx2 ] = readSplitLFW( listFilePath, imagePathsLFW )
%readSplitLFW Returns position indices in FV set for image_name pairs
%
%   Reads in the path to list of LFW pairs. 
%   Locates each image in the
%   imagePathsLFW cell-array by matching image-name to paths in
%   imagePathsLFW. 
%   The location in imagePathsLFW for an image name gives
%   its index in the array of pre-computed Fisher Vectors.

    fid = fopen(listFilePath);  %open file
    format = '%s %s';
    data = textscan(fid, format, 'delimiter', ' '); % read rest of the file
    
    names1 = data{1};
    names2 = data{2};
    
    imgIdx1 = zeros(1, length(names1));
    imgIdx2 = zeros(1, length(names2));
    
    % Mapping from image_name to index of imagePaths
    for i = 1:length(names1)
        
        for j = 1:length(imagePathsLFW)
            
            if ~isempty( strfind(imagePathsLFW{j}, names1{i}) )
                imgIdx1(i) = j;
            end
            
            if ~isempty( strfind(imagePathsLFW{j}, names2{i}) )
                imgIdx2(i) = j;
            end
            
            if imgIdx1(i) && imgIdx2(i) 
                break; % if both names have been found
            end
        end
    end
    
end

