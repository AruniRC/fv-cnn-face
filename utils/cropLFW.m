function [croppedFaceImg] = cropLFW(faceImg)
% faceImg Input cell array of RGB LFW images
    croppedFaceImg = cell(1, length(faceImg));
    for idxImg = 1:length(faceImg)
        
        img = rgb2gray(im2single(faceImg{idxImg}));

        % crop central 150x150 image
        croppedFaceImg{idxImg} = img(51:200, 51:200);
    end
end
    

