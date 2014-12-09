function images_resized = scaleLFW(image_array)
% scaleLFW Scales the height of all images in image_array to be 150,
% maintaining the aspect ratio. This makes it compatible with a Fisher GMM
% trained upon 150x150 centered crops on LFW.
    
    images_resized = cell(1, length(image_array));
    for i = 1:length(image_array)
        img = image_array{i};
        
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        
        img = im2single(img);
        
        [height, width] = size(img);  
        width = 150/height * width;
        height = 150;
        images_resized{i} = imresize(img, [height width]);
    end
end
        