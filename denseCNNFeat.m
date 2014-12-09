
function feats = denseCNNFeat(im, net, params)
 
    
    
    % handle single-channel image
    if size(im, 3) == 1
        im_ = zeros(size(im,1), size(im,2), 3);
        im_(:,:,1) = im;
        im_(:,:,2) = im;
        im_(:,:,3) = im;
        im = im_;
    end
    
    % Pre-processing image for CNN
    img = single(im);
    im = img - imresize((net.normalization.averageImage), [size(img,1) size(img,2)]);
    
    
    feats = cell(1, params.num_scales);
    scale = params.scale;

    
    for i = 1:params.num_scales
        
        % resize image
        if scale > 1            
            im_scale = imresize(im, 1 / scale);
        else
            im_scale = im;
        end
        
        % compute CNN-feats at this scale
        feats{i} = extractCNNFeat( net, im_scale, params );
        
        % increase scale
        scale = scale * params.scale_factor;
        
    end
    
    % put all features together
    feats = cat(2, feats{:});

    feats = single(feats);
    
    % L2 normalize
    if params.normalize
        feats = normc(feats);
    end
    
end

function feat = extractCNNFeat( net, img, params )
%EXTRACTCNNFEAT Extract conv5 layer features densely from un-resized image
%
% net   CNN structure. Optionally with layers after 'conv5' removed.
% img   3-channel image, un-preprocessed.

    res = vl_simplenn(net, img);
    % y = res(14).x;    % 13 x 13 x 512
    y = res(30).x;
    feat = reshape(y, size(y,1) * size(y,2), size(y,3)); % 169 x 512
    feat = feat'; % 512 x 169, column-wise data points
    
    if params.sqrt_map
        feat = feat./sum(feat);
        feat = sign(feat)*sqrt(abs(feat));
    end
    
    if params.aug_frames
        [xPos, yPos] = meshgrid([1:size(y,2)], [1:size(y,1)]);
        xyPos = cat(3, xPos/size(y,2), yPos/size(y,1));
        sp_aug = reshape(xyPos, size(y,1) * size(y,2), 2); % 169 x 2
        feat = cat(1, feat, sp_aug');
    end
    
end