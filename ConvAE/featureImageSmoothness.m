function [ res ] = featureImageSmoothness(im)
	% Returns the mean smoothness of an image.
    
    % Per-channel smoothness
    meanSmoothness = zeros(1, 3);
    for channel = 1:3
        [grad, ~] = imgradient(im(:, :, channel), 'intermediate');
        %figure; imshow(uint8(grad));
        meanSmoothness(channel) = mean(mean(grad));
    end
    meanSmoothness = mean(meanSmoothness);
    res = 1 - (meanSmoothness / (255 * 8)); % 8 is normalisation factor 
    % (1 - ...) is used so 1 corresponds to max smoothness
end

