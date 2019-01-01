function [ res ] = meanHue( im )
%returns the mean Hue of an image
h=rgb2hsv(im);
res=[];
res=[res mean(mean(h(:,:,1)))];
end

