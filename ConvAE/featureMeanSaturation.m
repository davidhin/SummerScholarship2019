function [ res ] = meanSaturation( im )
%returns the mean saturation of an image
h=rgb2hsv(im);
res=[];
res=[res mean(mean(h(:,:,2)))];
end