function [ res ] = imDistanceR( image1,image2,r )
%   Measures the per-pixel distances between two images in L2 transition space
%   For each x, for each y for each channel work out the difference. 
%  assumes that image1,image2,A and B are all the same size. 

    % bring both images into the 0,1.0 range. 
    %A=im2double(A);  
    %B=im2double(B); 
    % get the lowerBound reference image and upperBound ref image
    %lowerBound=min(A,B); 
    %upperBound=max(A,B);
    
    % get the distance between these per pixel.. this is part of the 
    %scaling factor to apply to differences 
    %diffs=double(upperBound-lowerBound)+eps;
    
    % the maximum difference this is also part of the scaling factor
    %rows=length(A(:,1,1));
    %cols=length(A(1,:,1));
    %maxDiff=ones(rows,cols,3);
    
    % scaling factors 
    %scaleFactors = maxDiff./diffs;


    % now find image differences
    image1=im2double(image1); % make range 0 to 1.0
    image2=im2double(image2); % make range 0 to 1.0
    diff=double(abs(image1-image2));
    %diff=diff.*scaleFactors; % scaled differences
    
    res=rthPowerMean(diff,r);
    % L2 distance is RMS....
    %diffSquared=diff.*diff;
    %meanDiffSquared=sum(sum(sum(diffSquared)))/prod(size(image1));
    %res=sqrt(meanDiffSquared);
end

