function [contribs, Mets] = contribution(images,metric)
    %takes in cell array of images and then assesses how far away
    %they are from each other on a feature metric determined by 
    %the function metric. Returns a vector of contributions for 
    % each image.
    % usage: contr = contributions(ims,@meanHue);
    
    % contribution is defined as the sum of the distances to the 
    % nearest neighbour for each image in each dimension.
    
    % apply the metric to each image producting a cell vector of 
    % vectors

    numImages=length(images);
    Mets = zeros(1,numImages,'double'); % matrix of metrics - one image per column
    for i = 1:numImages
        temp = images{i};
        %temp = metric(temp);
        Mets(i) = metric(temp);
    end

    % get the number of metric dimensions = number of rows of Mets
    dims=size(Mets); 
    % for each dimension calculate the contribution 
    
    contribs=zeros(1,numImages); % set the sum of contribs to zero
    for i=1:dims
        sing=singleMetricContribution(Mets(i,:));
        contribs=contribs+sing;
    end
    %toc
    %done
end


