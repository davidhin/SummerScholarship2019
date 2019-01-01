function [ contribs ] = singleMetricContribution( metrics )
%takes in a set of metrics and returns a set of contributions
%for each of those metrics based on distance from each other.

metLen=length(metrics); % extract length of metrics. 
indices=1:metLen;
Vals=[metrics;indices];  % indicies paired with metrics. 
Vals=sortrows(Vals',1)'; % sort by metrics

% now we have a sorted set of values measure the contribution 
% as being the the product of the distances to its neigbours in 
% a sorted order
% add metrics as a third row
Vals=[Vals;zeros(metLen)];  % init third row
range=Vals(1,metLen)-Vals(1,1); % calculate range
% initialise first and last elements to range-squared
Vals(3,1)=range^2;
Vals(3,metLen)=range^2;
% iterate over rest to get other contribs
for i=2:metLen-1
    Vals(3,i)=(Vals(1,i)-Vals(1,i-1))*(Vals(1,i+1)-Vals(1,i));
end

% now we have all our contributions in the third row
% sort by the index row and return the third row
Vals=sortrows(Vals',2)'; % sort by index row
contribs=Vals(3,:);
%normalise to be in the range of [0,1]
%maxContrib = max(contribs);
%contribs=contribs./maxContrib;

end


