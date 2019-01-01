%% this code should copy 1+1EA part of the code used in decodetest_colour.m.
% ensure autoencoders should be saved after running trainAutoencoder.m and
% should be loaded here.
% close all;
% load constantAutoenc0;
% load constantAutoenc1;
% load constantAutoenc2;
% load xTrainImages;

close all;

%% changes in code shouldn't take here but, in the decodetest_colour.m. Then copy paste it here.
imSize=128;
count = 0;
A=imread('Kand5f.jpg');
%A=imread('benchmarks/Slide1.jpg');
A=imresize(A,[imSize,imSize]);
B=imread('yellowkk.jpg');
%B=imread('benchmarks/Slide2.jpg');
B=imresize(B,[imSize,imSize]);

iters = 30000;
metric = @featureMeanHue; 
%metric = @featureImageEntropy;
contribs = @contribution;
% fitness = @featureImageSmoothness;
fitness = @SimA; %% Always 1
mu=6;
lambda=3;
dispInterval=100;
contributionTrace=[]; % holds sorted set of contributions.
mutRate=1/imSize; % Make this value bigger 1/64 maybe

%% Set all output file names %%
jobID = 6;
targ = 0.3;
prefix = strcat('id_a',int2str(jobID),'_MeanHue');
suffix = '_fitSmoothness';
chartName = strcat(prefix,'_chart_',suffix,'.jpg');
imageName = strcat(prefix,'_',suffix,'.jpg');
outputTimeName = strcat(prefix,'_',suffix,'.txt');
chartFeatureFile = strcat(prefix,'_chartFeautre_',suffix,'.txt');
chartFitnessFile = strcat(prefix,'_chartFitness_',suffix,'.txt');
xAxisLabel = char(metric);
yAxisLabel = strcat(char(fitness));

% containers
contributionVec = zeros(1,mu);
newContributionVec = zeros(1,mu);
oldSimVec = zeros(1,mu);
newSimVec = zeros(1,mu);
fitnessVec = zeros(1,mu);
newFitnessVec = zeros(1,mu);
metricVec = zeros(1,mu);
newMetricVec = zeros(1,mu);
fitnessPlotX = zeros(1,mu);
fitnessPlotY = zeros(1,mu);

% get mean of latent vector
tic

% latentMean = mean(mean(mean(encode(autoenc2,feat1))));
latentMean = normrnd(0,5,[64,1]); % This is not actually the means of the latent vector
sigma = 1;
samples = 64;

rng(1, 'twister');
% seed population
pop = cell(1,mu);
for i = 1:mu
    % testVec1=random('normal', latentMean, sigma, samples, 1);
    testVec1 = normrnd(0,5,[64,1]); 
    pop{i}=testVec1;
end

tic

% make a container for the lambda new individuals
newInds= cell(1,lambda);

plot_metricVec = [;];
plot_contribsVec = [;];
plot_fitnessVec = [;];

% loop for iters algorithms. 
for i = 1:iters
    % make new individuals
    perms=randperm(length(pop)); % generate permuation.
    substIdxs=perms(1:lambda); % the set of indexes for substitution
    for j=1:lambda
        sel=pop{perms(j)};
        ind=mutateLatentVector(sel,latentMean, sigma, mutRate); % This should be mute rate
        newInds{j}= ind; %new individual
    end


    substIdxs(substIdxs==-1)=[]; % now remove all -1's
    substIdxsVec = zeros(1,mu);
    for j = 1:length(substIdxs) 
        substIdxsVec(substIdxs(j)) = 1;
    end
    
    % compact newInds
    newInds=newInds(~cellfun(@isempty,newInds));
    	
    % create a copy of the population and substitute in the newInds
    newPop=pop;
    for j = 1:length(substIdxs)
        newPop{substIdxs(j)} = newInds{j};
    end
	
    
    % now tmpPop has the new inviduals substituted in
    % and pop has the old values. 
    % gather contributions for each according to metric function

    %convert pop vec to pop image
    popImage = cell(1,mu);
    newPopImage = cell(1,mu);
    for indx=1:mu
        popImage{indx} = decode(t, pop{indx});
        newPopImage{indx} = decode(t, newPop{indx});
    end	
    
    [contributions, metricVec] = contribs(popImage,metric);
    [newContributions, newMetricVec]=contribs(newPopImage,metric);

    % now we have contributions choose the best of the
    % corresponding new and old individuals 

    for j = 1:length(substIdxsVec)
        if substIdxsVec(j) == 1
            old = contributions(j);
            new = newContributions(j);

            fitnessVec(j) = fitness(popImage{j});
            newFitnessVec(j) = fitness(newPopImage{j});

            oldSimVec(j) = old*fitnessVec(j);
            newSimVec(j) = new*newFitnessVec(j); % fitness function is the key thing, just always return a 1 and will simulate the fitness not working
            
            if newSimVec(j) > oldSimVec(j)    %replace by pairwise comparisons, %% print line here and check how often this pairwise replacement is happening.
                pop{j} = newPop{j};
                metricVec(j) = newMetricVec(j); %replace metric val as well
                fitnessVec(j) = newFitnessVec(j);
            end
        end
    end
    % display the stats every dispIntervals iters
    if mod(i,dispInterval) == 0
        % sort images in terms of saturation scores %
        fitRes = [];
        pop2 = cell(1,mu);

        % get sorted order and show pic like that
        [sortMetricVec,idx_order] = sort(metricVec);

        sort(contributions)

        plot_metricVec = [plot_metricVec;metricVec];
        plot_contribsVec = [plot_contribsVec;contributions];
        plot_fitnessVec = [plot_fitnessVec;fitnessVec];

        for tt = 1:mu
           pop2{tt} = decode(t, pop{idx_order(tt)}); 
        end

        close all;
        figure('visible', 'on')
        % images are converted to uint8 from double. other images wont show!!
        showPop(pop2,2,3)
        saveas(gcf,imageName)
        %% end sort in terms of hue score %%

        fprintf('%s\t Run: %.0f\n',datestr(now),i);

%         figure(2)
%         matrixRow = i/dispInterval;
%         featureMatrix(matrixRow,:) = metricVec;
%         fitnessMatrix(matrixRow,:) = fitnessVec;
%         showChart(featureMatrix, fitnessMatrix,3,3);
%         xlabel(xAxisLabel);
%         ylabel(yAxisLabel) ;
%         fprintf(featurePlotFileID, '%d, %0.5f, %0.5f, %0.5f, %0.5f, %0.5f, %0.5f\n',i, featureMatrix(matrixRow,:));
%         fprintf(fitnessPlotFileID, '%d, %0.5f, %0.5f, %0.5f, %0.5f, %0.5f, %0.5f\n',i, fitnessMatrix(matrixRow,:));

%         saveas(gcf,chartName);
        toc
    end
end
toc
