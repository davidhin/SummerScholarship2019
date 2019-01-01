% Initial distribution
% rng(1, 'twister')
% data_init = normrnd(0,1,[64,1]);
% 
% % Import the images 
% global A
% global B
% A=imread('Kand5f.jpg');
% A=imresize(A,[128,128]);
% B=imread('yellowkk.jpg');
% B=imresize(B,[128,128]);
% 
% % Do feature based search.
% featureLen=64;
% opts=cmaes;
% opts.StopFunEvals=2000;
% x=cmaes('fitPlusSimABtimesMeanHue',zeros(featureLen,1),ones(featureLen,1)*10,opts);
% final=decode(t,x);
% imshow(decode(t,x))
% return
% Edit cmaesFitnessFunction.m

% Initial distribution
rng(1, 'twister')
data_init = normrnd(0,1,[64,1]);

% Import the images 
global A
global B
load('../Autoencoder/encodedImages.mat');
A=imread('Kand5f.jpg');
A=imresize(A,[128,128]);
B=imread('yellowkk.jpg');
B=imresize(B,[128,128]);

global metric;
global contribs;
global fitnessFunction;
metric = @featureMeanHue; 
contribs = @contribution;
fitnessFunction = @fit_minDistImg;

global metric_plot;
global contrib_plot;
global fitness_plot;
metric_plot = [];
contribs_plot = [];
fitness_plot = [];

global intervalCounter;
global distInterval;
intervalCounter = 0;
distInterval = 1000;

% Setup - Variables
global mu;
global samples;
global current_images;
mu = 5;
samples = 64;
lambda = 3;
sigma = 3; % Setting a smaller sigma is better

% Setup - CMAES
opts = cmaes;
opts.StopFunEvals = 10000;
opts.PopSize = 10; % Set population to around 5/10

% Initiate population
rng(1, 'twister');
pop = [];
stdev = [];
for i = 1:mu
    pop = [pop,normrnd(0,2,[samples,1])];
    %pop = [pop,zeros(samples,1)];
    stdev = [stdev,ones(samples,1)*sigma];
end

% pop(1:64,1) = encodedA;
% pop(1:64,3) = encodedB;

% Do feature based search.
% x = cmaes('fitPlusSimABtimesMeanHue',pop(1:64,1),stdev(1:64,1),opts);
tic
x = cmaes('cmaesFitnessFunction',reshape(pop, [samples*mu,1]),reshape(stdev, [samples*mu,1]),opts);
toc

% Decode final images
x = reshape(x, [samples,mu]);
x = num2cell(x, 1);
for indx=1:mu
    x{indx} = decode(t, x{indx});
end
final = x;

% Show final population
figure
showPop(final,2,3)

global metricVec;
global fitnessVec;
[contributions, metricVec] = contribs(final, metric)
figure
scatter(metricVec, ones(mu,1))

fitnessVec = [];
for indx=1:mu
    fitnessVec = [fitnessVec, fitnessFunction(final{indx})];
end
fitnessVec

return

