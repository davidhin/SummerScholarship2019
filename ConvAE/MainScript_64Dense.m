%% This script runs CMAES using the specified metric and fitness function
%% Run StartSocket before running this script

%% ------------------------------------------- %%
%%                     SETUP                   %%
%% ------------------------------------------- %%

% Setup - Fitness
global t;
global metric;
global contribs;
global fitnessFunction;
metric = @featureMeanHue; % Metric to be used
contribs = @contribution;
fitnessFunction = @fit_minDistImg; % Fitness function to be used

% Setup - Variables
global mu;
global samples;
global current_images;
global current_encoding;
mu = 8;
samples = 64;
sigma = 0.7; % Setting a smaller sigma is better
rng(3, 'twister');
load('starting_vector.mat', 'current_encoding');
start = current_encoding;

% Setup - CMAES
opts = cmaes;
opts.StopFunEvals = 150000;
opts.PopSize = 10; % Set population to around 5/10
opts.StopOnWarnings = 0;
opts.StopOnStagnation = 0;
opts.WarnOnEqualFunctionValues = 0;
opts.TolFun = -inf;

% Plot
global distInterval;
distInterval = 100;

%% ------------------------------------------- %%
%%                     CMAES                   %%
%% ------------------------------------------- %%

% Import the images 
global A
global B
load('../Autoencoder/encodedImages.mat');
A=imread('Kand5f.jpg');
A=imresize(A,[128,128]);
B=imread('yellowkk.jpg');
B=imresize(B,[128,128]);

% Plotting
global metric_plot;
global contrib_plot;
global fitness_plot;
global intervalCounter;
metric_plot = [];
contribs_plot = [];
fitness_plot = [];
intervalCounter = 0;

% Initiate population
pop = [];
stdev = [];
for i = 1:mu
    %pop = [pop,normrnd(0,2,[samples,1])];
    pop = [pop,start];
    %pop = [pop,zeros(samples,1)];
    stdev = [stdev,ones(samples,1)*sigma];
end

tic
x = cmaes('cmaesFitnessFunction',reshape(pop, [samples*mu,1]),reshape(stdev, [samples*mu,1]),opts);
toc

%% ------------------------------------------- %%
%%                   RESULTS                   %%
%% ------------------------------------------- %%

% Decode final images
x = reshape(x, [samples,mu]);
x = num2cell(x, 1);
for indx=1:mu
    x{indx} = decode(t, x{indx});
end
final = x;

global metricVec;
global fitnessVec;
[contributions, metricVec] = contribs(final, metric)
% figure
% scatter(metricVec, ones(mu,1))

fitnessVec = [];
for indx=1:mu
    fitnessVec = [fitnessVec, fitnessFunction(final{indx})];
end
fitnessVec

showPop(final,1,mu)
return
