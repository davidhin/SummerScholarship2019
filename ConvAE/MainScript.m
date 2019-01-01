% Setup socket
% global t
% t = tcpip('129.127.10.18', 8221, 'NetworkRole', 'client');
% t.OutputBufferSize = 2048;
% t.InputBufferSize = 1; % Change this
% fopen(t);
% return

load('../Autoencoder/encoded.mat','encoded_img');
% encoded_img
% imshow(decode(t,encoded_img))

% Initial distribution
rng(1, 'twister')
data_init = normrnd(0,1,[128,1]);
result_img = decode(t,data_init);

% Import the images 
global A
global B
A=imread('Kand5f.jpg');
A=imresize(A,[128,128]);
B=imread('yellowkk.jpg');
B=imresize(B,[128,128]);

% Initialisation 
latentMean=0.0;
sigma=1;
probability=0.02;
power=4;
samples=128;
best = encoded_img;
simA = 1-imDistanceR(A,decode(t,encoded_img),power);
simB = 1-imDistanceR(B,decode(t,encoded_img),power);
bestFeat = fitPlusSimABtimesMeanHue_old(simA,simB,mutImg) 
data_init = encoded_img;
oldVec = data_init;
mutVec = data_init;

% do feature based search.
featureLen=128;
opts=cmaes;
opts.LBounds=zeros(featureLen,1);
opts.UBounds=ones(featureLen,1);
opts.StopFunEvals=100000;
imDistanceR(A,decode(t,encoded_img),4)
% x=cmaes('fitPlusSimABtimesMeanHue',zeros(featureLen,1),ones(featureLen,1)*0.4,opts);
% return
x=cmaes('fitPlusSimABtimesMeanHue',encoded_img,ones(featureLen,1)*1.0,opts);
return

plot_best = [ ]

for i = 1:10000
    mutVec_old = mutVec;
    
    % Mutate
    for j = 1:samples
        if rand < probability  % changed probability to make it not so frequent
            mutVec(j)=mutVec(j)+random('normal', latentMean,sigma*0.1,1,1); %changed sigma to make it sane
        end
    end
    
    % Decode mutated vector
    mutImg = decode(t, mutVec);
    simA = 1-imDistanceR(A,mutImg,power);
    simB = 1-imDistanceR(B,mutImg,power);
    mutFeat = fitPlusSimABtimesMeanHue_old(simA,simB,mutImg); 
    
    % Keep best one
    if mutFeat < bestFeat
        best = mutVec;
        bestFeat = mutFeat;
    else
        mutVec = mutVec_old;
    end
 
    fprintf('%d %.5f \n', i, mutFeat)
    % Print
    if mod(i, 10) == 0
        fprintf('%d %.5f \n', i, bestFeat)
        plot_best = [plot_best bestFeat];
    end
end

plot([1:length(plot_best)], plot_best)
final = decode(t,best)
imshow(final)

%% Close socket
% data = 'disconnect'
% fwrite(t, data)
% fclose(t)
% delete(t);
% clear t
% echotcpip('off');
