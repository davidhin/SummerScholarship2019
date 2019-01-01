% Setup socket
% global t
% t = tcpip('129.127.10.18', 8220, 'NetworkRole', 'client');
% t.OutputBufferSize = 2048;
% t.InputBufferSize = 1; % Change this
% fopen(t);
% return

% Initial distribution
rng(1, 'twister')
data_init = normrnd(0,1,[64,1]);
result_img = decode(t,data_init);

% Import the images 
global A
global B
A=imread('Kand5f.jpg');
A=imresize(A,[128,128]);
B=imread('yellowkk.jpg');
B=imresize(B,[128,128]);

% load('../Autoencoder/encoded_64Dense.mat','encoded_img');
% encoded_img = double(encoded_img')
% fitPlusSimABtimesMeanHue(encoded_img)

% options = optimset('PlotFcns',@optimplotfval);
% fminsearch_result = fminsearch('fitPlusSimABtimesMeanHue',encoded_img,options);
% return



% do feature based search.
featureLen=64;
opts=cmaes;
% opts.LBounds=-ones(featureLen,1)*5;
% opts.UBounds=ones(featureLen,1)*5;
% opts.PopSize = 5;
opts.StopFunEvals=20000;
x=cmaes('fitPlusSimABtimesMeanHue',zeros(featureLen,1),ones(featureLen,1)*10,opts);
imshow(decode(t,x))
return

% options = optimset('PlotFcns',@optimplotfval);
% fminsearch_result = fminsearch('fitPlusSimABtimesMeanHue',data_init,options);
% return

latentMean=0.0;
sigma=1;
probability=0.02;
power=4;
samples=64;
best = encoded_img;
simA = 1-imDistanceR(A,decode(t,data_init),power);
simB = 1-imDistanceR(B,decode(t,data_init),power);
bestFeat = fitPlusSimABtimesMeanHue_old(simA,simB,decode(t,data_init))
oldVec = data_init;
mutVec = data_init;
plot_best = [ ]

for i = 1:50000
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

plot([1:length(plot_best)], plot_best);
final = decode(t,best);
imshow(final)

return 


%% Close socket
% data = 'disconnect'
% fwrite(t, data)
% fclose(t)
% delete(t);
% clear t
% echotcpip('off');
