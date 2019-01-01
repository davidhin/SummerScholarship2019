function [ fitness ] = fitPlusSimABtimesMeanHue(image)
%Fitness function that maximises difference of similarity of image B
%(SimB)to SimA
% Will tend to produce a smooth image biased toward image B but with some
% new features.
    global t;
    global A;
    global B;
    p = 4;
    image = decode(t, image);
    simA=1.0-imDistanceR(A,image,p);
    simB=1.0-imDistanceR(B,image,p);
    fitness=1.0-simA;
    %fitness=(1.0-simA)/(featureMeanHue(image));
    % fitness=(1.0-(simB))*(1.0-featureMeanHue(image));
    %fitness=(2.0-(plus(simB,simA)))
    %fitness=(2.0-(plus(simB,simA)))/featureMeanHue(image);
    % fitness = rdivide(1.0-simA,1.0-simB);
    % fitness=(rdivide(1.0-simB,1.0-simA))*(1.0-featureMeanHue(image));
    
    % adjHue=peakTransform(featureMeanHue(image),0.1)*0.9;
    %adjHue=peakTransform(featureMeanHue(image),0.15)*0.9;
    % adjHue=peakTransform(featureMeanHue(image),0.5)*0.9;
    % adjHue=peakTransform(featureMeanHue(image),0.7)*0.9;
    % adjHue=peakTransform(featureMeanHue(image),0.9)*0.9;
    % fitness=(max(1.0-simB,1.0-simA))*(1.0-adjHue);
    %fitness=(1.0-simA)/(adjHue);
    %fitness=max(simB,simA))/(adjHue);


end

