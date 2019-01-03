function [ fitness ] = fitPlusSimABtimesMeanHue(image)
    % Fitness function that maximises difference of similarity of image A
    global t;
    global A;
    global B;
    p = 2;
    simA=imDistanceR(A,image,p);
    simB=imDistanceR(B,image,p);
    simA=2^((1.0-simA-0.8)*40);
    simB=2^((1.0-simB-0.8)*40);
    x = mean([simA, simB]);
    % fitness=2^((1.0-x-0.8)*50);
    fitness=min(5, x);
end
