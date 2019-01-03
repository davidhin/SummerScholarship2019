function [ fitness ] = fitPlusSimABtimesMeanHue(image)
    % Fitness function that maximises difference of similarity of image A
    global t;
    global A;
    global B;
    p = 2;
    simA=imDistanceR(A,image,p);
    simB=imDistanceR(B,image,p);
    x = mean([simA, simB]);
    fitness=2^((1.0-x-0.85)*70);
    fitness=min(5, fitness);
end
