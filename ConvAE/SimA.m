function [ fitness ] = fitPlusSimABtimesMeanHue(image)
%Fitness function that maximises difference of similarity of image A
    global t;
    global A;
    global B;
    p = 2;
    simA=1.0-imDistanceR(A,image,p);
    simB=1.0-imDistanceR(B,image,p);
    x = max(simA, simB);
    fitness=2^((x-0.70)*50);
end
