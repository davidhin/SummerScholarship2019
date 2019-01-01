%plots distance from A and B of image transistion from one image to
% another

% set up images
imSize=100;
close all;
% test the distance funciton
A=imread('images/Kand5f.jpg');
A=imresize(A,[imSize,imSize]);
B=imread('images/yellowkk.jpg');
B=imresize(B,[imSize,imSize]);

% swap
%tmp=A;
%A=B;
%B=tmp;

% sim measures
simAB=1.0-imDistance(A,B);
simAA=1.0-imDistance(A,A);
simBA=1.0-imDistance(B,A);
simBB=1.0-imDistance(B,B);

% starting values
As=[simAA,simBA];
Bs=[simAB,simBB];

% plot those points
scatter(As,Bs);

% now image transition
As=[];
Bs=[];
res2=A;
bestSim=simAB+simAA;
for i=1:2000
    tempres=channelMutateTransition(res2,A,B,0.01,20);
    totaloffset=totaloffset+offset;
    simresA=1.0-imDistance(tempres,A);
    simresB=1.0-imDistance(tempres,B);
    combSimRes=simresA+simresB;
    if combSimRes <= bestSim
        disp(['better ',num2str(combSimRes)]);
        As=[As,simresA];
        Bs=[Bs,simresB];
        bestSim=combSimRes;
        res2=tempres;
    end
end
hold on;
% plot those points
scatter(As,Bs,'filled');
figure;
imshow(res2);
