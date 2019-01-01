function [ res ] = rthPowerMean( V,r )
%calculates the rth power mean of a vector
elems=prod(size(V));
V=reshape(V,[1,elems]);
Vr=V.^r;
mVr=sum(Vr)/elems;
res=mVr^(1/r);

end

