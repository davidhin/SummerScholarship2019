function [ res ] = peakTransform(val,targ )
%Scales val to 1.0 at targ and to 0 at both 1 and 0.
% val and targ are in range [0,1]
if targ==1.0
    res=val;
    return
end

if targ==0.0
    res=1.0-val;
    return
end

if val<targ
    res=val*(1.0/targ);
elseif val>targ
    res=(1.0-val)*(1.0/(1.0-targ));
else 
    res=1.0;  % val == targ
end
end

