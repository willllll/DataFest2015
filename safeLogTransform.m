function [ xt ] = safeLogTransform( X )
%perform log transformation on matrix X so that log(0) = 0, log(1)>log(0)
%   xt = log(X+1), divide each column with its maximum so that the range is
%   0-1.

xt = log(X+1);
tmp = max(xt);
for i = 1:size(xt,2)
    xt(:,i) = xt(:,i)/tmp(i);
end

end

