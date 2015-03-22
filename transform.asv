% perform transformation on input variables

X = xdata;
X(:,21:37) = safeLogTransform(X(:,21:37));
X(:,38:42) = safeCatTransform(X(:,38:42));
X(:,43:56) = safeLogTransform(X(:,43:56));
X(:,57:60) = safeCatTransform(X(:,57:60));
X(:,61:65) = safeLogTransform(X(:,61:65));
X(:,66:70) = safeCatTransform(X(:,66:70));
X(:,71) = safeLogTransform(X(:,71));
X(:,72:76) = safeCatTransform(X(:,72:76));
X(:,77:124) = safeLogTransform(X(:,77:124));
X(:,125:166) = safeCatTransform(X(:,125:166));
X(:,167:171) = safeLogTransform(X(:,167:171));

% perform transformation on output values

nmake = 51;
viskey_y = ydata(:,1);
viskey_y = num2str(viskey_y);
make_num = ydata(:,6);
y = zeros(length(X),nmake);
for i = 1:length(viskey_y)
    tmp = strmatch(viskey_y(i,:),viskey_x);
    for j = 1:length(tmp)
        y(tmp(j),make_num(i)) = 1;
    end
    display(i)
end
