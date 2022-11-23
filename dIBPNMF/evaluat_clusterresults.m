function [ JC, FM, F1 ] = evaluat_clusterresults( c1, c2 )
%EVALUAT_CLUSTERRESULTS Summary of this function goes here
%   Detailed explanation goes here

l = length(c1);

a = 0.0;
b = 0.0;
c = 0.0;
d = 0.0;

for i = 1 : l
    for j = 1 : l
        
        if i ~= j && c1(i) == c1(j) && c2(i) == c2(j)
            a = a + 1;
        end
        
        if i ~= j && c1(i) == c1(j) && c2(i) ~= c2(j)
            b = b + 1;
        end
        
        if i ~= j && c1(i) ~= c1(j) && c2(i) == c2(j)
            c = c + 1;
        end
        
        if i ~= j && c1(i) ~= c1(j) && c2(i) ~= c2(j)
            d = d + 1;
        end
        
    end
end

JC = a/(a+b+c);
FM = ( (a/(a+b)) * (a/(a+c)))^0.5;
F1 = 2*a^2/ (2*a^2 + a*c + a*b);


end

