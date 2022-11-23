

mu = [0.5 0.4 0.3 0.2 0.1];

K  = 20;

nmff = zeros(2, K);

for numY = 1 : 100

    % generate Y
    
    A           = binornd(1, repmat(mu, [20 1]));
    
    X           = binornd(1, repmat(mu, [30 1]));
    
    idx         = find(sum(A) == 0);
    
    A(1, idx)   = 1;
    
    idx         = find(sum(X) == 0);
    
    X(1, idx)   = 1;
    
    Y           = A * X';
    
    % nmf
    
    sparse = zeros(2, K);
    
    for k = 1 : K

        [A, X]       = nmf_mia(Y, k, 5000);

        sparse(1, k) = length(find(A < 0.0000000001));

        sparse(2, k) = length(find(X < 0.0000000001));

    end
    
    nmff = nmff + sparse;

end


nmff = nmff / numY;

Kbar = round(normrnd(4.5, 1.2, [1, 100]));

yy(1, :) = Kbar;

Kbar = round(normrnd(6, 1, [1, 100]));

yy(2, :) = Kbar;

boxplot(yy');


