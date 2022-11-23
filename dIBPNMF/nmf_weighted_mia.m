function [ A, X, C ] = nmf_weighted_mia( Y, W, K, niter )
%%%%%%
%
%   classical NMF  
%
%   J(A, X) = || W .* (Y - AX) ||^2 
%
%
%%%%%%

%% get dimensions of Y

tic;

Y(Y<0)      = eps;

[row,col]   = size(Y);

%% set parameters

opts        = struct;
opts.tol    = 1e-10;
opts.niter  = niter;
opts.sigma  = 1e-5;
opts.delta  = 1e-5;

%% Initialization 
tic
A = rand(row,K);
X = rand(K,col);


%% main loop

cost_old = 100000000;

cost_old = sum(sum(( W.* (Y-A*X) ).^2));

tic
for it = 1: opts.niter
    
    
    %% devrivative of A 
    Deri_A      = W .*(A * X) * X' - (W .* Y) * X';
    
    % find index of elements in A with negative derivative
    idx_A       = find(Deri_A < 0);
    
    % max(A, sigma)
    A(idx_A)    = max(A(idx_A), opts.sigma);
    
    % compute step size
    eta_A       = A ./ ( ( ( W .*(A * X) * X').^(0.5) + ( W .* Y * X').^(0.5) ) .* (W .* (A * X) * X').^(0.5) + opts.delta ) ;
    
    % gradient decent optimization
    A           = A - eta_A .* Deri_A;
    
    %% devrivative of X    
    Deri_X      =  A' * (W .* (A * X)) - A' * (W .* Y) ;
    
    % find index of elements in X with negative derivative
    idx_X       = find(Deri_X < 0);
    
    % max(X, sigma)
    X(idx_X)    = max(X(idx_X), opts.sigma);
    
    % compute step size
    eta_X       = X ./ ( ( (A'*(W .* (A*X))).^(0.5) + (A'* (W .* Y)).^(0.5) ) .* (A'*(W .* (A*X))).^(0.5) + opts.delta ) ;
    
    % gradient decent optimization
    X           = X - eta_X .* Deri_X;
    
    
    
    %%
    cost    = sum(sum((W.* (Y-A*X)).^2));
       
    costsum = cost ;
    
    C(it)   = costsum;
    
     if it > 1
    
        if costsum > cost_old
            fprintf('ALL iter number: %d, new cost of ALL: %d, old cost of ALL: %d\n',it,costsum,cost_old);
            break;
        else
            stop = abs(cost_old - costsum) <= opts.tol*cost_old;

            fprintf('ALL iter number: %d, new cost of ALL: %d, old cost of ALL: %d\n',it,costsum,cost_old);
            cost_old = costsum;
        end
     else
         fprintf('ALL iter number: %d, new cost of ALL: %d, old cost of ALL: %d\n',it,costsum,cost_old);
         stop = false;
         cost_old = costsum;
     end
 
    
    if stop 
        fprintf('ALL done!');
        break; 
    end
end  

disp(['time of main loop:',num2str(toc)]);

end

