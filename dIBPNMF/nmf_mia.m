function [ A, X ] = nmf_mia( Y, N, iteration)
%%%%%%
%
%   classical NMF  
%
%   J(A, X) = || Y - AX ||^2 
%
%
%%%%%%

%% get dimensions of Y

tic;
Y(Y<0) = eps;

[row,col]=size(Y);

%% set parameters

opts = struct;
opts.tol = 1e-10;
opts.niter = iteration;

alpha_a = 1;
alpha_x = 1;

%% Initialization 
tic
A = rand(row,N);
X = rand(N,col);


%% main loop

cost_old = 100000000;
tic
for it = 1: opts.niter
    
    
    %% 
%     X = max(eps, X .* ( max(eps, (A'*Y - alpha_a) ./ (A'*A*X))  ).^ 0.5 );
%     A = max(eps, A .* ( max(eps, (Y*X' - alpha_x) ./ (A*X*X'))  ).^ 0.5 );
    
    X = max(eps, X .* ( max(eps, (A'*Y ) ./ (A'*A*X))  ).^ 0.5 );
    A = max(eps, A .* ( max(eps, (Y*X' ) ./ (A*X*X'))  ).^ 0.5 );

    %%
    cost = sum(sum((Y-A*X).^2));
       
    costsum = cost ;
    
    C(it) = costsum;
    
     if it > 100
    
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

