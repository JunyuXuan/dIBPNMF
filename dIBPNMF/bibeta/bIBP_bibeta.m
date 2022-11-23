% Implementation of bivairate beta distribution-basded dependent IBP by MCMC inference
% 
% 
% 
% Input:
%
% Output:
% 
%
%            
% 
% Example of calling function:
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ L_list, K1_list, K2_list, RealK_list, A_out, X_out, L_max, K_out, a_out, b_out] = bIBP_bibeta( max_iteration, N, a, b, Y, alpha1, alpha2 )

    
    RealK_list  = [];
    K1_list     = [];
    K2_list     = [];
    
    L_list      = [];
    
    L_max       = -100000000000000; 
    
    A_out = [];
    X_out = [];
    
    % load data
    
    
    % initialize the first inactive feature with 
    disp(' Bibeta-dIBP-NMF Initialization.................. ');
    
    [K_max, N, mu, Z, A, X, V, a, b, hyperp_bibeta, hyperp_alpha] = initialization(N, a, b, alpha1, alpha2, Y);
    
    % start Gibbs sampling
    disp(' Bibeta-dIBP-NMF Start Gibbs sampling .................. ');
     
    num_iteration = 1;
    
    FLAGAX = 1;
    FLAGAB = 1;
    
    epsilon = 0.0001;
    
    hbb=figure('name','Bibeta-bIBP-NMF');
    
    while num_iteration <= max_iteration
        
        
        
        timevalue = tic;
        
        disp([' ---------- Bibeta-dIBP-NMF interation ', num2str(num_iteration)]);
        
        disp('             ******************** Update mu ');
        timemu = tic;
%         [mu]                        = Update_mu(mu, Z, K_max, N, a, b);
        
        [mu]                        = mex_update_mu_bibeta(mu, sum(Z{1}), sum(Z{2}), K_max, N(1), N(2), a, b);
        fprintf('             ------------------------      use time = %d\n', toc(timemu));
        
        disp('             ******************** Update Z ');
        timez = tic;
        [Z, A, X]                   = Update_Z(Z, N, mu, K_max, A, V, X, Y, epsilon);
%         [Z, A, X]                   = Update_Z2(Z, N, mu, K_max, A, V, X, Y, epsilon);
        
        fprintf('             ------------------------      use time = %d\n', toc(timez));
        
        [RealK, K1, K2, idx]        = countK(Z);
        
        disp('             ******************** Update a, b ');
        timeab = tic;
        [a, b, FLAGAB]              = Update_ab(a, b, mu, hyperp_bibeta, FLAGAB);
        
        fprintf('             ------------------------      use time = %d\n', toc(timeab));
        
        disp('             ******************** Update V ');
        timev = tic;
        [V, A, X]                   = Update_V(V, N, Z, K_max, A, X, Y, alpha1, alpha2, epsilon);
        
        fprintf('             ------------------------      use time = %d\n', toc(timev));
        
        disp('             ******************** Update alpha1 and alpha2 ');
        timealpha = tic;
        [alpha1, alpha2]            = Update_alpha(alpha1, alpha2, V, hyperp_alpha);
        
        fprintf('             ------------------------      use time = %d\n', toc(timealpha));
        
        
        disp('             ******************** Compute Likelihood ');
        timel = tic;
        [L]                         = LogLikelihood(Z, A, X, Y, epsilon);
        
        fprintf('             ------------------------      use time = %d\n', toc(timel));
        
        
        fprintf('             ******************** Log Likelihood = %f\n',L);
        
        RealK_list                  = [RealK_list RealK];
        K1_list                     = [K1_list K1];
        K2_list                     = [K2_list K2];
        
        L_list                      = [L_list L];
        
        
        if L > L_max
           
            A_out = A(:, idx);
            X_out = X(:, idx);
            K_out = RealK;
            
            a_out = a;
            b_out = b;
                        
            L_max = L;
        end
        
        %% draw figure
        
        YY = A(:, idx) * X(:, idx)';
        
        
        subplot(4, 2, 1);
        imagesc(Y);
        colorbar();
        title('Original Y')
        
        
        subplot(4, 2, 2);
        imagesc(YY);
        colorbar();
        title('Reconstructed Y')
        
        subplot(4, 2, 3);
        imagesc(A(:, idx));
        colorbar();
        title('A')
        
        subplot(4, 2, 4);
        imagesc(X(:, idx));
        colorbar();
        title('X')
        
        subplot(4, 2, 5);
        plot(L_list);
        title('L list')
        
        subplot(4, 2, 6);
        plot(RealK_list);
        title('K list')
        
        subplot(4, 2, 7);
        plot(K1_list);
        title('K1 list')
        
        subplot(4, 2, 8);
        plot(K2_list);
        title('K2 list')
        
        %%%%%
%         hh=colorbar;
%         set(hh,'position',[0.92 0.1 0.02 0.82])
        
        drawnow;
        hold on;
        
        
        %%
        
        fprintf('             ******************** Number of Real K = %d, K1 = %d, K2 = %d,   iteration use time: %d \n',RealK, K1, K2, toc(timevalue));
        
        
        num_iteration               = num_iteration + 1;
    end

end

%-----------------------------------------------------------------
% 
%  randomly set the initial values for latent variables 
% 
%-----------------------------------------------------------------
function [K_max, N, mu, Z, A, X, V, a, b, hyperp_bibeta, hyperp_alpha] = initialization(N, a, b, alpha1, alpha2, Y)

    % truncated level
    K_max           =   max(N);
        
    % parameters of bivariate beta distribution (c = 1)
    a               =   a;
    b               =   b;
    
    hyperp_bibeta   =   1;
    
    hyperp_alpha    =   1;

    % number of data
    N1              =   N(1);
    
    N2              =   N(2);
     
    % mu 
    mu1             =   betarnd(a, 1, [1 K_max]);
    
    mu1             =   cumprod(mu1);
    
    mu2             =   betarnd(b, 1, [1 K_max]);
    
    mu2             =   cumprod(mu2);
    
    mu              =   [mu1; mu2];
    
    % IBP matrix Z[t][n][k]
    Z_1             =   binornd(1, repmat(mu1, [ N(1) 1]));
    
    Z_2             =   binornd(1, repmat(mu2, [ N(2) 1]));
            
    Z               =   {Z_1, Z_2};
       
    V1              =   gamrnd(1, 1/alpha1, [N1 K_max]);
    
    V2              =   gamrnd(1, 1/alpha2, [N2 K_max]);
    
    V               =   {V1, V2};
    
    A               =   V1 .* Z_1;
    
    X               =   V2 .* Z_2;

end

%-----------------------------------------------------------------
% 
%  count the number of columns with Z > 0
% 
%-----------------------------------------------------------------
function [K, K1, K2, idx] = countK(Z)

    Z1      = Z{1};
    Z2      = Z{2};

    z1_col  = sum(Z1);
    z2_col  = sum(Z2);
        
    idx     = find(z1_col > 0 & z2_col > 0);
    
    K       = length(idx);
    
    K1      = length(find(z1_col > 0));
    
    K2      = length(find(z2_col > 0));
    
    %fprintf('-------              real K  = %d \n ', K);
    
end


%--------------------------------------------------------------------------------
% 
%  Update Z  
%         
%---------------------------------------------------------------------------------
function [Z, A, X] = Update_Z(Z, N, mu, K_max, A, V, X, Y, epsilon)

    timez1 = tic;

    N1          = N(1);
    N2          = N(2);
    
    Z1          = Z{1};
    Z2          = Z{2};
    
    V1          = V{1};
    V2          = V{2};
    
    mu1         = mu(1, :);
    mu2         = mu(2, :);
    
    log_mu1     = log(mu(1, :) +eps);
    log_mu2     = log(mu(2, :) +eps);
    
    log_mu1_r   = log(1 - mu(1, :) +eps);
    log_mu2_r   = log(1 - mu(2, :) +eps);
    
    
    parfor m = 1 : N1
       
        Z1m         = Z1(m, :);
        V1m         = V1(m, :);
        
        Ym          = Y(m, :);
        
        Z1m(1)      = 1;
        
        TMP1        = Z1m .* V1m * X';
        
        TMP0        = TMP1 - X(:, 1)' * V1m(1);
        
        for k = 1 : K_max
           
            if mu1(k) > 0.000001
            
                log_p_z_1    = log_mu1(k) + sum(log(TMP1 +epsilon) - Ym ./ (TMP1 +epsilon) ) ;
                log_p_z_0 	 = log_mu1_r(k) + sum(log(TMP0 +epsilon) - Ym ./ (TMP0 +epsilon) ) ;

                p            = 1 ./ ( 1 + exp(log_p_z_0 - log_p_z_1));

                Z1m(k)       = binornd(1, p);


                if k < K_max

                    if Z1m(k) == 1
                       TMP0 = TMP1;                                
                    end

                    if Z1m(k+1) == 1

                        TMP1 = TMP0;

                        TMP0 = TMP1 - X(:, k+1)' * V1m(k+1);

                    else

                        TMP1 = TMP0 + X(:, k+1)' * V1m(k+1);
                    end

                end
                
            else
               
                Z1m(k)   = 0;
                
            end
            
        end
        
%         Z1m = mex_update_Z1_bibeta(Z1m, V1m, K_max, mu1, Y(m, :), X, N2, epsilon, TMP0);
        
        Z1(m, :) = Z1m;
        
        %fprintf(' Z1:    m = %d  \n',m);
        
    end
    
    A = Z1 .* V1;
    
    fprintf(' Z1:    use time = %d  \n',toc(timez1));
    timez2 = tic;
    
    parfor n = 1 : N2
       
        Z2n = Z2(n, :);
        V2n = V2(n, :);
        
        Yn  = Y(:, n);
        
        %% sequential 
        Z2n(1)      = 1;
        
        TMP1        = A * transpose(Z2n .* V2n);
        
        TMP0        = TMP1 - A(:, 1) * V2n(1);
        
        for k = 1 : K_max
           
            if mu2(k) > 0.000001
            
                log_p_z_1    = log_mu2(k) + sum(log(TMP1 +epsilon) - Yn ./ (TMP1 +epsilon) ) ;
                log_p_z_0 	 = log_mu2_r(k) + sum(log(TMP0 +epsilon) - Yn ./ (TMP0 +epsilon) ) ;

                p            = 1 ./ ( 1 + exp(log_p_z_0 - log_p_z_1));

                Z2n(k)       = binornd(1, p);

                if k < K_max

                    if Z2n(k) == 1
                       TMP0 = TMP1;                                
                    end

                    if Z2n(k+1) == 1

                        TMP1 = TMP0;

                        TMP0 = TMP1 - A(:, k+1)* V2n(k+1);

                    else

                        TMP1 = TMP0 + A(:, k+1)* V2n(k+1);
                    end

                end
            else
                Z2n(k)       = 0;
            end
            
        end
        
%         Z2n = mex_update_Z2_bibeta(Z2n, V2n, K_max, mu2, Y(:, n), A, N1, epsilon);
        
        %fprintf(' Z2:    n = %d  \n',n);
        
        
        Z2(n, :) = Z2n;
        
        

    end
    
    X = Z2 .* V2;
    
    
    fprintf(' Z2:    use time = %d  \n',toc(timez2));
    
    if sum(sum(Z1)) == 0
       
        Z1(:, 1) = 1;
        
    end
    
    if sum(sum(Z2)) == 0
       
        Z2(:, 1) = 1;
        
    end
    
    Z = {Z1, Z2};
                  
end


function [Z, A, X] = Update_Z2(Z, N, mu, K_max, A, V, X, Y, epsilon)
    
    N1          = N(1);
    N2          = N(2);
    
    Z1          = Z{1};
    Z2          = Z{2};
    
    V1          = V{1};
    V2          = V{2};
    
    log_mu1     = log(mu(1, :) +eps);
    log_mu2     = log(mu(2, :) +eps);
    
    log_mu1_r   = log(1 - mu(1, :) +eps);
    log_mu2_r   = log(1 - mu(2, :) +eps);

    
    for k = 1 : K_max

        Xk           = X(:, k);

        V1k          = V1(:, k);

        Z1(:, k)     = 0;

        TMP10        = Z1 .* V1 * X';
        
        TMP11        = repmat(Xk', [N1 1]) .* repmat(V1k, [1 N2])+ TMP10;

        log_p_z_1    = log_mu1(k) + sum(log(TMP11 +epsilon) - Y ./ (TMP11 +epsilon) , 2);
        log_p_z_0 	 = log_mu1_r(k) + sum(log(TMP10 +epsilon) - Y ./ (TMP10 +epsilon) , 2);

        p            = 1 ./ ( 1 + exp(log_p_z_0 - log_p_z_1));

        Z1(:, k)     = binornd(1, p);
        
        A(:, k)      = Z1(:, k) .* V1(:, k);

        %
        

        Ak           = A(:, k);

        V2k          = V2(:, k);

        Z2(:, 1) 	 = 0;
    
        TMP20        = A * transpose(Z2 .* V2);
        
        TMP21        = repmat(Ak, [1 N2]) .* repmat(V2k', [N1, 1]) + TMP10;
        
        log_p_z_1    = log_mu2(k) + sum(log(TMP21 +epsilon) - Y ./ (TMP21 +epsilon) , 1);
        log_p_z_0 	 = log_mu2_r(k) + sum(log(TMP20 +epsilon) - Y ./ (TMP20 +epsilon) , 1);

        p            = 1 ./ ( 1 + exp(log_p_z_0 - log_p_z_1));

        Z2(:, k)     = binornd(1, p);
        
        X(:, k)      = Z2(:, k) .* V2(:, k);

        
        %
        
        
        
        fprintf(' k = %d  \n',k);
        
    end
    
    Z = {Z1, Z2};
    
    

end
%--------------------------------------------------------------------------------
% 
%  Update mu for all features 
%
%       through the chain using M-H
%         
%---------------------------------------------------------------------------------
function [mu] = Update_mu(mu, Z, K_max, N, a, b)


    Z_1         = Z{1};
    Z_2         = Z{2};

    for k = 1 : K_max
                       
        if k == 1
            mu_k_minus  = ones(2, 1);
        else
            mu_k_minus  = mu(:, k-1);
        end
        
        if k == K_max
            mu_k_plus   = zeros(2, 1);
        else
            mu_k_plus   = mu(:, k+1);
        end
                 
        mu(:, k)    = MH_mu_bibeta(k, mu_k_minus, mu(:,k), mu_k_plus, Z_1(:,k), Z_2(:,k), N, K_max, a, b);
        
    end
    
end

%--------------------------------------------------------------------------------
% 
%  Update V
%         
%---------------------------------------------------------------------------------
function [V, A, X] = Update_V(V, N, Z, K_max, A, X, Y, alpha1, alpha2, epsilon)

    tic;

    N1          = N(1);
    N2          = N(2);
    
    Z1          = Z{1};
    Z2          = Z{2};
    
    V1          = V{1};
    V2          = V{2};
      
    timea       = tic;
       
    parfor m = 1 : N1
       
        Z1m = Z1(m, :);
        V1m = V1(m, :);
        Ym  = Y(m, :);
        
        %%
        for k = 1 : K_max
           
            if Z1m(k) == 1
                           
                V1m(k) = MH_V1_FGM(V1m(k), alpha1, k, Ym, Z1m, V1m, X, epsilon);
            
            else
                V1m(k) = gamrnd(1, 1/alpha1);
            
            end
            
        end
        
        V1(m, :) = V1m;
        
        %%  V1m = mex_Update_V_FGM(K_max, V1m, Z1m, Y(m, :), X, N1, N2, alpha1, epsilon)
%         V1(m, :) = mex_update_V1_FGM(K_max, V1m, Z1m, Y(m, :), X, N1, N2, alpha1, epsilon);
        
    end
    
    A = Z1 .* V1;
    
    fprintf(' A:    use time = %d  \n',toc(timea));
    
    timex = tic;
    
    parfor n = 1 : N2
       
        Z2n = Z2(n, :);
        V2n = V2(n, :);
        Yn  = Y(:, n);
        
        %%
        for k = 1 : K_max
            
            if Z2n(k) == 1        
                V2n(k) = MH_V2_FGM(V2n(k), alpha2, k, Yn, Z2n, V2n, A, epsilon);
            else
                V2n(k) = gamrnd(1, 1/alpha2);
            end
        end
        
        V2(n, :) = V2n;
        
        %%  V1m = mex_Update_V_FGM(K_max, V1m, Z1m, Y(m, :), X, N1, N2, alpha1, epsilon)
% %         V2(n, :) = mex_update_V2_FGM(K_max, V2n, Z2n, Y(:, n), A, N1, N2, alpha2, epsilon);
        
    end
    
    X = Z2 .* V2;
        
    V = {V1, V2};

    fprintf(' X:    use time = %d  \n',toc(timex));
    
    fprintf('-----                   V use time:  %d \n', toc);
    
end


%--------------------------------------------------------------------------------
% 
%  update [a, b] by MH
% 
%              bibeta(mu; a, b) * gamma(a; hyperp_bibeta, 1) * gamma(b; hyperp_bibeta, 1)
%             
%         
%---------------------------------------------------------------------------------
function [a, b, FLAGAB] = Update_ab(a, b, mu, hyperp_bibeta, FLAGAB)

    if FLAGAB == 1

        a   = MH_a_bibeta(a, b, mu, hyperp_bibeta);

        b   = MH_b_bibeta(a, b, mu, hyperp_bibeta);
        
        FLAGAB = 0;
    else


        b   = MH_b_bibeta(a, b, mu, hyperp_bibeta);

        a   = MH_a_bibeta(a, b, mu, hyperp_bibeta);

        FLAGAB = 1;
    end
   
    fprintf('-------                a = %f,  b = %f \n ', a, b);

end


function [alpha1, alpha2] = Update_alpha(alpha1, alpha2, V, hyperp_alpha)

    V1 = V{1};
    V2 = V{2};
    
    alpha1 = gamrnd(hyperp_alpha + numel(V1), 1/(1 + sum(sum(V1))));
    
    alpha2 = gamrnd(hyperp_alpha + numel(V2), 1/(1 + sum(sum(V2))));
    
    fprintf('-------                alpha1 = %f,  alpha2 = %f \n ', alpha1, alpha2);
    
end


function [L] = LogLikelihood(Z, A, X, Y, epsilon)

    Z1      = Z{1};
    Z2      = Z{2};

    z1_col  = sum(Z1);
    z2_col  = sum(Z2);
        
    idx     = find(z1_col > 0 & z2_col > 0);
    
    % exponential distribution
    L       = sum(sum(log(exppdf(Y, A(:, idx) * X(:, idx)' + epsilon ) + eps )));
    
end




