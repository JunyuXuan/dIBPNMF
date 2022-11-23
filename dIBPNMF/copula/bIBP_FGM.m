% Implementation of FGM copula-basded dependent IBP by MCMC inference
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

function [ L_list, K1_list, K2_list, RealK_list, A_out, X_out, L_max, K_out, alpha_out, rho_out] = bIBP_FGM( max_iteration, N, a, b, Y, Yidx, Ymidx, Ynidx, alpha1, alpha2 )

    
    RealK_list  = [];
    K1_list     = [];
    K2_list     = [];
    
    L_list      = [];
    
    L_max       = -100000000000000; 
    
    A_out = [];
    X_out = [];
    
    % load data
    
    
    % initialize the first inactive feature with 
    disp(' FGM-bIBP-NMF Initialization.................. ');
    
    [K_max, N, mu, Z, rho, A, X, V, alpha, para_alpha, hyperp_alpha, epsilon] = initialization(N, a, b, alpha1, alpha2, Y);
    
    % start Gibbs sampling
    disp(' FGM-bIBP-NMF Start Gibbs sampling .................. ');
     
    num_iteration = 1;
    
    FLAGV = 1;
    FLAGALPHA = 1;
    
    hcp=figure('name','Copula-bIBP-NMF');
    
    while num_iteration <= max_iteration
        
        
        timevalue = tic;
        
        disp([' ---------- FGM-bIBP-NMF interation ', num2str(num_iteration)]);
        
        disp('             ******************** Update mu ');
        timemu = tic;
%         [mu]                        = Update_mu(mu, Z, K_max, N, alpha(1), alpha(2), rho);
                
        [mu]                        = mex_update_mu_FGM(mu, sum(Z{1}), sum(Z{2}), K_max, N(1), N(2), alpha(1), alpha(2), rho);
        
        fprintf('             ------------------------      use time = %d\n', toc(timemu));
        
        disp('             ******************** Update Z ');
        [Z, A, X]                   = Update_Z(Z, N, mu, K_max, A, V, X, Y,  Ymidx, Ynidx, epsilon);
                
        [RealK, K1, K2, idx]        = countK(Z);
        
        disp('             ******************** Update rho ');
        [rho]                       = Update_rho(rho, mu, alpha);
        
        disp('             ******************** Update alpha ');
        [alpha, FLAGALPHA]          = Update_alpha_FGM(alpha, mu, para_alpha, rho, FLAGALPHA);
        
        
        
        disp('             ******************** Update V ');
        [V, A, X, FLAGV]            = Update_V(V, N, Z, K_max, A, X, Y, Ymidx, Ynidx, alpha1, alpha2, epsilon, FLAGV);
        
        disp('             ******************** Update alpha1 and alpha2 ');
        [alpha1, alpha2]            = Update_alpha(alpha1, alpha2, V, hyperp_alpha);
        
        
        disp('             ******************** Compute Likelihood ');
        [L]                         = LogLikelihood(Z, A, X, Y, Yidx, epsilon);
        
        
        
        fprintf('             ******************** Log Likelihood = %f\n',L);
        
        RealK_list                  = [RealK_list RealK];
        K1_list                     = [K1_list K1];
        K2_list                     = [K2_list K2];
        
        L_list                      = [L_list L];
        
        
        if L > L_max
           
            A_out = A(:, idx);
            X_out = X(:, idx);
            K_out = RealK;
            
            alpha_out = alpha;
            rho_out = rho;
            
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
        num_iteration               = num_iteration + 1;
        
        
        fprintf('             ******************** Number of Real K = %d, K1 = %d, K2 = %d, iteration use time: %d, \n',RealK, K1, K2, toc(timevalue));
        
        
    end

end

%-----------------------------------------------------------------
% 
%  randomly set the initial values for latent variables 
% 
%-----------------------------------------------------------------
function [K_max, N, mu, Z, rho, A, X, V, alpha, para_alpha, hyperp_alpha, epsilon] = initialization(N, a, b, alpha1, alpha2, Y)

    % truncated level
    K_max           =   max(N);
        
    % parameters of FGM
    alpha(1)        =   a;
    alpha(2)        =   b;
    
    para_alpha      =   1;
    
    % rho of FGM copula
    rho             =   1;
    
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
    
    epsilon         =   0.0001;
    
    A               =   V1 .* Z_1 ;
    
    X               =   V2 .* Z_2 ;

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
function [Z, A, X] = Update_Z(Z, N, mu, K_max, A, V, X, Y, Ymidx, Ynidx, epsilon)


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
        
        midx        = Ymidx{m};
        
        Z1m(1)      = 1;
        
        TMP1        = Z1m .* V1m * X';
        
        TMP0        = TMP1 - X(:, 1)' * V1m(1);
        
        for k = 1 : K_max
           
            if mu1(k) > 0.000001
            
                log_p_z_1    = log_mu1(k) + sum(log(TMP1(midx) +epsilon) - Ym(midx) ./ (TMP1(midx) +epsilon) ) ;
                log_p_z_0 	 = log_mu1_r(k) + sum(log(TMP0(midx) +epsilon) - Ym(midx) ./ (TMP0(midx) +epsilon) ) ;

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
       
        Z2n         = Z2(n, :);
        V2n         = V2(n, :);
        
        Yn          = Y(:, n);
        
        nidx        = Ynidx{n};
        
        %% sequential 
        Z2n(1)      = 1;
        
        TMP1        = A * transpose(Z2n .* V2n);
        
        TMP0        = TMP1 - A(:, 1) * V2n(1);
        
        for k = 1 : K_max
           
            if mu2(k) > 0.000001
            
                log_p_z_1    = log_mu2(k) + sum(log(TMP1(nidx) +epsilon) - Yn(nidx) ./ (TMP1(nidx) +epsilon) ) ;
                log_p_z_0 	 = log_mu2_r(k) + sum(log(TMP0(nidx) +epsilon) - Yn(nidx) ./ (TMP0(nidx) +epsilon) ) ;

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

%--------------------------------------------------------------------------------
% 
%  Update mu for all features 
%
%       through the chain using M-H
%         
%---------------------------------------------------------------------------------
function [mu] = Update_mu(mu, Z, K_max, N, a, b, rho)

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
        
        Z_1         = Z{1};
        Z_2         = Z{2};

        mu(:, k)    = MH_mu_FGM(k, mu_k_minus, mu(:,k), mu_k_plus, Z_1(:,k), Z_2(:,k), N, K_max, a, b, rho);
                
        %fprintf('-----                   mu(%d) = [%f , %f]:   \n', k, mu(1, k), mu(2, k));
        
    end
    
end

%--------------------------------------------------------------------------------
% 
%  Update V
%         
%---------------------------------------------------------------------------------
function [V, A, X, FLAGV] = Update_V(V, N, Z, K_max, A, X, Y, Ymidx, Ynidx, alpha1, alpha2, epsilon, FLAGV)

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
        
        midx = Ymidx{m};
        
        %%
        for k = 1 : K_max
           
            if Z1m(k) == 1
                           
                V1m(k) = MH_V1_FGM(V1m(k), alpha1, k, Ym, midx, Z1m, V1m, X, epsilon);
            
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
        
        nidx = Ynidx{n};
        
        %%
        for k = 1 : K_max
            
            if Z2n(k) == 1        
                V2n(k) = MH_V2_FGM(V2n(k), alpha2, k, Yn, nidx, Z2n, V2n, A, epsilon);
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
%  update alpha by posterior
% 
%             p(alpha1; ~) =  p(mu1; alpha1, 1) * p(alpha1; para_alpha)
%             
%                            
%             p(alpha2; ~) =  p(mu2; alpha2, 1) * p(alpha2; para_alpha)
%         
%---------------------------------------------------------------------------------
function [alpha, FLAGALPHA] = Update_alpha_FGM(alpha, mu, para_alpha, rho, FLAGALPHA)

    if FLAGALPHA == 1

        alpha(1) = MH_alpha_FGM(1, alpha, para_alpha, mu, rho);
        alpha(2) = MH_alpha_FGM(2, alpha, para_alpha, mu, rho);
    
        FLAGALPHA = 0;
    
    else
       
        alpha(2) = MH_alpha_FGM(2, alpha, para_alpha, mu, rho);
        alpha(1) = MH_alpha_FGM(1, alpha, para_alpha, mu, rho);
        
        FLAGALPHA = 1;
        
    end
            
    fprintf('-------                alpha = [%f, %f] \n ', alpha(1), alpha(2));

end

%--------------------------------------------------------------------------------
% 
%  update copula/rho by 
% 
%             likelihood(mu; rho) * prior(rho)
%             
%                            by MH
%
%         
%---------------------------------------------------------------------------------
function rho = Update_rho(rho, mu, alpha)

    rho = MH_rho_FGM(rho, mu, alpha);
    
    fprintf('-------                rho = %f \n ', rho);

end

function [alpha1, alpha2] = Update_alpha(alpha1, alpha2, V, hyperp_alpha)

    V1 = V{1};
    V2 = V{2};
    
    alpha1 = gamrnd(hyperp_alpha + numel(V1), 1/(1 + sum(sum(V1))));
    
    alpha2 = gamrnd(hyperp_alpha + numel(V2), 1/(1 + sum(sum(V2))));
    
    fprintf('-------                alpha1 = %f,  alpha2 = %f \n ', alpha1, alpha2);
    
end


function [L] = LogLikelihood(Z, A, X, Y, Yidx, epsilon)

    Z1      = Z{1};
    Z2      = Z{2};

    z1_col  = sum(Z1);
    z2_col  = sum(Z2);
        
    idx     = find(z1_col > 0 & z2_col > 0);
    
    % exponential distribution
    rY      = A(:, idx) * X(:, idx)';
    
    L       = sum(sum(log(exppdf(Y(Yidx), rY(Yidx) + epsilon ) + eps )));
    
end




