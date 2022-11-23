% Implementation of Depentent IBP by MCMC inference
%
% from S. Willianmnson, P. Orbanz, Z. Ghahramani,(2007)
%      'Dependent Indian Buffet Process'
%
%
% Input:
%
% Output:
%
%
%
% Example of calling function:
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [  L_list, K1_list, K2_list, K_list, A_out, X_out, L_max, K_out ] = dIBP( max_iteration, Y )

    % load data
    %     data        = loaddata();

    L_list      = [];
    K_list      = [];
    K1_list     = [];
    K2_list     = [];

    L_max       = -100000000000000;

    % initialize the first inactive feature with
    disp(' GP-dIBP-NMF Initialization.................. ');

    [K_max, alpha, alpha_para, T, N, sigma, s, s_para, rho, Z, h, g, mu, A, X, V, alpha1, alpha2, hyperp_alpha] = initialization(Y);

    % start Gibbs sampling
    disp(' GP-dIBP-NMF Start Gibbs sampling .................. ');

    FLAGV           = 1;

    num_iteration   = 1;
        
    epsilon         = 0.01;
    
    hgp=figure('name','GP-bIBP-NMF');

    while num_iteration <= max_iteration
        
        
        timevalue = tic;
        
        disp([' ---------- GP-dIBP-NMF interation ', num2str(num_iteration)]);

        disp('             ******************** Update Sigma ');
        [Sigma]                     = Update_Sigma(K_max, T, sigma, s, g);

        disp('             ******************** Update mu ');
%         [mu]                        = Update_mu(mu, g, Sigma, rho, K_max, alpha, Z, T, N);
        mutime = tic;
        [mu]                        = mex_update_mu(mu, g(1, :), g(2, :), Sigma(:, 1, 1), Sigma(:, 2, 2), rho, K_max, alpha, sum(Z{1}), sum(Z{2}), T, N(1), N(2));

        fprintf('-------              mu takes time = %d \n ', toc(mutime));
        
        disp('             ******************** Update g ');
        [g]                         = Update_g(T, g, Sigma, rho, h, K_max, N);

        disp('             ******************** Update h ');
        [h]                         = Update_h2(N, K_max, g, mu, rho, Z, Sigma);

        disp('             ******************** Update s ');
        [s]                         = Update_s(s, T, K_max, g, sigma, s_para);
        
        
        disp('             ******************** Update alpha ');
        [alpha]                     = Update_alpha(alpha, alpha_para, K_max, mu(K_max), mu);
        
        disp('             ******************** Update Z ');
        [Z, A, X]                   = Update_Z(Z, N, mu, K_max, A, V, X, Y, epsilon, Sigma, rho, g);



        [K1, K2, K, idx]            = countK(Z);

        
        disp('             ******************** Update V ');
        [V, A, X, FLAGV]            = Update_V(V, N, Z, K_max, A, X, Y, alpha1, alpha2, epsilon, FLAGV);

        disp('             ******************** Update alpha1 and alpha2 ');
        [alpha1, alpha2]            = Update_alphaV(alpha1, alpha2, V, hyperp_alpha);


        disp('             ******************** Compute Likelihood ');
        [L]                         = LogLikelihood(Z, A, X, Y, epsilon);


        fprintf('             ******************** Log Likelihood = %f\n',L);


        K_list(num_iteration)       = K;

        K1_list(num_iteration)      = K1;

        K2_list(num_iteration)      = K2;

        L_list(num_iteration)       = L;


        %
        if L > L_max

            A_out = A(:, idx);
            X_out = X(:, idx);
            K_out = K;

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
        plot(K_list);
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
        
        fprintf('      ~~~~~~~~~~~~~~~~~~~~~~~~  K1 = %d,  K2 = %d,  K = %d, iteration use time: %d \n', K1, K2, K, toc(timevalue));
        
        num_iteration               = num_iteration + 1;

    end


end

%-----------------------------------------------------------------
%
%  randomly set the initial values for latent variables
%
%-----------------------------------------------------------------
function [K_max, alpha, alpha_para, T, N, sigma, s, s_para, rho, Z, h, g, mu, A, X, V, alpha1, alpha2, hyperp_alpha] = initialization(data)

    % number of data
    N           = [size(data, 1), size(data, 2)];

    N1          = N(1);

    N2          = N(2);

    %
    alpha       = 1;

    % alpha ~ Gamma(alpha_para, 1)
    alpha_para  = 1;

    % truncate level
    K_max       = max(N);

    % number of covariates
    T           = 2;

    % parameter of covariance function Sigma
    sigma       = 1;

    % parameter of covariance function Sigma
    s           = repmat(10, 1, K_max);

    % s ~ normal(s_para, 2)
    s_para      = 1;

    % covarance of Gamma
    rho         = 1;

    

    % stick weights
    mu          = betarnd(alpha, 1, [1 K_max]);

    mu          = cumprod(mu);

    % IBP matrix Z[t][n][k]

    Z_1         =   binornd(1, repmat(mu, [ N(1) 1]));

    Z_2         =   binornd(1, repmat(mu, [ N(2) 1]));

    Z           =   {Z_1, Z_2};

    % h functions GP at matrix level
    h1          = rand(N(1), K_max);

    h2          = rand(N(2), K_max);

    h           = {h1, h2};

    % g functions GP at the top level
    g           = rand(T, K_max);


    % likelihood
    hyperp_alpha = 1;

    alpha1      = 1;

    alpha2      = 1;

    V1          =   gamrnd(1, 1/alpha1, [N1 K_max]);

    V2          =   gamrnd(1, 1/alpha2, [N2 K_max]);

    V           =   {V1, V2};

    A           =   V1 .* Z_1;

    X           =   V2 .* Z_2;

end

%-----------------------------------------------------------------
%
%  count the number of columns with Z > 0
%
%-----------------------------------------------------------------
function [K1, K2, K, idx] = countK(Z)

    Z1      = Z{1};
    Z2      = Z{2};

    z1_col  = sum(Z1);
    z2_col  = sum(Z2);



    idx     = find(z1_col > 0 );

    K1      = length(idx);

    idx     = find( z2_col > 0);

    K2      = length(idx);

    idx     = find(z1_col > 0 & z2_col > 0);

    K       = length(idx);

    fprintf('-------              K  = %d  K1 = %d  K2 = %d \n ', K, K1, K2);

end

%-----------------------------------------------------------------
%
%  sample stick weights b(k) by Metroplis-Hasting
%
%
%
%-----------------------------------------------------------------
function mu = Update_mu(mu, g, Sigma, rho, K, alpha, z, T, N)
        
    timemu = tic;
        
    z1 = z{1};
    z2 = z{2};
    
    for k = 1 : K

        if k == 1                   %k, b_m, b_p, b_K, sigma1, sigma2, rho, g1, g2, alpha, z1, z2, T, N
            sample     = MH_mu(k, 1, mu(k+1), mu(K), Sigma(k,1,1), Sigma(k,2,2), rho, g(1, k), g(2, k), alpha, sum(z1(:, k)), sum(z2(:, k)), T, N);

        else if k== K
                sample = MH_mu(k, mu(k-1), 0, mu(K), Sigma(k,1,1), Sigma(k,2,2), rho, g(1, k), g(2, k), alpha, sum(z1(:, k)), sum(z2(:, k)), T, N);
            else
                sample = MH_mu(k, mu(k-1), mu(k+1), mu(K), Sigma(k,1,1), Sigma(k,2,2), rho, g(1, k), g(2, k), alpha, sum(z1(:, k)), sum(z2(:, k)), T, N);
            end
        end

        mu(k)   = sample;

    end
        
    fprintf('-------              mu takes time = %d \n ', toc(timemu));
        
end

%---------------------------------------------------------------
%
%  sample function values g by MH
%
%       g(k) ~ N(g(k) | 0, Sigma)* TT TT N( h(n,k,t) | g(k,t), Gamma(t,t))
%
%
%
%---------------------------------------------------------------
function g_new = Update_g(T, g, Sigma, rho, h, K, N)

    tic;

    g_new  = zeros(T, K);

    h1 = h{1};
    h2 = h{2};
    
    parfor k = 1 : K
        
        %g_new(:, k) = MH_g(k, g, T, N, h, rho, reshape(Sigma(k, :, :), [T T]));
        
        g_new(:, k) = mex_MH_g(k, g, T, N(1), N(2), h1(:, k), h2(:, k), rho, reshape(Sigma(k, :, :), [T T]));
        
    end

    fprintf('-------              g takes time = %d \n ', toc);

end

%--------------------------------------------------------------------------------
% 
% sample binary matrix Z
%
%       z(n,k,t) ~ bernoulli(gamma(k, t))* datalikelihood
%         
%---------------------------------------------------------------------------------
function [Z, A, X] = Update_Z(Z, N, b, K_max, A, V, X, Y, epsilon, Sigma, rho, g)

    sigma_tt     = reshape(Sigma(:, 1, 1), [K_max 1]);

    gamma_kt1    = normcdf(norminv(b, 0, sigma_tt'+rho) - g(1,:), 0, rho);

    sigma_tt2    = reshape(Sigma(:, 2, 2), [K_max 1]);
        
    gamma_kt2    = normcdf(norminv(b,0,sigma_tt2'+rho) - g(2,:), 0, rho);
            

    timez1 = tic;

    N1          = N(1);
    N2          = N(2);
    
    Z1          = Z{1};
    Z2          = Z{2};
    
    V1          = V{1};
    V2          = V{2};
    
    mu1         = gamma_kt1;
    mu2         = gamma_kt2;
    
    log_mu1     = log(gamma_kt1 +eps);
    log_mu2     = log(gamma_kt2 +eps);
    
    log_mu1_r   = log(1 - gamma_kt1 +eps);
    log_mu2_r   = log(1 - gamma_kt2 +eps);
    
    
    parfor m = 1 : N1
       
        Z1m = Z1(m, :);
        V1m = V1(m, :);
        
        Ym  = Y(m, :);
        
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

%---------------------------------------------------------------
%
%  sample function values h
%
%       h(n,k,t) ~ Gaussian(h | g, Gamma(t,t))
%
%---------------------------------------------------------------
function h = Update_h(N, K, g, b, rho, z, Sigma)

    tic;

    z1 = z{1};
    z2 = z{2};

    h1 = zeros(N(1), K);

    t  = 1;

    parfor n = 1 : N(1)

        for k = 1 : K

            sigma_tt    = Sigma(k,t, t);

            b_t_k       = norminv(b(k),0,sigma_tt+rho);

            cdf_bk      = normcdf(b_t_k, g(t, k), rho);

            if z1(n, k) > 0
                u           = unifrnd(0, cdf_bk);

                h1(n, k)    = norminv(u, g(t, k), rho);
            else
                u           = unifrnd(cdf_bk, 1);

                h1(n, k)    = norminv(u, g(t, k), rho);
            end
        end

    end

    h2 = zeros(N(2), K);

    t  = 2;

    parfor n = 1 : N(2)

        for k = 1 : K

            sigma_tt    = Sigma(k,t, t);

            b_t_k       = norminv(b(k),0,sigma_tt+rho);

            cdf_bk      = normcdf(b_t_k, g(t, k), rho);

            if z2(n, k) > 0
                u           = unifrnd(0, cdf_bk);

                h2(n, k)    = norminv(u, g(t, k), rho);
            else
                u           = unifrnd(cdf_bk, 1);

                h2(n, k)    = norminv(u, g(t, k), rho);
            end
        end

    end

    h = {h1, h2};

    fprintf('-------              h takes time = %d \n ', toc);
end

%---------------------------------------------------------------
%
%  sample function values h
%
%       h(n,k,t) ~ Gaussian(h | g, Gamma(t,t))
%
%---------------------------------------------------------------
function h = Update_h2(N, K, g, b, rho, z, Sigma)

    tic;

    z1 = z{1};
    z2 = z{2};

    h1 = zeros(N(1), K);
    h2 = zeros(N(2), K);
    
    parfor k = 1 : K
        
       h1k         = h1(:, k);
       h2k         = h2(:, k);
        
       sigma_k     = [Sigma(k,1, 1) Sigma(k,2, 2)];
               
       b_k         = b(k);
       
       b_t_k       = norminv(b_k, 0, sigma_k+rho);
       
       cdf_bk      = normcdf(b_t_k, g(:, k)', rho);
       
       idx1        = find(z1(:, k) > 0);
       idx2        = find(z2(:, k) > 0);
       
       u1          = unifrnd(0, cdf_bk(1), [length(idx1) 1]);
       u2          = unifrnd(0, cdf_bk(2), [length(idx2) 1]);
       
       h1k(idx1)   = norminv(u1, g(1, k), rho);
       
       h2k(idx2)   = norminv(u2, g(2, k), rho);
       
       h1(:, k)    = h1k;
       h2(:, k)    = h2k;
       
    end
    
    h = {h1, h2};

    fprintf('-------              h takes time = %d \n ', toc);
end


%---------------------------------------------------------------
%
%  sample alpha by M-H
%
%       alpha ~ gamma(a, 1) * p( b(1:K) | alpha) ,
%
%       p( b(1:K) | alpha) = alpha^K * b_K^alpha * TT b_k^-1
%
%---------------------------------------------------------------
function alpha = Update_alpha(alpha, alpha_para, K, b_K, b)

    tic;

    alpha = MH_alpha( alpha, alpha_para, K, b_K, b);

    fprintf('-------              alpha takes time = %d \n ', toc);
end

%---------------------------------------------------------------
%
%  sample s by Hybrid Monte Carlo
%
%       p(s) ~ normpdf(g_k, 0, Sigma(k)+rho)*gampdf(s|a,1)
%
%---------------------------------------------------------------
function s_new = Update_s(s, T, K, g, sigma, s_para)
    tic;

    s_new = zeros(1, K);

    parfor k = 1 : K
        
        %s_new(k) = HMC_s(s(k), T, sigma, s_para, g(:, k));
        
        s_new(k) = mex_HMC_s(s(k), T, sigma, s_para, g(:, k));
                
        %disp(['                             ------------ k = ', num2str(k),'   s(k) = ', num2str(s(k))]);
    end

    fprintf('-------              s takes time = %d \n ', toc);
end

%---------------------------------------------------------------
%
%  renew Sigma by new s
%
%---------------------------------------------------------------
function [Sigma]  = Update_Sigma(K, T, sigma, s, g)

    tic;

    Sigma = zeros(K, T, T);

    parfor k = 1 : K

        Sigma(k, :, :)  = sigma * exp(-1 * (( ( repmat(g(:, k)', [T 1]) - repmat(g(:, k), [1 T]) ) .^ 2 )./s(k)) ) + 10*ones(T);

        a               = eigs(reshape(Sigma(k,:,:), [T T]));

        idx             = find(a < 0, 1);

        if ~isempty(idx)
            disp('                      singler ! ');

        end

    end

    fprintf('-------              Sigma takes time = %d \n ', toc);
end



%--------------------------------------------------------------------------------
% 
%  Update V
%         
%---------------------------------------------------------------------------------
function [V, A, X, FLAGV] = Update_V(V, N, Z, K_max, A, X, Y, alpha1, alpha2, epsilon, FLAGV)

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



function [alpha1, alpha2] = Update_alphaV(alpha1, alpha2, V, hyperp_alpha)

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



