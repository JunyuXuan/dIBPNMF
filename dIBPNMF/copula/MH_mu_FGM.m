function [mu] = MH_mu_FGM(k, mu_k_minus, mu_k, mu_k_plus, Z_k_1, Z_k_2, N, K_max, a, b, rho)

    % SOME CONSTANTS
    nSamples    = 40;
    burnIn      = 10;
    interval    = 3;
    
    % INTIIALZE SAMPLER
    x               = zeros(nSamples, 2);
    x(1,:)          = mu_k;
    t               = 1;
    accteptednum    = 0;    
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
%         tic;
        
        xStar   = sample_q(mu_k_minus, mu_k_plus, a, b);
        
        x_t_1   = reshape(x(t-1, :), [1 2]);
        
%         fprintf('-----   proposal  toc:  %d \n', toc);
        
        % CORRECTION FACTOR
%         log_q_xstar     = (a-1)*log(xStar(1)) + (b-1)*log(xStar(2)) ;
%         
%         log_q_st1       = (a-1)*log(x_t_1(1)) + (b-1)*log(x_t_1(2)) ;
%                         
%         c               = exp( log_q_st1 - log_q_xstar);  
        c = 1;
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
%         tic;
        
        log_p_xstar     = log_likelihood_mu( xStar, mu_k_minus, mu_k_plus, Z_k_1, Z_k_2, N, a, b, rho, k, K_max);

        log_p_st1       = log_likelihood_mu( x_t_1, mu_k_minus, mu_k_plus, Z_k_1, Z_k_2, N, a, b, rho, k, K_max);

        p               = exp( log_p_xstar - log_p_st1);

        ratio           = min([1, p * c]);
        
%         fprintf('-----   likelihood   toc:  %d \n', toc);
        
        % ACCEPT OR REJECT?
        u   = rand;
        
        if u < ratio
    %         fprintf('-----     MH       xstar is accepted.    \n');
            x(t, :)         = xStar;
            
            accteptednum    = accteptednum + 1;
        else
            x(t, :)         = x_t_1;
    %         fprintf('-----     MH       xstar is rejected.    \n');
        end
%         fprintf('-----     t:  %d \n', t);

    end

    %fprintf('-----     Accepted rate:  %f \n', accteptednum/nSamples);

    samples = x(burnIn:interval:end, :);
    
    is      = randperm(length(samples));
        
    mu      = reshape(samples(is(1), :), [2 1]);
            
end


%--------------------------------------------------------------------------------
% 
%  log likelihood of mu_k,   
% 
%          p(Z|mu) * p(mu_k | mu_k-1) * p(mu_k+1 | mu_k) 
%
%                   p(Z_k|mu_k) = TT TT mu_k^z * (1 - mu_k)^(1-z)
%         
%---------------------------------------------------------------------------------
function f = log_likelihood_mu(mu_k, mu_k_minus, mu_k_plus, Z_k_1, Z_k_2, N, a, b, rho, k, K_max)

    % likelihood p(Z | mu_k)

    f_l     = 0;
     
    % number of z(i,k) = 1 on feature k in Z_1
    n_1_1   = sum(Z_k_1);
        
    n_1_0   = N(1) - n_1_1;
        
    % number of z(i,k) = 1 on feature k in Z_2
    n_2_1   = sum(Z_k_2);
        
    n_2_0   = N(2) - n_2_1;
        
    %
    f_l     = f_l + n_1_1 * log(mu_k(1)) + n_1_0 * log(1 - mu_k(1))...
                + n_2_1 * log(mu_k(2)) + n_2_0 * log(1 - mu_k(2));
    
    % pdf of FGM copula
    
    if k ~= K_max
    
        f_b     = -1 * log(mu_k(1)) - log(mu_k(2));

        f_b     = f_b + log(1 + rho*(2* (mu_k(1)/mu_k_minus(1))^a - 1)*(2* (mu_k(2)/mu_k_minus(2))^b - 1)  );
        
        f_b     = f_b + log(1 + rho*(2* (mu_k_plus(1)/mu_k(1))^a - 1)*(2* (mu_k_plus(2)/mu_k(2))^b - 1)  );
              
    else
        
        f_b     = log(1 + rho*(2* (mu_k(1)/mu_k_minus(1))^a - 1)*(2* (mu_k(2)/mu_k_minus(2))^b - 1)  );
        
    end
    
    
    %
    f   = f_l +f_b;           
     
     
end

%--------------------------------------------------------------------------------
% 
%  sample a new mu from two independent truncated beta distribution  
% 
%         
%         
%---------------------------------------------------------------------------------
function mu_k = sample_q(mu_k_minus, mu_k_plus, a, b)

    %
%     cdfh        = betacdf(mu_k_minus(1), a, 1);
%     cdfl        = betacdf(mu_k_plus(1), a, 1);
%     
%     u           = unifrnd(cdfl, cdfh);
%     
%     mu_k(1)     = betainv(u, a, 1);
%     
%     %
%     cdfh        = betacdf(mu_k_minus(2), b, 1);
%     cdfl        = betacdf(mu_k_plus(2), b, 1);
%     
%     u           = unifrnd(cdfl, cdfh);
%     
%     mu_k(2)     = betainv(u, b, 1);

    mu_k(1)     = unifrnd(mu_k_plus(1), mu_k_minus(1));
    
    mu_k(2)     = unifrnd(mu_k_plus(2), mu_k_minus(2));

end





