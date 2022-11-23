function [alpha] = MH_alpha_FGM(k, a, para_alpha, mu, rho)

    nu = mu ./ [[1; 1], mu(:, 1:end-1)];
    

    % SOME CONSTANTS
    nSamples    = 800;
    burnIn      = 100;
    interval    = 30;
    
    % INTIIALZE SAMPLER
    x               = zeros(1, nSamples);
    x(1)            = a(k);
    t               = 1;
    accteptednum    = 0;    
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
        xStar   = sample_q(para_alpha);
        
        x_t_1   = x(t-1);

        % CORRECTION FACTOR
        c       = 1; 
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
           
        log_p_xstar     = log_likelihood(xStar, nu, rho, a, k);

        log_p_st1       = log_likelihood(x_t_1, nu, rho, a, k);

        p               = exp( log_p_xstar - log_p_st1);

        ratio           = min([1, p * c]);

        % ACCEPT OR REJECT?
        u               = rand;
        
        if u < ratio
            x(t)            = xStar;
            
            accteptednum    = accteptednum + 1;
        else
            x(t)            = x_t_1;
        end
        
    end

    fprintf('-----     Accepted rate:  %f \n', accteptednum/nSamples);

    samples     = x(burnIn:interval:end);
    
    is          = randperm(length(samples));
    
    alpha       = samples(is(1));
    
end


%--------------------------------------------------------------------------------
% 
%  log likelihood of alpha,   
% 
%         p(mu | alpha) ~ copula-pdf(mu, alpha, 1)
%
%                   
%         
%---------------------------------------------------------------------------------
function f = log_likelihood(alpha, mu, rho, a, k)
                   
    % copula pdf p(u, v)
    if k == 1 
        f       = sum( log(1 + rho * ((2*(mu(1, :)) .^alpha - 1) .* (2*(mu(2, :)) .^(a(2)) - 1) + eps)) );
    else
        f       = sum( log(1 + rho * ((2*(mu(1, :)) .^a(1) - 1) .* (2*(mu(2, :)) .^alpha - 1) + eps)) );
    end
    
end

%--------------------------------------------------------------------------------
% 
%  sample a new alpha from prior  
% 
%         
%
%                  
%         
%---------------------------------------------------------------------------------
function x = sample_q(para_alpha)

    x = gamrnd(para_alpha, 1);

end


