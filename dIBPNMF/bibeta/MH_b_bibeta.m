function [para_b] = MH_b_bibeta(a, b, mu, hyperp_bibeta)

    mu = mu ./ [[1; 1], mu(:, 1:end-1)];

    % SOME CONSTANTS
    nSamples    = 500;
    burnIn      = 50;
    interval    = 5;
    
    % INTIIALZE SAMPLER
    x               = zeros(nSamples, 1);
    x(1)            = b;
    t               = 1;
    accteptednum    = 0;    
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
        xStar   = sample_q(hyperp_bibeta);
        
        x_t_1   = x(t-1);

        % CORRECTION FACTOR
        c       = exp(xStar - x_t_1); 
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
           
        log_p_xstar     = log_likelihood(xStar, a, mu);

        log_p_st1       = log_likelihood(x_t_1, a, mu);

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
    
    leng        = length(samples);
    
    rp          = randperm(leng);
    
    para_b      = x(rp(1));
    
end


%--------------------------------------------------------------------------------
% 
%  log likelihood of alpha,   
% 
%         p(mu | alpha) ~ bibetapdf(mu| a, b)
%
%                   
%         
%---------------------------------------------------------------------------------
function f = log_likelihood(para, a, mu)

    b = para;
           
    f = (b-1)* sum( log(mu(2, :) + eps) ) ...
        + b* sum( log(1 - mu(1, :) + eps) ) ...
        - (a+b+1)* sum( log(1 - prod(mu, 1) + eps) );
    
end

%--------------------------------------------------------------------------------
% 
%  sample a new [a, b] from prior  
% 
%                
%         
%---------------------------------------------------------------------------------
function x = sample_q(hyperp_bibeta)

    x = gamrnd(hyperp_bibeta, 1);

end


