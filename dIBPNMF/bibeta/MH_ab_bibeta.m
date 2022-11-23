function [para_a, para_b] = MH_ab_bibeta(a, b, mu, hyperp_bibeta)

    nu = mu ./ [[1; 1], mu(:, 1:end-1)];
    

    % SOME CONSTANTS
    nSamples    = 500;
    burnIn      = 50;
    interval    = 5;
    
    % INTIIALZE SAMPLER
    x               = zeros(nSamples, 2);
    x(1, :)         = [a, b];
    t               = 1;
    accteptednum    = 0;    
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
        xStar   = sample_q(hyperp_bibeta);
        
        x_t_1   = x(t-1, :);

        % CORRECTION FACTOR
        c       = exp(xStar - x_t_1); 
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
           
        log_p_xstar     = log_likelihood(xStar, nu);

        log_p_st1       = log_likelihood(x_t_1, nu);

        p               = exp( log_p_xstar - log_p_st1);

        ratio           = min([1, p * c]);

        % ACCEPT OR REJECT?
        u               = rand;
        
        if u < ratio
    %         fprintf('-----     MH       xstar is accepted.    \n');
            x(t, :)            = xStar;
            
            accteptednum       = accteptednum + 1;
        else
            x(t, :)            = x_t_1;
    %         fprintf('-----     MH       xstar is rejected.    \n');
        end
        %fprintf('-----     t:  %d \n', t);

    end

    fprintf('-----     Accepted rate:  %f \n', accteptednum/nSamples);

    samples     = x(burnIn:interval:end, :);
    
    para        = reshape(mean(samples, 1), [1 2]);
    
    para_a       = para(1);
    
    para_b       = para(2);
    
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
function f = log_likelihood(para, mu)

    a = para(1);
    b = para(2);
    
       
    f = (a-1)* sum( log(mu(1, :) + eps) ) + (b-1)* sum( log(mu(2, :) + eps) ) ...
        + b* sum( log(1 - mu(1, :) + eps) ) + a* sum( log(1 - mu(2, :) + eps) ) ...
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

    x = gamrnd(hyperp_bibeta, 1, [1 2]);

end


