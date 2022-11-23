function [rho] = MH_rho_FGM(rho_old, mu, alpha)

    nu = mu ./ [[1; 1], mu(:, 1:end-1)];

    % SOME CONSTANTS
    nSamples    = 500;
    burnIn      = 100;
    interval    = 20;
    
    % INTIIALZE SAMPLER
    x               = zeros(1,nSamples);
    x(1)            = rho_old;
    t               = 1;
    accteptednum    = 0;    
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
        xStar   = sample_q();
        
        x_t_1   = x(t-1);

        % CORRECTION FACTOR
        c       = 1; 
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
           
        log_p_xstar     = log_likelihood(xStar, nu, alpha);

        log_p_st1       = log_likelihood(x_t_1, nu, alpha);

        p               = exp( log_p_xstar - log_p_st1);

        ratio           = min([1, p * c]);

        % ACCEPT OR REJECT?
        u   = rand;
        
        if u < ratio
    %         fprintf('-----     MH       xstar is accepted.    \n');
            x(t)            = xStar;
            
            accteptednum    = accteptednum + 1;
        else
            x(t)            = x_t_1;
    %         fprintf('-----     MH       xstar is rejected.    \n');
        end
        %fprintf('-----     t:  %d \n', t);

    end

    fprintf('-----     Accepted rate:  %f \n', accteptednum/nSamples);

    samples     = x(burnIn:interval:end);
    
    is          = randperm(length(samples));
    
    rho         = samples(is(1));
    
end


%--------------------------------------------------------------------------------
% 
%  log likelihood of rho  
% 
%         copula-pdf(mu) = 1 + rho*(2*BETACDF(mu1) - 1)*(2*BETACDF(mu2) - 1) 
%         
%---------------------------------------------------------------------------------
function f = log_likelihood(rho, mu, alpha)
        
    f       = sum( log(1 + rho * ((2*(mu(1, :)) .^(alpha(1)) - 1) .* (2*(mu(2, :)) .^(alpha(2)) - 1) + eps)) );
                   
end

%--------------------------------------------------------------------------------
% 
%  sample a new rho from uniform distribution [-1, 1]                  
%         
%---------------------------------------------------------------------------------
function x = sample_q()

    x = unifrnd(-1, 1);

end


