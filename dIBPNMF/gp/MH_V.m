function [v] = MH_V(v, alpha, y)

    % SOME CONSTANTS
    nSamples    = 100;
    burnIn      = 30;
    interval    = 5;
    
    % INTIIALZE SAMPLER
    x               = zeros(1, nSamples);
    x(1)            = v;
    t               = 1;
    accteptednum    = 0;    
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
%         tic;
        
        xStar   = sample_q(alpha);
        
        x_t_1   = x(t-1);
        
%         fprintf('-----   proposal  toc:  %d \n', toc);
        
        % CORRECTION FACTOR
        c = 1;
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
%         tic;
        
        log_p_xstar     = log_likelihood( xStar, y);

        log_p_st1       = log_likelihood( x_t_1, y);

        p               = exp( log_p_xstar - log_p_st1);

        ratio           = min([1, p * c]);
        
%         fprintf('-----   likelihood   toc:  %d \n', toc);
        
        % ACCEPT OR REJECT?
        u   = rand;
        
        if u < ratio
    %         fprintf('-----     MH       xstar is accepted.    \n');
            x(t)         = xStar;
            
            accteptednum = accteptednum + 1;
        else
            x(t)         = x_t_1;
    %         fprintf('-----     MH       xstar is rejected.    \n');
        end
%         fprintf('-----     t:  %d \n', t);

    end

    %fprintf('-----     Accepted rate:  %f \n', accteptednum/nSamples);

    samples = x(burnIn:interval:end);
    
    is      = randperm(length(samples));
        
    v       = samples(is(1));
            
end



function f = log_likelihood(x_tmp, y)

    f       = log(inerse_gamma_pdf(y, 2, x_tmp + eps) );           
          
end


function a = sample_q(para)

    a = gamrnd(1, 1/para);
        
end

function ig = inerse_gamma_pdf(x, alpha, beta)

    ig = (beta .^ alpha ./gamma(alpha)) .* (x.^ (-1 * alpha - 1)) .* exp(-beta ./ x);

end



