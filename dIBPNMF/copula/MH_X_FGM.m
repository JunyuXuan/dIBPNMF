function [xnk] = MH_X_FGM(xnk, k, xx, vz, y, aa)

    % SOME CONSTANTS
    nSamples    = 5;
    burnIn      = 3;
    interval    = 3;
    
    % INTIIALZE SAMPLER
    x               = zeros(1, nSamples);
    x(1)            = xnk;
    t               = 1;
    accteptednum    = 0;    
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
%         tic;
        
        xStar   = sample_q(2, vz);
        
        x_t_1   = x(t-1);
        
%         fprintf('-----   proposal  toc:  %d \n', toc);
        
        % CORRECTION FACTOR
        c = 1;
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
%         tic;
        
        log_p_xstar     = log_likelihood( xStar, k, y, aa, xx);

        log_p_st1       = log_likelihood( x_t_1, k, y, aa, xx);

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
        
    xnk     = samples(is(1));
            
end



function f = log_likelihood(x_tmp, k, y, aa, xx)

    xx(k)   = x_tmp;
         
    lamda   = aa * xx';
    
    % weibull distribution 
    f       = sum( -1 * log(lamda + eps) -  y ./ (lamda + eps));
    
    % exponential distribution
%     f       = sum( log(lamda + eps) -  y .* lamda );

end


function a = sample_q(para1, para2)

    a = gamrnd(para1, 1/para2);
    
    a = 1/a;
    
end





