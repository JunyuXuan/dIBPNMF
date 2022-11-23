function [ alpha ] = MH_alpha( alpha, alpha_para, K, b_K, b)

    % DEFINE FUNCTION
    log_func        = inline('K*log(x+eps) +x *log(b_K + eps) + sum(-1 * log(b + eps)) ','x','alpha_para','K','b_K','b');
    
    % SOME CONSTANTS
    nSamples        = 100;
    burnIn          = 10;

    % INTIIALZE SAMPLER
    x               = zeros(1, nSamples);
    x(1)            = alpha;
    t               = 1;
    accteptednum    = 0;
    
    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t       = t+1;

        % SAMPLE FROM PROPOSAL
        xStar   = gamrnd(alpha_para, 1);
        x_t_1   = x(t-1);

        % CORRECTION FACTOR
        c       = 1;

        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO

        log_p_xstar = log_func(xStar, alpha_para, K, b_K, b);

        log_p_st1   = log_func(x_t_1, alpha_para, K, b_K, b);

        p           = exp( log_p_xstar - log_p_st1);

        ratio       = min([1, p * c]);

        % ACCEPT OR REJECT?
        u           = rand;
        
        if u < ratio
    %         fprintf('-----     MH       xstar is accepted.    \n');
            x(t)            = xStar;
            accteptednum    = accteptednum + 1;
        else
            x(t)            = x(t-1);
    %         fprintf('-----     MH       xstar is rejected.    \n');
        end


    end

    
    samples = x(burnIn:end);
    
    is      = randperm(length(samples));
        
    alpha   = samples(is(1));
    
    fprintf('-----     Accepted rate:  %f, alpha : %f \n', accteptednum/nSamples, alpha);
    
end

