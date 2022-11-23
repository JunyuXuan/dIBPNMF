function [v] = MH_V2_FGM(v, alpha, k, Yn, Z2n, V2n, A, epsilon)

    V2n(k) = 0;

    TMP = A * transpose(Z2n .* V2n);
    
    Ak  = A(:, k);

    % SOME CONSTANTS
    nSamples    = 10;
    burnIn      = 1;
    interval    = 1;
    
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
        
        xStar   = gamrnd(1, 1/alpha);
        
        x_t_1   = x(t-1);
        
%         fprintf('-----   proposal  toc:  %d \n', toc);
        
        % CORRECTION FACTOR
        c = 1;
        
        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO
%         tic;
        
        log_p_xstar     = log_likelihood( xStar, TMP, Yn, Ak, epsilon);

        log_p_st1       = log_likelihood( x_t_1, TMP, Yn, Ak, epsilon);

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



function f = log_likelihood(x_tmp, TMP, Yn, Ak, epsilon)

    TMP2 = TMP + x_tmp * Ak;

%     f  = sum(log(exppdf(Yn, A * transpose(Z2n .* V2n) + epsilon) + eps  ) );     
          
    f = sum( -log(TMP2 + epsilon) - Yn ./ (TMP2+ epsilon) );
end




