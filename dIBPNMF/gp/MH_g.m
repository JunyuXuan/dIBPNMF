function [ sample ] = MH_g(k, g, T, N, h, rho, Sigma_k)
%  sample b through Metroplis-Hastings

    % DEFINE PROPOSAL DENSITY

    % SOME CONSTANTS
    nSamples        = 100;
    burnIn          = 20;

    % INTIIALZE SAMPLER
    x               = zeros(nSamples, 2);
    x(1, :)         = g(:, k)';

    t               = 1;
    accteptednum    = 0;

    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t           = t+1;

        % SAMPLE FROM PROPOSAL
        xStar       = sample_q(k, Sigma_k, T);
        x_t_1       = x(t-1, :);

        % CORRECTION FACTOR
        c           = 1;

        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO

        log_p_xstar = log_func(xStar, k, N, h, rho);

        log_p_st1   = log_func(x_t_1, k, N, h, rho);

        p           = exp( log_p_xstar - log_p_st1);

        ratio       = min([1, p * c]);

        % ACCEPT OR REJECT?
        u           = rand;

        if u < ratio
            %         fprintf('-----     MH       xstar is accepted.    \n');
            x(t, :)         = xStar;
            accteptednum    = accteptednum + 1;
        else
            x(t, :)         = x(t-1, :);
            %         fprintf('-----     MH       xstar is rejected.    \n');
        end
        %fprintf('-----     t:  %d \n', t);

    end

    %fprintf('-----     Accepted rate:  %f \n', accteptednum/nSamples);

    samples     = x(burnIn:end, :);

    is          = randperm(size(samples, 1));
        
    sample      = samples(is(1), :);
    
   % sample      = mean(samples, 1)';

end

function f = log_func(x, k, N, h, rho)

    f     = 0;
    
    h1    = h{1};
    h2    = h{2};
    
    f = f + sum(log(eps + normpdf(h1(:, k), x(1), rho)));
    
    f = f + sum(log(eps + normpdf(h2(:, k), x(2), rho)));
    
end

function f = sample_q(k, Sigma_k, T)

    %Sigma_k = reshape(Sigma(k, :, :), [T T]);
    
    f       = mvnrnd([0, 0],Sigma_k);
  
end


