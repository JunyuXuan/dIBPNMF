function [ sample ] = MH_mu(k, b_m, b_p, b_K, sigma1, sigma2, rho, g1, g2, alpha, z1, z2, T, N )
%  sample b through Metroplis-Hastings

    % DEFINE PROPOSAL DENSITY

    % SOME CONSTANTS
    nSamples        = 100;
    burnIn          = 30;
    Interval        = 5;

    % INTIIALZE SAMPLER
    x               = zeros(1, nSamples);
    x(1)            = sample_q(b_m, b_p);
    t               = 1;
    accteptednum    = 0;

    % RUN METROPOLIS-HASTINGS SAMPLER
    while t < nSamples

        t           = t+1;

        % SAMPLE FROM PROPOSAL
        xStar       = sample_q(b_m, b_p);
        x_t_1       = x(t-1);

        % CORRECTION FACTOR
        c           = 1;

        % CALCULATE THE (CORRECTED) ACCEPTANCE RATIO

        log_p_xstar = log_func(xStar,  k, b_K, alpha, T, z1, z2, sigma1, sigma2, rho, g1, g2, N);

        log_p_st1   = log_func(x_t_1,  k, b_K, alpha, T, z1, z2, sigma1, sigma2, rho, g1, g2, N);

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
        %fprintf('-----     t:  %d \n', t);

    end

    %fprintf('-----     Accepted rate:  %f \n', accteptednum/nSamples);

    samples = x(burnIn:Interval:end);

    is      = randperm(length(samples));
        
    sample  = samples(is(1));
    
end

function f = log_func(v, k, b_K, alpha, T, z1, z2, sigma1, sigma2, rho, g1, g2, N)

    
    gamma_kt1 = normcdf(norminv(v,0,sigma1+rho) - g1, 0, rho);

    gamma_kt2 = normcdf(norminv(v,0,sigma2+rho) - g2, 0, rho);


    f   = log(b_K^alpha ./ v);
    
    f   = f + z1 * log( eps + gamma_kt1 ) + (N(1) - z1)  * log( eps + 1 - gamma_kt1 )...
            + z2 * log( eps + gamma_kt2 ) + (N(2) - z2)  * log( eps + 1 - gamma_kt2 );
    
    
end

function f = sample_q(b_m, b_p)

    f = unifrnd(b_p, b_m);
  
end

%-----------------------------------------------------------------
% 
%  compute the gamma_kt 
% 
%-----------------------------------------------------------------
function gamma_kt = gammakt(v, sigma_tt, rho, g_kt)

    gamma_kt = normcdf(norminv(v,0,sigma_tt+rho) - g_kt, 0, rho);

end


