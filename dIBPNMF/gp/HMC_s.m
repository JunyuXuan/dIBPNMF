function v = HMC_s(s_k, T, sigma, a, g_k)

    % STEP SIZE
    delta       = 0.3;
    nSamples    = 50;
    burnin      = 0.2 * nSamples;
    L           = 10;
    keps        = 1e-2;

    % DEFINE KINETIC ENERGY FUNCTION
    K           = inline('p^2/2','p');

    % INITIAL STATE
    x           = zeros(1, nSamples);
    x0          = s_k ^ 0.5;
    x(1)        = x0;

    t = 1;
    while t < nSamples
        t       = t + 1;
        
        % set position
        x0      = x(t-1);
        
        % SAMPLE RANDOM MOMENTUM
        p0      = randn();

        %% SIMULATE HAMILTONIAN DYNAMICS
        % FIRST 1/2 STEP OF MOMENTUM
        pStar   = p0 - delta/2*dU(x0, T, sigma, a, g_k);

        % FIRST FULL STEP FOR POSITION/SAMPLE
        xStar   = x0 + delta*pStar;

        % FULL STEPS
        for jL = 1:L-1
            
            if isnan(xStar)
                xStar = eps;
            end
                        
            % MOMENTUM
            pStar = pStar - delta*dU(xStar, T, sigma, a, g_k);
            
            % POSITION/SAMPLE
            xStar = xStar + delta*pStar;
        end

        % LAST HALP STEP
        pStar = pStar - delta/2*dU(xStar, T, sigma, a, g_k);

        if isnan(xStar)
            xStar = eps;
        end
        
        % EVALUATE ENERGIES AT
        % START AND END OF TRAJECTORY
        U0      = U(x0, T, sigma, a, g_k);
        UStar   = U(xStar, T, sigma, a, g_k);

        K0      = K(p0);
        KStar   = K(pStar);

        % ACCEPTANCE/REJECTION CRITERION
        alpha   = min(1,exp((U0 + K0) - (UStar + KStar)));

        u = rand;
        if u < alpha
            x(t) = xStar;
        else
            x(t) = x0;
        end

    end

    samples = x(burnin:end) ;
    
    is      = randperm(length(samples));
        
    v       = samples(is(1)).^2;
    
end

%----------------------------------------------------------------------------
% 
%  POTENTIAL ENERGY FUNCTION:
%
%               U(x) = -log(p(x)), p(x) = gamma(x;a)*normal(g_k; 0, Sigma_k)
%
%----------------------------------------------------------------------------
function f = U(v, T, sigma, a, g_k)
   
    Sigma_k     = sigma * exp(-1 * (( ( repmat(g_k', [T 1]) - repmat(g_k, [1 T]) ) .^ 2 )./(v^2)) ) ;%+ 10*ones(T);
        
    inv_Sigma_k = inv(Sigma_k+ 0.0001*eye(T));
 
    f           = 0.5*log(det(Sigma_k)+eps) + 0.5* g_k' * inv_Sigma_k * g_k  ...
                    - (a-1)*log(v) + v;
end

%---------------------------------------------------------------
% 
%  GRADIENT OF POTENTIAL ENERGY:
%
%               dU(x) = d(-log(p(x)))/dx
%---------------------------------------------------------------
function f = dU(v, T, sigma, a, g_k)
    
    
    %Sigma_k     = Update_Sigma(T, g_k, sigma, v);
    
    Sigma_k     = sigma * exp(-1 * (( ( repmat(g_k', [T 1]) - repmat(g_k, [1 T]) ) .^ 2 )./ (v^2)) ) ;%+ 10 * ones(T);
    
    dSigma_k    = sigma * exp(-1 * (( ( repmat(g_k', [T 1]) - repmat(g_k, [1 T]) ) .^ 2 )./ (v^2)) ) ...
                    .* (2 * (( ( repmat(g_k', [T 1]) - repmat(g_k, [1 T]) ) .^ 2 )./(v^3)) );
        
    inv_Sigma_k = inv(Sigma_k+ 0.0001*eye(T));

    f           = 0.5 * trace( inv_Sigma_k * dSigma_k)...
                    -0.5 * g_k' * inv_Sigma_k * dSigma_k * inv_Sigma_k * g_k...
                        - (a-1)/v + 1;
            
end

%---------------------------------------------------------------
% 
%  renew Sigma by new s
%
%---------------------------------------------------------------
function [Sigma_k]  = Update_Sigma(T, g_k, sigma, s)

    Sigma_k         = zeros(T, T);
    
    Sigma_k(:, :)   = sigma * exp(-1 * (( ( repmat(g_k', [1 T]) - repmat(g_k, [T 1]) ) .^ 2 )./(s^2)) );
    
end

%---------------------------------------------------------------
% 
%  gradient of Sigma_k by s
%
%---------------------------------------------------------------
function [dSigma_k]  = dSigma(T, sigma, s)

    dSigma_k        = zeros(T, T);
    
    tt              = 1:T;
        
    dSigma_k(:, :)  = sigma * ( exp(-(( repmat(tt', [1 T]) - repmat(tt, [T 1]) ) .^ 2 )./(s^2)) ...
                        ...%.*  (1 * (( repmat(tt', [1 T]) - repmat(tt, [T 1]) ) .^ 2 )./ (s^(2))  )) ;
                        .*  (2 * (( repmat(tt', [1 T]) - repmat(tt, [T 1]) ) .^ 2 )./ (s^(3))  )) ;
    
    dSigma_k(:, :)   = sigma * exp(-1 * (( ( repmat(g_k', [1 T]) - repmat(g_k, [T 1]) ) .^ 2 )./(s^2)) );
    
end

