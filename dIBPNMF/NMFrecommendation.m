

Y       = Y1b;

Yt      = Y1t;

idxt    = find(Yt > 0);

[N1, N2] = size(Y);

W       = zeros(N1, N2);

idx     = find(Y > 0);

W(idx)  = 1;

max_N   = 100;

MAENMF  = zeros(1, max_N);

for k = 1 : max_N
   
    [A, X, C]      = nmf_weighted_mia(Y, W, k, 1000);
    
    rY             = A * X;
    
    MAENMF(k)      = sum(abs(rY(idx) - Yt(idx)));
    
end


save 'nmf_results_m100k_Y1b.mat' MAENMF;




