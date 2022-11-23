
clear;

addpath('bibeta');
addpath('copula');
addpath('gp');

iteration = 10;


for i = 1 : 1

N1 = 20;
N2 = 30;

Y = randi([0 1], N1, N2);

% GP-dIBP-NMF
[  L_list, K1_list, K2_list, K_list, Ag_out, Xg_out, L_max ]  = dIBP( iteration, Y );

% BB-dIBP-NMF
a = 1;
b = 1;
alpha1 = 1;
alpha2 = 1;

[ K1_list, K2_list, K_list, Ab_out, Xb_out ] = bIBP_bibeta( iteration, [N1 N2], a, b, Y, alpha1, alpha2);

% C-dIBP-NMF
Yidx  = find(Y > 0);
Ymidx = cell(1, N1);
Ynidx = cell(1, N2);

for m = 1 : N1   
    Ymidx{m} = find(Y(m, :) > 0); 
end

for n = 1 : N2   
    Ynidx{n} = find(Y(:, n) > 0); 
end


[ L_list, K1_list, K2_list, RealK_list, Ac_out, Xc_out, L_max, K_out, alpha_out, rho_out] = bIBP_FGM( iteration, [N1 N2], a, b, Y, Yidx, Ymidx, Ynidx, alpha1, alpha2 );

%
idx = find(Ag_out > 0);
Ag_out(idx) = 1;

idx = find(Xg_out > 0);
Xg_out(idx) = 1;

idx = find(Ab_out > 0);
Ab_out(idx) = 1;

idx = find(Xb_out > 0);
Xb_out(idx) = 1;

idx = find(Ac_out > 0);
Ac_out(idx) = 1;

idx = find(Xc_out > 0);
Xc_out(idx) = 1;

Vg = mean(abs(sum(Ag_out) - sum(Xg_out)));
Vb = mean(abs(sum(Ab_out) - sum(Xb_out)));
Vc = mean(abs(sum(Ac_out) - sum(Xc_out)));



end