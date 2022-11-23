

clear;

load '/home/jxuan/workspace_matlab/bIBP/codes/syn_data.mat'

% N1 = 20;
% N2 = 30;
% 
% Y = randi([0 1], N1, N2);


iteration = 1000;


[  L_list, K1_list, K2_list, K_list, A_out, X_out, L_max ]  = dIBP( iteration, Y );


error_bibp = sum(sum(abs(Y - A_out*X_out')))

%save 'gp_results_syn.mat' L_list K1_list K2_list K_list A_out X_out L_max error_bibp

