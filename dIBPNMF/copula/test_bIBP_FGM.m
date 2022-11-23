
clear;

tic;

a = 1;
b = 1;
N1 = 20;
N2 = 30;


%Y = rand(N1, N2);

Y = randi([0 1], N1, N2);


iteration = 1000;

alpha1 = 1;%/mean(mean(Y));
alpha2 = alpha1;

%[ K1_list, K2_list, K_list, A_out, X_out ] = bIBP_bibeta( iteration, [N1 N2], a, b, Y, alpha1, alpha2);

[ K1_list, K2_list, K_list, A_out, X_out ] = bIBP_FGM( iteration, [N1 N2], a, b, Y, alpha1, alpha2);


% x  = tabulate(K_list(1:end));
x1 = tabulate(K1_list(1:end));
x2 = tabulate(K2_list(1:end));
x3 = tabulate(K_list(1:end));


h=figure;

%
subplot(1, 3, 1);

plot(x1(:, 1), x1(:, 2)/sum(x1(:,2)), '+r');
          
hold on;

plot(x1(:, 1), poisspdf(x1(:, 1), a*harmonic(N1)), 'og');


%
subplot(1, 3, 2);

plot(x2(:, 1), x2(:, 2)/sum(x2(:,2)), '+r');
          
hold on;

plot(x2(:, 1), poisspdf(x2(:, 1), b*harmonic(N2)), 'og');

%
subplot(1, 3, 3);

plot(x3(:, 1), x3(:, 2)/sum(x3(:,2)), '+r');
          
[A, X] = nmf_mia(Y, 18);


error_fgm  = sum(sum(Y - A_out*X_out'))

error_nmf  = sum(sum(Y - A*X))


% %savefig(h, 'bIBP_K.fig');
% saveas(h,'bIBP_K_alpha1_1_alpha2_1','fig') ;
% 
% fprintf('-------             all time = %d \n ', toc);


