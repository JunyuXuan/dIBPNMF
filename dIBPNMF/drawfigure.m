

% % MAE = zeros(5, 100);
% % 
% % MAE(1, :) = MAENMF1;
% % MAE(2, :) = MAENMF2;
% % MAE(3, :) = MAENMF3;
% % MAE(4, :) = MAENMF4;
% % MAE(5, :) = MAENMF5;
% % 
% % 
% % meanmae = mean(MAE, 1);
% % 
% % vmae = std(MAE, 1);
% 
% MAEIBP = zeros(5, 3);
% 
% MAEIBP(1,:) = MAEY1b;
% MAEIBP(2,:) = MAEY2b;
% MAEIBP(3,:) = MAEY3b;
% MAEIBP(4,:) = MAEY4b;
% MAEIBP(5,:) = MAEY5b;
% 

figure;

meanmae = rand(1,20);

vmae = 1;

x = [1:20, fliplr(1:20)];

y = [meanmae-vmae, fliplr(meanmae + vmae)];

fill(x,y,[.90,.90,.90],'EdgeColor','None');

hold on;

plot(1:20, meanmae, 'black', 'LineWidth',2);




% % y1 = [ones(1,100) * meanibp(1)-vibp(1), fliplr(ones(1,100) * meanibp(1) + vibp(1))];
% % 
% % fill(x,y1,[.65,.65,.65],'EdgeColor','None');
% % 
% % plot(ones(1,100) * meanibp(1), '-.k', 'LineWidth', 2);



% y2 = [ones(1,100) * meanibp(2)-vibp(2), fliplr(ones(1,100) * meanibp(2) + vibp(2))];
% 
% fill(x,y2,[.65,.65,.65],'EdgeColor','None');
% 
% plot(ones(1,100) * meanibp(2), '-.k', 'LineWidth', 2);




% y3 = [ones(1,100) * meanibp(3)-vibp(3), fliplr(ones(1,100) * meanibp(3) + vibp(3))];
% 
% fill(x,y3,[.65,.65,.65],'EdgeColor','None');
% 
% plot(ones(1,100) * meanibp(3), '-.k', 'LineWidth', 2);






