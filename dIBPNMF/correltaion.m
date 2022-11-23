

mux = [0.5 0.4 0.3 0.2 0.1];


%muy = zeros(5, 5);


while 1
    
   y_tmp = betarnd(1, 1, [1 5]);
    
   if corr(mux', y_tmp') <= 1 & corr(mux', y_tmp') > 0.99
      
       muy(9, :) = y_tmp;
       
       break;
       
   end
       
end


