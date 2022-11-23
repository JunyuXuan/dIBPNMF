
%idx = load('/home/jxuan/workspace_matlab/citeseer/docid_classid_m.txt');

%% nmf
  
%max_N = max(N1, N2);

max_N = 20;

errorall = zeros(100, 1, 20);

sparseall = zeros(100, 2, 20);


for i = 1 : 100

    Y = rand(20, 30);

    error  = zeros(1, max_N);

    sparse = zeros(2, max_N);

    % CR    = zeros(3,max_N);

    for k = 1 : max_N

        [A, X]      = nmf_mia(Y, k, 5000);

        error(k) 	= (sum(sum(abs(Y - A*X))));

        sparse(1, k) = length(find(A < 0.0000000001));

        sparse(2, k) = length(find(X < 0.0000000001));

    %     nA          = normr(A);
    %     
    %     nidx        = kmeans(nA, 6);
    %     
    %     [nJC, nFM, nF1] = evaluat_clusterresults( idx, nidx);
    %     
    %     CR(1, k) = nJC;
    %     CR(2, k) = nFM;
    %     CR(3, k) = nF1;

    end

    errorall(i, :, :) = error;
    
    sparseall(i, :, :) = sparse;
    
end

save 'nmf_results_rand.mat' sparseall;

xx = mean(errorall, 1);
yy = mean(sparseall, 1);

