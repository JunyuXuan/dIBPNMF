function idx = SpectralClustering( D , class_number, FILEPATH)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% class_number = 2;

% normalization

% D = D - diag(diag(D));
%D = normr(D);

% lapalance matrix 
%d = diag(sum(D, 2));
%D = D + 10*eps*speye(size(D));
%L = d-D;



%clustering
idx = kmeans(D, class_number);

%output
% fid=fopen(FILEPATH,'w');
% fprintf(fid,'%u\n',idx);
% fclose(fid);

end

