
Y = Y4b;

Yt = Y5t;

A = bb_A_out;

X = bb_X_out;

rY = A * X';

idx = find(Yt > 0);

bbMAE5 = sum(abs(rY(idx) - Yt(idx)))


A = cp_A_out;

X = cp_X_out;

rY = A * X';

idx = find(Yt > 0);

cpMAE5 = sum(abs(rY(idx) - Yt(idx)))


A = gp_A_out;

X = gp_X_out;

rY = A * X';

idx = find(Yt > 0);

gpMAE5 = sum(abs(rY(idx) - Yt(idx)))

MAEY5b = [bbMAE5 cpMAE5 gpMAE5];


