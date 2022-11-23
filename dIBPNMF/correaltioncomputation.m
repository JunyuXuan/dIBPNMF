clear;

format long

a = 2.5;
b = 4;
c = 1;

hy = hypergeom( [a+1,  b+1, a+b+1], [a+b+2, a+b+2], 1) %genHyper ,0,3,8

h =  a * b * gamma(a+1)*gamma(b+1)/ (gamma(a+b+2)* (a+b+1)  ) 

EXY = hy * h

VXVY = ( a/((a+1)^2 *(a+2)) ) * (  b/((b+1)^2*(b+2)) )

EX = a/(a+1)
EY = b/(b+1)

EXEY = EX * EY 

coco = (EXY - EXEY ) / VXVY^0.5

