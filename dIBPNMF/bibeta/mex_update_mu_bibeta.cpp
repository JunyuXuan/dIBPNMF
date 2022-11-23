
// [mu]  = Update_mu(mu, g, Sigma, rho, K_max, alpha, Z, T, N)


#include "mex.h" 

//#include "stdafx.h" 
#include <numeric> 
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

using namespace std;

#include "ranlib.hpp"
#include "rnglib.hpp"

// log_likelihood_mu(xStar, mu_k_minus, mu_k_plus, Z1sum, Z2sum, N1, N2, A, B, k, K)

double log_likelihood_mu(double* xStar, double* mu_k_minus, double* mu_k_plus, int Z1sum, int Z2sum,int N1, int N2, double A, double B, int k, int K)
{
    
    int n11 = Z1sum;    
    int n10 = N1 - n11;
    
    int n21 = Z2sum;    
    int n20 = N2 - n21;
    
    double fl = n11*log(*(xStar) + 0.0000000001) + n10 * log(1 - *(xStar)+ 0.0000000001) 
                    + n21*log(*(xStar+1) + 0.0000000001) + n20*log(1 - *(xStar+1) + 0.0000000001) ;
    
    double xstart_minus1 = exp(log(*(xStar)  + 0.0000000001) - log(*(mu_k_minus) + 0.0000000001));
    double xstart_minus2 = exp(log(*(xStar+1)  + 0.0000000001) - log(*(mu_k_minus +1) + 0.0000000001));
    
    double plus_xstart1 = exp(log(*(mu_k_plus)  + 0.0000000001) - log(*(xStar) + 0.0000000001));
    double plus_xstart2 = exp(log(*(mu_k_plus+1)  + 0.0000000001) - log(*(xStar +1) + 0.0000000001));
    
    if(xstart_minus1 > 1){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        xstart_minus1 = 1;
    }
    
    if(xstart_minus1 < 0){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        xstart_minus1 = 0;
    }
    
    if(xstart_minus2 > 1){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        xstart_minus2 = 1;
    }
    
    if(xstart_minus2 < 0){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        xstart_minus2 = 0;
    }
    
    if(plus_xstart1 > 1){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        plus_xstart1 = 1;
    }
    
    if(plus_xstart1 < 0){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        plus_xstart1 = 0;
    }
    
    if(plus_xstart2 > 1){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        plus_xstart2 = 1;
    }
    
    if(plus_xstart2 < 0){
        //mexPrintf(" k = %d, xstart_minus1= %f \n", k, xstart_minus1);
        plus_xstart2 = 0;
    }
    
    
    if(k == (K-1)){
        
        fl = fl + (A-1)*log(xstart_minus1 + 0.0000000001) + (B-1)*log(xstart_minus2 + 0.0000000001) 
                + B * log(1 - xstart_minus1 + 0.0000000001) + A * log(1 - xstart_minus2 + 0.0000000001)
                - (A+B+1)*log(1 - xstart_minus1 * xstart_minus2 + 0.0000000001 ) ;
        
    }else{
        
        fl = fl - log(*(xStar) + 0.0000000001) - log(*(xStar+1) + 0.0000000001);
        
        fl = fl + (A-1)*log(xstart_minus1 + 0.0000000001) + (B-1)*log(xstart_minus2 + 0.0000000001) 
                + B * log(1 - xstart_minus1 + 0.0000000001) + A * log(1 - xstart_minus2 + 0.0000000001)
                - (A+B+1)*log(1 - xstart_minus1 * xstart_minus2 + 0.0000000001 ) ;
        
        fl = fl + (A-1)*log(plus_xstart1 + 0.0000000001) + (B-1)*log(plus_xstart2 + 0.0000000001) 
                + B * log(1 - plus_xstart1 + 0.0000000001) + A * log(1 - plus_xstart2 + 0.0000000001)
                - (A+B+1)*log(1 - plus_xstart1  * plus_xstart2 + 0.0000000001) ;
        
    }
    
    return fl;
    
    
}




// mu_k  = MH_mu(k, K, mu_k_minus, mu_k, mu_k_plus, (int) *(Z1sum + k), (int) *(Z2sum + k), N1, N2, A, B);

void MH_mu(double* returned,int k, int K, double* mu_k_minus, double* mu_k, double* mu_k_plus, int Z1sum, int Z2sum, int N1, int N2, double A, double B)
{
//     if (k <= 10)
//     mexPrintf("in MH_mu k = %d ....... mu1(k-1) = %f, mu1(k+1) = %f, mu2(k-1) = %f, mu2(k+1) = %f. \n", k, *(mu_k_minus), *(mu_k_plus), *(mu_k_minus+1), *(mu_k_plus+1));
    
    srand((unsigned)time(0) * k);
    
    int nSamples        = 20;
    int burnIn          = 30;
    int Interval        = 5;
    
	double samples[2][nSamples];
   
    double x_tmp[2];
    
    x_tmp[0] = *(mu_k);
    x_tmp[1] = *(mu_k + 1);
    
    double xStar[2];
    double x_t_1[2];
    //double* returned = (double*) mxCalloc( 2 , sizeof( double ));
    
	int t   = 1;
	int num = 0;

	double log_p_xstar, log_p_st1, p, ratio, u;

	while(num < nSamples){
    
        u = rand() /double(RAND_MAX);
        
        while(u > 1 || u < 0){
            u = rand() /double(RAND_MAX);
        }
        
        xStar[0]       = u * ( *(mu_k_minus) - *(mu_k_plus)) + *(mu_k_plus);    
        
        u = rand() /double(RAND_MAX);
        
        while(u > 1 || u < 0){
            u = rand() /double(RAND_MAX);
        }
        
		xStar[1]       = u * ( *(mu_k_minus + 1) - *(mu_k_plus + 1)) + *(mu_k_plus + 1);
        
        if ( (xStar[0] < *(mu_k_plus) || xStar[0] > *(mu_k_minus)) && xStar[0] <= 0.000001){
            //mexPrintf("prblem 1 ! k = %d ....... mu(k-1) = %f, new mu(k) = %f, mu(k+1) = %f. \n", k, *(mu_k_minus), xStar[0], *(mu_k_plus));
            xStar[0] = *(mu_k_plus);
        }
    
        if (xStar[1] < *(mu_k_plus+1) || xStar[1] > *(mu_k_minus+1) && xStar[1] <= 0.000001){
            //mexPrintf("prblem 2 ! k = %d ....... mu(k-1) = %f, new mu(k) = %f, mu(k+1) = %f. \n", k, *(mu_k_minus+1), xStar[1], *(mu_k_plus+1));
            xStar[1] = *(mu_k_plus+1);
        }
        
        
		x_t_1[0]       = x_tmp[0];
        x_t_1[1]       = x_tmp[1];
        
        // log_likelihood_mu(xStar, mu_k_minus, mu_k_plus, Z1sum, Z2sum, N1, N2, A, B, k, K)
           
		log_p_xstar = log_likelihood_mu(xStar, mu_k_minus, mu_k_plus, Z1sum, Z2sum, N1, N2, A, B, k, K);
          
		log_p_st1   = log_likelihood_mu(x_t_1, mu_k_minus, mu_k_plus, Z1sum, Z2sum, N1, N2, A, B, k, K);
    
		p           = exp( log_p_xstar - log_p_st1);
        
        if (isnan(log_p_xstar) || isnan(log_p_st1))
            mexPrintf(" k = %d, xStar[0]= %f, xStar[1]= %f, log_p_xstar = %f, x_t_1[0] = %f, x_t_1[1] = %f, log_p_st1 = %f, p = %f . \n", k, xStar[0], xStar[1], log_p_xstar, x_t_1[0], x_t_1[1], log_p_st1, p);
    
        if (p <= 1){
            ratio = p;
        }else{                 
            ratio = 1;
        }
		   
		u           = rand() /double(RAND_MAX);
    
		if (u < ratio){
			x_tmp[0]           = xStar[0];	
            x_tmp[1]           = xStar[1];	
		}else{
			x_tmp[0]           = x_t_1[0];
            x_tmp[1]           = x_t_1[1];
        }	
            
		if (t > burnIn){

			if( t % Interval == 0 ){

				samples[0][num] = x_tmp[0];
                samples[1][num] = x_tmp[1];
				num++;
			}

		}
        
        t++;
   
	}
	
    int selected 	= (rand() % (nSamples-1 -0+1))+ 0;
    
    *(returned)     = samples[0][selected];
    *(returned + 1) = samples[1][selected];
       
    //return returned;
    
}

// update_mu(mu_new_in, MU, Z1sum, Z2sum, A, B, K, N1, N2 )

void update_mu(double* mu_new_in, double* MU, double* Z1sum, double* Z2sum, double A, double B, int K, int N1, int N2 )
{
    
    double* mu_k_minus  = (double*) mxCalloc( 2 , sizeof( double ));
    double* mu_k        = (double*) mxCalloc( 2 , sizeof( double ));
    double* mu_k_plus 	= (double*) mxCalloc( 2 , sizeof( double ));
    
    // returned value of MH
    double* mu_k_tmp    = (double*) mxCalloc( 2 , sizeof( double ));
    
    // for each K 
    // should be parallel
    for (int k = 0; k < K; k++){
                
        if (k == 0){
            *(mu_k_minus)     = 1;
            *(mu_k_minus+1)   = 1;
        }else{
            *(mu_k_minus)     = (double) *(mu_new_in + (k-1));
            *(mu_k_minus+1)   = (double) *(mu_new_in + K + (k-1));
        }
        
        if (k == (K-1)){
            *(mu_k_plus)      = 0;
            *(mu_k_plus+1)    = 0;
        }else{
            *(mu_k_plus)      = (double) *(MU + (k+1)*2 );
            *(mu_k_plus+1)    = (double) *(MU + (k+1)*2 + 1);
        }
        
        *(mu_k)   = (double) *(MU + k*2);
        *(mu_k+1) = (double) *(MU + k*2 + 1);
                
        // MH_mu_bibeta(k, mu_k_minus, mu(:,k), mu_k_plus, Z_1(:,k), Z_2(:,k), N, K_max, a, b);
        MH_mu(mu_k_tmp, k, K, mu_k_minus, mu_k, mu_k_plus, (int) *(Z1sum + k), (int) *(Z2sum + k), N1, N2, A, B);
        
        *(mu_new_in + k)        = *(mu_k_tmp);
        *(mu_new_in + K + k)    = *(mu_k_tmp + 1);
        
    }
    
    //
    mxFree(mu_k_minus);
    mxFree(mu_k);
    mxFree(mu_k_plus);
    mxFree(mu_k_tmp);
   
}


// [mu]   = Update_mu(mu, Z1sum, Z2sum, K_max, N1, N2, a, b)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *MU;
    
    double A, B; 
                
    double *Z1sum, *Z2sum;
    
    int K, N1, N2;
    
    
    /*  collect input external formate variables */
    MU      = mxGetPr(prhs[0]);
    
    Z1sum   = mxGetPr(prhs[1]);  
    Z2sum   = mxGetPr(prhs[2]); 
       
    K       = (int) mxGetScalar(prhs[3]);    
    
    N1      = (int) mxGetScalar(prhs[4]);  
    N2      = (int) mxGetScalar(prhs[5]);  
    
    A       = (double) mxGetScalar(prhs[6]);  
    B       = (double) mxGetScalar(prhs[7]);  
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *MU_NEW; 
    
    /* output variables internal formate */
    double *mu_new_in; 
    
        
    mu_new_in   = (double *) mxCalloc( K*2 , sizeof( double ));
    
        
    /* run the model */
    update_mu(mu_new_in, MU, Z1sum, Z2sum, A, B, K, N1, N2 );
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ]  = mxCreateDoubleMatrix( 2, K, mxREAL );
    
    MU_NEW     = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {
        
        *(MU_NEW + k*2 )    =  *(mu_new_in + k );
        *(MU_NEW + k*2 + 1) =  *(mu_new_in + K + k );
        
//         if (k <= 5)
//             mexPrintf("k = %d ....... mu_new_in1 = %f, mu_new_in2 = %f. \n", k, *(mu_new_in + k), *(mu_new_in + K + k));
        
    }
    
    //
    mxFree(mu_new_in);
}








