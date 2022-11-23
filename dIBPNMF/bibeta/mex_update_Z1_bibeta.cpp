
// Z1m = mex_update_Z1_bibeta(Z1m, V1m, K_max, mu, Y(m, :), X, N2, epsilon);


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


// update_z1(z1m_new_in, Z1m, V1m, K, MU, Ym, X, N2, epsilon )

void update_z1(int* z1m_new_in, double* Z1m, double* V1m, int K, double* MU, double* Ym, double* X, 
        int N2, double epsilon, double* L )
{
        
    srand((unsigned)time(0) * (*(MU + 1)));
    
    double mu_tmp;
   
    double* L1 = (double *) mxCalloc( N2 , sizeof( double ));
    
    double likelihood0, likelihood1;
    
    double log_p_z_0, log_p_z_1, p, u, v1mk;
    
    // for each K 
    for (int k = 0; k < K; k++){
                
        mu_tmp = (double)*(MU + k);
       
        log_p_z_1 = log(mu_tmp + 0.0000000001);
        log_p_z_0 = log(1 - mu_tmp + 0.0000000001);
        
        v1mk = (double) *(V1m + k);
        
        for(int n = 0 ; n < N2; n++){
                     
            likelihood0 = (double)*(L + n);
            likelihood1 =  likelihood0 + (double)*(X + k * N2 + n) * v1mk;
                        
            log_p_z_1 = log_p_z_1 - log(likelihood1 + epsilon + 0.00000001) - (double)*(Ym + n) / (likelihood1 + epsilon) ;
            log_p_z_0 = log_p_z_0 - log(likelihood0 + epsilon + 0.00000001) - (double)*(Ym + n) / (likelihood0 + epsilon) ;
            
            *(L1 + n) = likelihood1;
        }
        
        p      = 1 / ( 1 + exp(log_p_z_0 - log_p_z_1));
        
        u      = rand() /double(RAND_MAX);
    
		if (u < p){
			*(z1m_new_in + k)  = 1;
            
            L = L1;                        
		}else{
			*(z1m_new_in + k)  = 0;
        }	
               
        
    }
    
    mxFree(L1);
       
}


// Z1m = mex_update_Z1_bibeta(Z1m, V1m, K, mu, Y(m, :), X, N2, epsilon, L)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *Z1m, *V1m, *MU, *Ym, *X, *L;
    
    double epsilon; 
                
    int K, N2;
    
    
    /*  collect input external formate variables */
    Z1m     = mxGetPr(prhs[0]);    
    V1m     = mxGetPr(prhs[1]); 
    K       = (int) mxGetScalar(prhs[2]); 
    MU      = mxGetPr(prhs[3]); 
       
    Ym      = mxGetPr(prhs[4]);    
    X       = mxGetPr(prhs[5]);
   
    N2      = (int) mxGetScalar(prhs[6]);  
    
    epsilon = (double) mxGetScalar(prhs[7]);  
    
    L       = mxGetPr(prhs[8]);
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *Z1m_NEW; 
    
    /* output variables internal formate */
    int *z1m_new_in; 
            
    z1m_new_in   = (int *) mxCalloc( K , sizeof( int ));
    
        
    /* run the model */
    update_z1(z1m_new_in, Z1m, V1m, K, MU, Ym, X, N2, epsilon, L );
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ]  = mxCreateDoubleMatrix( 1, K, mxREAL );
    
    Z1m_NEW    = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {        
        *(Z1m_NEW + k)    =  (double)*(z1m_new_in + k );        
    }
    
    //
    mxFree(z1m_new_in);
}








