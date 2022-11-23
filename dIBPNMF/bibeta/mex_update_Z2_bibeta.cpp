
// Z2n = mex_update_Z2_bibeta(Z2n, V2n, K_max, mu2, Y(:, n), A, N1, epsilon)


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


// update_z2(z2n_new_in, Z2n, V2n, K, MU, Yn, A, N1, epsilon )

void update_z2(int* z2n_new_in, double* Z2n, double* V2n, int K, double* MU, double* Yn, double* A, 
        int N1, double epsilon )
{
        
    srand((unsigned)time(0) * (*(MU + 1)));
    
    double mu_tmp;
    
    double likelihood1, likelihood2;
    
    double log_p_z_0, log_p_z_1, p, u, tmp;
    
    // for each K 
    for (int k = 0; k < K; k++){
                
        mu_tmp = (double)*(MU + k);
       
        log_p_z_1 = log(mu_tmp + 0.0000000001);
        log_p_z_0 = log(1 - mu_tmp + 0.0000000001);
        
        for(int n = 0 ; n < N1; n++){
         
            likelihood1 = 0;
            likelihood2 = 0;
            
            for (int ki = 0; ki < K; ki++){
                
                if(ki < k){
                    tmp = (*(z2n_new_in + ki)) * ((double)*(V2n + ki)) *((double)*(A + ki * N1 + n));
                    likelihood1 = likelihood1 + tmp;
                    likelihood2 = likelihood2 + tmp;
                }else if(ki == k){
                    likelihood1 = likelihood1 + 1 * ((double)*(V2n + ki)) *((double)*(A + ki * N1 + n));
                }else{
                    tmp = ((int)*(Z2n + ki)) * ((double)*(V2n + ki)) *((double)*(A + ki * N1 + n));
                    likelihood1 = likelihood1 + tmp;
                    likelihood2 = likelihood2 + tmp;
                }
            }   
            
            log_p_z_1 = log_p_z_1 - log(likelihood1 + 0.00000001) - exp(log((double)*(Yn + n) + 0.00000001 ) - log(likelihood1 + 0.00000001)) ;
            log_p_z_0 = log_p_z_0 - log(likelihood2 + 0.00000001) - exp(log((double)*(Yn + n) + 0.00000001 ) - log(likelihood2 + 0.00000001)) ;
        }
        
        p      = 1 / ( 1 + exp(log_p_z_0 - log_p_z_1));
        
        u      = rand() /double(RAND_MAX);
    
		if (u < p){
			*(z2n_new_in + k)  = 1;
		}else{
			*(z2n_new_in + k)  = 0;
        }	
               
        
    }
    
   
}


// Z2n = mex_update_Z2_bibeta(Z2n, V2n, K_max, mu2, Y(:, n), A, N1, epsilon)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *Z2n, *V2n, *MU, *Yn, *A;
    
    double epsilon; 
                
    int K, N1;
    
    
    /*  collect input external formate variables */
    Z2n     = mxGetPr(prhs[0]);    
    V2n     = mxGetPr(prhs[1]); 
    K       = (int) mxGetScalar(prhs[2]); 
    MU      = mxGetPr(prhs[3]); 
       
    Yn      = mxGetPr(prhs[4]);    
    A       = mxGetPr(prhs[5]);
   
    N1      = (int) mxGetScalar(prhs[6]);  
    
    epsilon = (double) mxGetScalar(prhs[7]);  
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *Z2n_NEW; 
    
    /* output variables internal formate */
    int *z2n_new_in; 
            
    z2n_new_in   = (int *) mxCalloc( K , sizeof( int ));
    
        
    /* run the model */
    update_z2(z2n_new_in, Z2n, V2n, K, MU, Yn, A, N1, epsilon );
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ]  = mxCreateDoubleMatrix( 1, K, mxREAL );
    
    Z2n_NEW    = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {        
        *(Z2n_NEW + k)    =  (double)*(z2n_new_in + k );        
    }
    
    //
    mxFree(z2n_new_in);
}








