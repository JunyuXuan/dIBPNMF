
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

// #include "ranlib.hpp"
// #include "rnglib.hpp"


#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <limits>
//     using std::numeric_limits;

    
    
// log_likelihood_xn(xStar, k, M, K, XN, YN, A)

double log_likelihood_xn(double xStar, int k, int M, int K, double* XN, double* YN, double* A)
{
    double fl = 0, lamdam;
    
    for (int m = 0; m < M; m++){
     
        lamdam = 0;
        
        for (int ki = 0; ki < K; ki++){
         
            if (ki == k){
                lamdam = lamdam + xStar * (*(A + ki*M + m)) ;
            }else{
                lamdam = lamdam + *(XN + ki) * (*(A + ki*M + m)) ;
            }
        }
        
        fl = fl - log(lamdam + 0.00000000001) - *(YN + m) / (lamdam + 0.00000000001) ;
        
    }
    
    
    return fl;
    
}

// MH_xn(*(XN + k), k, XN, *(V2N +k)* (*(Z2N + k) + epsilon), YN, A, K, M)

double MH_xn(double xn, int k, double* XN, double vz, double* YN, double* A, int K, int M)
{
  
    srand((unsigned)time(0) * k);
    
    int nSamples        = 10;
    int burnIn          = 10;
    int Interval        = 5;
    
    typedef boost::mt19937                     ENG;    // Mersenne Twister
    typedef boost::gamma_distribution<double>  GAM;   // Normal Distribution
    typedef boost::variate_generator<ENG,GAM>  GEN;    // Variate generator
 
    ENG  eng;
    GAM  gam(2);
    GEN  gen(eng,gam);
    
    
	double samples[nSamples];
   
    double x_tmp = xn;
    double xStar;
    double x_t_1;
    
	int t   = 1;
	int num = 0;

	double log_p_xstar, log_p_st1, p, ratio, u;

	while(num < nSamples){
    
//         mexPrintf("in MH_mu k = %d  num = %d sigma1k = %f sigma2k = %f . \n", k, t, sigma1k, sigma2k);
    
        xStar       = 1/(vz*gen());
    
		x_t_1       = x_tmp;
                
     
		log_p_xstar = log_likelihood_xn(xStar, k, M, K, XN, YN, A);
    
		log_p_st1   = log_likelihood_xn(x_t_1, k, M, K, XN, YN, A);
    
		p           = exp( log_p_xstar - log_p_st1);
        
        
        if (p <= 1){
            ratio = p;
        }else{                 
            ratio = 1;
        }
		   
		u           = rand() /double(RAND_MAX);
    
		if (u < ratio){
			x_tmp           = xStar;	            	
		}else{
			x_tmp           = x_t_1;            
        }	
            
		if (t > burnIn){

			if( t % Interval == 0 ){

				samples[num] = x_tmp;                
				num++;
			}

		}
        
        t++;
   
	}
	
    int selected 	= (rand() % (nSamples-1 -0+1))+ 0;
     
    return samples[selected];
    
}

// update_xn(xn_new_in, XN, V2N, Z2N, YN, A, epsilon, K, M)

void update_xn(double* xn_new_in, double* XN, double* V2N, double* Z2N, double* YN, 
        double* A, double epsilon, int K, int M)
{
    double xn_k_tmp;    
    
    // for each K 
    // should be parallel
    for (int k = 0; k < K; k++){
              
        xn_k_tmp = MH_xn(*(XN + k), k, XN, *(V2N +k)* (*(Z2N + k) + epsilon), YN, A, K, M);
        
        *(xn_new_in + k)  = xn_k_tmp;
                
    }
       
}

// Update_Xn(Xn, V2n, Z2n, Yn, A, epsilon, K_max, M)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *XN, *V2N, *Z2N, *YN, *A;
    
    double epsilon; 
                
    int K, M;
    
    
    /*  collect input external formate variables */
    XN          = mxGetPr(prhs[0]);
    V2N         = mxGetPr(prhs[1]);
    Z2N         = mxGetPr(prhs[2]);
    YN          = mxGetPr(prhs[3]);
    A           = mxGetPr(prhs[4]);
    
    epsilon     = (double) mxGetScalar(prhs[5]);    
    K           = (int) mxGetScalar(prhs[6]);  
    M           = (int) mxGetScalar(prhs[7]); 
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *XN_NEW; 
    
    /* output variables internal formate */
    double *xn_new_in; 
            
    xn_new_in   = (double *) mxCalloc( K , sizeof( double ));
    
        
    /* run the model */
    
    update_xn(xn_new_in, XN, V2N, Z2N, YN, A, epsilon, K, M);
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ] = mxCreateDoubleMatrix( 1, K, mxREAL );
    
    XN_NEW    = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {        
        *(XN_NEW + k )    =  *(xn_new_in + k );                
    }
    
    //
    mxFree(xn_new_in);
    
}








