
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

    
    
// log_likelihood_am(xStar, k, N, K, AM, YM, X)

double log_likelihood_am(double xStar, int k, int N, int K, double* AM, double* YM, double* X)
{
    double fl = 0, lamdan;
    
    for (int n = 0; n < N; n++){
     
        lamdan = 0;
        
        for (int ki = 0; ki < K; ki++){
         
            if (ki == k){
                lamdan = lamdan + xStar * (*(X + ki*N + n)) ;
            }else{
                lamdan = lamdan + *(AM + ki) * (*(X + ki*N + n)) ;
            }
        }
        
        fl = fl - log(lamdan + 0.00000000001) - *(YM + n) / (lamdan + 0.00000000001) ;
        
    }
    
    
    return fl;
    
}

double MH_am(double am, int k, double* AM, double vz, double* YM, double* X, int K, int N)
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
   
    double x_tmp = am;
    double xStar;
    double x_t_1;
    
	int t   = 1;
	int num = 0;

	double log_p_xstar, log_p_st1, p, ratio, u;

	while(num < nSamples){
    
//         mexPrintf("in MH_mu k = %d  num = %d sigma1k = %f sigma2k = %f . \n", k, t, sigma1k, sigma2k);
    
        xStar       = 1/(vz*gen());
    
		x_t_1       = x_tmp;
                
        // log_likelihood_mu(xStar, sigma1, sigma2, Z1sum, Z2sum, N1, N2, g1, g2, rho, alpha, muK)
		log_p_xstar = log_likelihood_am(xStar, k, N, K, AM, YM, X);
    
		log_p_st1   = log_likelihood_am(x_t_1, k, N, K, AM, YM, X);
    
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

void update_am(double* am_new_in, double* AM, double* V1M, double* Z1M, double* YM, 
        double* X, double epsilon, int K, int N, int L, double* IDX)
{
    double am_k_tmp;    
    
    // for each K 
    // should be parallel
    for (int ki = 0; ki < L; ki++){
              
        k                   = *(IDX + ki);
        
        am_k_tmp            = MH_am(*(AM + ki), ki, AM, *(V1M +k)* (*(Z1M + k) + epsilon), YM, X, K, N);
        
        *(am_new_in + k)    = am_k_tmp;
                
    }
       
}

// [mu]  = Update_Am(Am, V1m, Z1m, Ym, X, epsilon, K_max, N)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *AM, *V1M, *Z1M, *YM, *X, *IDX;
    
    double epsilon; 
                
    int K, N, L;
    
    
    /*  collect input external formate variables */
    AM          = mxGetPr(prhs[0]);
    V1M         = mxGetPr(prhs[1]);
    Z1M         = mxGetPr(prhs[2]);
    YM          = mxGetPr(prhs[3]);
    X           = mxGetPr(prhs[4]);
    
    epsilon     = (double) mxGetScalar(prhs[5]);    
    K           = (int) mxGetScalar(prhs[6]);  
    N           = (int) mxGetScalar(prhs[7]); 
    L           = (int) mxGetScalar(prhs[8]);
    IDX         = mxGetPr(prhs[9]);
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *AM_NEW; 
    
    /* output variables internal formate */
    double *am_new_in; 
            
    am_new_in   = (double *) mxCalloc( K , sizeof( double ));
    
        
    /* run the model */
    
    update_am(am_new_in, AM, V1M, Z1M, YM, X, epsilon, K, N, L, IDX);
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ] = mxCreateDoubleMatrix( 1, K, mxREAL );
    
    AM_NEW    = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {        
        *(AM_NEW + k )    =  *(am_new_in + k );                
    }
    
    //
    mxFree(am_new_in);
    
}








