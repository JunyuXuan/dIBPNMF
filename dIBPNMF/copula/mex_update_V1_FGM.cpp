
// V1m = mex_Update_V_FGM(K_max, V1m, Z1m, Y(m, :), X, alpha1, epsilon)


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


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>

//#include "storage_adaptors.hpp"

using namespace boost::numeric::ublas;
 
typedef boost::mt19937                     ENG;    // Mersenne Twister
typedef boost::gamma_distribution<double>  GAM;   // Normal Distribution
typedef boost::variate_generator<ENG,GAM>  GEN;    // Variate generator
 

// log_likelihood_v(xStar, k, K, N2, Z1m, V1m, X, Ym, epsilon)

double log_likelihood_v(double xStar, int k, int K, int N2, double* Z1m, double* V1m, double* X, double* Ym, double epsilon)
{
 
    double fl=0,lamdan;
    
    for (int n = 0; n < N2; n++){
        
        lamdan = 0;
        
        for (int ki = 0; ki < K ; ki++){
            
            if(ki == k)
            	lamdan = lamdan + ((int)*(Z1m + k)) * (*(V1m + k)) * (*(X + k * N2 + n));
            else
                lamdan = lamdan + ((int)*(Z1m + k)) * (xStar) * (*(X + k * N2 + n));
        }
        
        lamdan = lamdan  + epsilon;
        
        fl = fl -log(lamdan) - *(Ym + n) / lamdan;
        
    }
    
    
    return fl;
    
    
}

// MH_v(*(V1m + k), alpha, k, K, N2, Ym, Z1m, V1m, X, epsilon);

double MH_v(double v1mk, double alpha, int k, int K, int N2, double* Ym, double* Z1m, double* V1m, double* X, double epsilon, GEN  gen)
{

    srand((unsigned)time(0) * k);
    
    int nSamples        = 20;
    int burnIn          = 30;
    int Interval        = 5;
    
	double samples[nSamples];
   
    double x_tmp;
    
    x_tmp = v1mk;
    
    double xStar;
    double x_t_1;
    
	int t   = 1;
	int num = 0;

	double log_p_xstar, log_p_st1, p, ratio, u;

    
    //
    
    
	while(num < nSamples){
    
        xStar       = 1/alpha*gen();
        
		x_t_1       = x_tmp;
        
        
        // log_likelihood_mu(xStar, mu_k_minus, mu_k_plus, Z1sum, Z2sum, N1, N2, A, B, k, K, RHO)
		log_p_xstar = log_likelihood_v(xStar, k, K, N2, Z1m, V1m, X, Ym, epsilon);
    
		log_p_st1   = log_likelihood_v(x_t_1, k, K, N2, Z1m, V1m, X, Ym, epsilon);
    
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

// update_v(v1_new_in, V1m, Z1m, Ym, X, N1, N2, K, alpha, epsilon )

void update_v(double* v1_new_in, double* V1m, double* Z1m, double* Ym, double* X, int N1, int N2, int K, double alpha, double epsilon)
{
    
    double tmp;
    
    //
     
    ENG  eng;
    GAM  gam(1);
    GEN  gen(eng,gam);
        
    
    // for each K 
    // should be parallel
    for (int k = 0; k < K; k++){
                
        if((int)*(Z1m + k) == 1)
        
            tmp                 = MH_v(*(V1m + k), alpha, k, K, N2, Ym, Z1m, V1m, X, epsilon, gen);
        else
            tmp                 = 1/alpha*gen();
            
        *(v1_new_in + k)    = tmp;
        
    }
    
   
}


// V1m = mex_Update_V1_FGM(K_max, V1m, Z1m, Y(m, :), X, N1, N2, alpha1, epsilon)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *V1m, *Z1m, *Ym, *X;
    
    double alpha, epsilon;              
    
    int K, N1, N2;
    
    
    /*  collect input external formate variables */
          
    K       = (int) mxGetScalar(prhs[0]);   
    
    V1m     = mxGetPr(prhs[1]);  
    Z1m     = mxGetPr(prhs[2]);     
    Ym      = mxGetPr(prhs[3]);  
    X       = mxGetPr(prhs[4]); 
    
    N1      = (int) mxGetScalar(prhs[5]);  
    N2      = (int) mxGetScalar(prhs[6]);  
    
    alpha   = (double) mxGetScalar(prhs[7]);  
    epsilon = (double) mxGetScalar(prhs[8]);    
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *V1M_NEW; 
    
    /* output variables internal formate */
    double *v1_new_in; 
            
    v1_new_in   = (double *) mxCalloc( K , sizeof( double ));
    
        
    /* run the model */
    update_v(v1_new_in, V1m, Z1m, Ym, X, N1, N2, K, alpha, epsilon );
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ]  = mxCreateDoubleMatrix( 1, K, mxREAL );
    
    V1M_NEW    = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {
        
        *(V1M_NEW + k )    =  *(v1_new_in + k );
        
    }
    
    //
    mxFree(v1_new_in);
}








