
// V2n = mex_update_V_FGM(K_max, V2n, Z1m, Y(:, n), A, N1, N2, alpha2, epsilon)


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
 

// log_likelihood_v(xStar, k, K, N1, Z2n, V2n, A, Yn, epsilon)

double log_likelihood_v(double xStar, int k, int K, int N1, double* Z2n, double* V2n, double* A, double* Yn, double epsilon)
{
 
    double fl=0,lamdam;
    
    for (int m = 0; m < N1; m++){
        
        lamdam = 0;
        
        for (int ki = 0; ki < K ; ki++){
            
            if(ki == k)
            	lamdam = lamdam + ((int)*(Z2n + k)) * (*(V2n + k)) * (*(A + k * N1 + m));
            else
                lamdam = lamdam + ((int)*(Z2n + k)) * (xStar) * (*(A + k * N1 + m));
        }
        
        lamdam = lamdam  + epsilon;
        
        fl = fl -log(lamdam) - *(Yn + m) / lamdam;
        
    }
    
    
    return fl;
    
    
}

// MH_v(*(V2n + k), alpha, k, K, N1, Yn, Z2n, V2n, A, epsilon, gen)

double MH_v(double v2nk, double alpha, int k, int K, int N1, double* Yn, double* Z2n, double* V2n, double* A, double epsilon, GEN  gen)
{

    srand((unsigned)time(0) * k);
    
    int nSamples        = 20;
    int burnIn          = 30;
    int Interval        = 5;
    
	double samples[nSamples];
   
    double x_tmp;
    
    x_tmp = v2nk;
    
    double xStar;
    double x_t_1;
    
	int t   = 1;
	int num = 0;

	double log_p_xstar, log_p_st1, p, ratio, u;

	while(num < nSamples){
    
        xStar       = 1/alpha*gen();
        
		x_t_1       = x_tmp;
        
        
        // 
		log_p_xstar = log_likelihood_v(xStar, k, K, N1, Z2n, V2n, A, Yn, epsilon);
    
		log_p_st1   = log_likelihood_v(x_t_1, k, K, N1, Z2n, V2n, A, Yn, epsilon);
    
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

// update_v(v2_new_in, V2n, Z2n, Yn, A, N1, N2, K, alpha, epsilon )

void update_v(double* v2_new_in, double* V2n, double* Z2n, double* Yn, double* A, int N1, int N2, int K, double alpha, double epsilon)
{
    
    double tmp;
    
    //
     
    ENG  eng;
    GAM  gam(1);
    GEN  gen(eng,gam);
        
    
    // for each K 
    // should be parallel
    for (int k = 0; k < K; k++){
                
        if((int)*(Z2n + k) == 1)
        
            tmp                 = MH_v(*(V2n + k), alpha, k, K, N1, Yn, Z2n, V2n, A, epsilon, gen);
        else
            tmp                 = 1/alpha*gen();
            
        *(v2_new_in + k)    = tmp;
        
    }
    
   
}


// V1m = mex_update_V_FGM(K_max, V2n, Z1m, Y(:, n), A, N1, N2, alpha2, epsilon)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *V2n, *Z2n, *Yn, *A;
    
    double alpha, epsilon;              
    
    int K, N1, N2;
    
    
    /*  collect input external formate variables */
          
    K       = (int) mxGetScalar(prhs[0]);   
    
    V2n     = mxGetPr(prhs[1]);  
    Z2n     = mxGetPr(prhs[2]);     
    Yn      = mxGetPr(prhs[3]);  
    A       = mxGetPr(prhs[4]); 
    
    N1      = (int) mxGetScalar(prhs[5]);  
    N2      = (int) mxGetScalar(prhs[6]);  
    
    alpha   = (double) mxGetScalar(prhs[7]);  
    epsilon = (double) mxGetScalar(prhs[8]);    
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *V2N_NEW; 
    
    /* output variables internal formate */
    double *v2_new_in; 
            
    v2_new_in   = (double *) mxCalloc( K , sizeof( double ));
    
        
    /* run the model */
    update_v(v2_new_in, V2n, Z2n, Yn, A, N1, N2, K, alpha, epsilon );
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ]  = mxCreateDoubleMatrix( 1, K, mxREAL );
    
    V2N_NEW    = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {
        
        *(V2N_NEW + k )    =  *(v2_new_in + k );
        
    }
    
    //
    mxFree(v2_new_in);
}








