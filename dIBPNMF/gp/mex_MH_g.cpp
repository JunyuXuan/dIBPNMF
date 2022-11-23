
// g_new(:, k) = MH_g(k, g, T, N, h, rho, Sigma);


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


double normal_pdf(double x, double m, double s)
{
    static const float inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}


// log_likelihood_mu(xStar, h1k, h2k, rho, N1, N2)

double log_likelihood_g(double* xStar, double* h1k, double* h2k, double rho, int N1, int N2)
{
 
    double returnvalue = 0;
    
    for (int i = 0; i < N1; i++){    
        returnvalue = returnvalue + log(normal_pdf(*(h1k+i), *(xStar), rho) + 0.0000000001) ;
    }
    
    for (int i = 0; i < N2; i++){    
        returnvalue = returnvalue + log(normal_pdf(*(h2k+i), *(xStar+1), rho) + 0.0000000001) ;
    }
    
    if(isnan(returnvalue)){
         mexPrintf("in log_likelihood_g....... returnvalue = %f  \n", returnvalue);   
    }
    
    return returnvalue;
    
}

// sample_xStar(returned, Sigma_k)

void sample_xStar(double* returned, double* Sigma_k)
{
    
//     *(returned)   = rand() /double(RAND_MAX);
//     *(returned+1) = rand() /double(RAND_MAX);
    
    float meanv[2];
    
    meanv[0] = 0;
    meanv[1] = 0;
    
    float covm[4];
    
    covm[0] = *(Sigma_k);
    covm[1] = *(Sigma_k+1);
    covm[2] = *(Sigma_k+2);
    covm[3] = *(Sigma_k+3);
    
    int p = 2;
    
    float parm[2];
    
    setgmn( meanv, covm, p, parm );
    
    float* sample = genmn(parm);
    
//     mexPrintf("in sample_xStar....... sample[0] = %f, sample[1] = %f  \n", (double) *(sample), (double) *(sample+1)); 
    
    *(returned)     = (double) *(sample);
    *(returned+1)   = (double) *(sample+1);
    
        
}

// MH_g(g_new_in, k, G, T, N1, N2, H1, H2, RHO, Sigmak);

void MH_g(double* g_new_in, int k, double* G, int T, int N1, int N2, double* H1, double* H2, double RHO, double* Sigmak)
{
  
    srand((unsigned)time(0) * k);
    
    int nSamples        = 20;
    int burnIn          = 30;
    int Interval        = 5;
    
	double samples[2][nSamples];
   
    double x_tmp[2];
    double xStar[2];
    double x_t_1[2];
    
	int t   = 1;
	int num = 0;

	double log_p_xstar, log_p_st1, p, ratio, u;

	while(num < nSamples){
    
        sample_xStar(xStar, Sigmak);
    
//         mexPrintf("in MH_g....... xStar[0] = %f, xStar[1] = %f  \n", xStar[0], xStar[1]); 
    
        
		x_t_1[0]       = x_tmp[0];
        x_t_1[1]       = x_tmp[1];
                
        // log_likelihood_mu(xStar, h1k, h2k, rho, N1, N2)
        
		log_p_xstar = log_likelihood_g(xStar, H1, H2, RHO, N1, N2);
    
		log_p_st1   = log_likelihood_g(x_t_1, H1, H2, RHO, N1, N2);
    
		p           = exp( log_p_xstar - log_p_st1);
        
        
        if (p <= 1){
            ratio = p;
        }else{                 
            ratio = 1;
        }
		   
		u           = rand() /double(RAND_MAX);
    
		if (u < ratio){
			x_tmp[0]        = xStar[0];	
            x_tmp[1]        = xStar[1];	
		}else{
			x_tmp[0]        = x_t_1[0];
            x_tmp[1]        = x_t_1[1];
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
	
    int selected  = (rand() % (nSamples-1 -0+1))+ 0;
     
    *(g_new_in)   = samples[0][selected];
    *(g_new_in+1) = samples[1][selected];
    
}

// g_new(:, k) = MH_g(k, g, T, N1, N2, h1k, h2k, rho, Sigmak);

void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *G, *SIGMA1, *SIGMA2, *Sigmak, *H1, *H2;
    
    double RHO;                 
    
    int k, T, N1, N2;
    
    
    /*  collect input external formate variables */
    k       = (int) mxGetScalar(prhs[0]);
    G       = mxGetPr(prhs[1]);
    T       = (int) mxGetScalar(prhs[2]);
    N1      = (int) mxGetScalar(prhs[3]);
    N2      = (int) mxGetScalar(prhs[4]);
    
    H1      = mxGetPr(prhs[5]);
    H2      = mxGetPr(prhs[6]);
    
    RHO     = (double) mxGetScalar(prhs[7]);    
          
    Sigmak  = mxGetPr(prhs[8]);  
     
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *G_NEW; 
    
    /* output variables internal formate */
    double *g_new_in; 
            
    g_new_in   = (double *) mxCalloc( 2 , sizeof( double ));
            
    /* run the model */
    
    // mex_update_mu(mu, g(1, :), g(2, :), Sigma(:, 1, 1), Sigma(:, 2, 2), rho, K_max, alpha, sum(Z{1}), sum(Z{2}), T, N(1), N(2))
    
    MH_g(g_new_in, k, G, T, N1, N2, H1, H2, RHO, Sigmak);
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ] = mxCreateDoubleMatrix( 2, 1, mxREAL );    
    G_NEW     = mxGetPr( plhs[ 0 ] );
                
    *(G_NEW  )      =  *(g_new_in  );   
    *(G_NEW +1 )    =  *(g_new_in +1 );   
    
    //
    mxFree(g_new_in);
    
}








