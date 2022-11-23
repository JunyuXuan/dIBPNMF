
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

// #include "ranlib.hpp"
// #include "rnglib.hpp"


#include <boost/math/distributions/normal.hpp>
    using boost::math::normal;
#include <limits>
//     using std::numeric_limits;


// Returns the erf() of a value (not super precice, but ok)
// double erf(double x)
// {  
//  double y = 1.0 / ( 1.0 + 0.3275911 * x);   
//  return 1 - (((((
//         + 1.061405429  * y
//         - 1.453152027) * y
//         + 1.421413741) * y
//         - 0.284496736) * y 
//         + 0.254829592) * y) 
//         * exp (-x * x);      
// }
// 
// 
// // Returns the probability of [-inf,x] of a gaussian distribution
// double cdf(double x, double mu, double sigma)
// {
// 	return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2.))));
// }

// log_likelihood_mu(xStar, k, K, sigma1, sigma2, Z1sum, Z2sum, N1, N2, g1, g2, rho, alpha, muK)

double log_likelihood_mu(double xStar, int k, int K, double sigma1, double sigma2, int z1, int z2, 
        int N1, int N2, double g1, double g2, double rho, double alpha, double muK, normal myNormal1, normal myNormal2, normal myNormal3)
{
//     mexPrintf("in log_likelihood_mu.......  \n");
//         
//     mexPrintf("in log_likelihood_mu....... after definitions, xStar = %f. \n", xStar);
    
    double norminvvalue1;
    try{      
        norminvvalue1   = quantile(myNormal1, xStar);//norminv(xStar, 0, sigma1 + rho);
    }
    catch(const std::exception& e)
    {
//         mexPrintf("error!! in log_likelihood_mu.......  \n", e.what());
        if(norminvvalue1 > 0){
            norminvvalue1 = (numeric_limits<double>::max)();
        }else{
            norminvvalue1 = (numeric_limits<double>::min)();
        }
    }
    
    double norminvvalue2;
    try{      
        norminvvalue2   = quantile(myNormal2, xStar);//norminv(xStar, 0, sigma2 + rho);
    }
    catch(const std::exception& e)
    {
//         mexPrintf("error!! in log_likelihood_mu.......  \n", e.what());
        if(norminvvalue2 > 0){
            norminvvalue2 = (numeric_limits<double>::max)();
        }else{
            norminvvalue2 = (numeric_limits<double>::min)();
        }
    }
        
    
//     mexPrintf("in log_likelihood_mu....... norminvvalue1 = %f, norminvvalue2 = %f, g1 = %f, g2 = %f. \n", norminvvalue1, norminvvalue2, g1, g2);
    
    
    double gamma_kt1 = cdf(myNormal3, norminvvalue1 - g1);
    double gamma_kt2 = cdf(myNormal3, norminvvalue2 - g2);
        
    double fl;
    
    if (k == (K-1)){
    
        fl = (alpha-1)*log(xStar+0.0000000001) 
                    + z1*log( gamma_kt1 + 0.0000000001) + (N1 - z1) * log(1 - gamma_kt1+ 0.0000000001) 
                    + z2*log( gamma_kt2 + 0.0000000001) + (N2 - z2) * log(1 - gamma_kt2 + 0.0000000001) ;
    }else{
        
        fl = alpha* log(muK +0.0000000001) - log(xStar + 0.0000000001)
                    + z1*log( gamma_kt1 + 0.0000000001) + (N1 - z1) * log(1 - gamma_kt1+ 0.0000000001) 
                    + z2*log( gamma_kt2 + 0.0000000001) + (N2 - z2) * log(1 - gamma_kt2 + 0.0000000001) ;
    }
    
    return fl;
    
}

// MH_mu(k, K, mu_k_minus, mu_k, mu_k_plus, muK, z1sum, z2sum, g1k, g2k, sigma1k, sigma2k, N1, N2, rho, alpha)

double MH_mu(int k, int K, double mu_k_minus, double mu_k, double mu_k_plus, double muK, int Z1sum, int Z2sum, double g1k,
        double g2k, double sigma1k, double sigma2k, int N1, int N2, double RHO, double alpha)
{
//     if (k <= 10)
//     mexPrintf("in MH_mu k = %d ....... mu1(k-1) = %f, mu1(k+1) = %f, mu2(k-1) = %f, mu2(k+1) = %f. \n", k, *(mu_k_minus), *(mu_k_plus), *(mu_k_minus+1), *(mu_k_plus+1));
    
    srand((unsigned)time(0) * k);
    
    normal myNormal1(0, sigma1k+RHO);
    normal myNormal2(0, sigma2k+RHO);    
    normal myNormal3(0, RHO);
    
    int nSamples        = 20;
    int burnIn          = 30;
    int Interval        = 5;
    
	double samples[nSamples];
   
    double x_tmp = mu_k;
    double xStar;
    double x_t_1;
    
	int t   = 1;
	int num = 0;

	double log_p_xstar, log_p_st1, p, ratio, u;

	while(num < nSamples){
    
//         mexPrintf("in MH_mu k = %d  num = %d sigma1k = %f sigma2k = %f . \n", k, t, sigma1k, sigma2k);
    
        u = rand() /double(RAND_MAX);
        
        while(u > 1 || u < 0){
            u = rand() /double(RAND_MAX);
        }
        
        xStar      = u * ( mu_k_minus - mu_k_plus) + mu_k_plus;    
        
        if ( (xStar < mu_k_plus || xStar > mu_k_minus) && xStar <= 0.000001){
            xStar = mu_k_plus;
        }
    
		x_t_1       = x_tmp;
                
        // log_likelihood_mu(xStar, sigma1, sigma2, Z1sum, Z2sum, N1, N2, g1, g2, rho, alpha, muK)
		log_p_xstar = log_likelihood_mu(xStar, k, K, sigma1k, sigma2k, Z1sum, Z2sum, N1, N2, g1k, g2k, RHO, alpha, muK, myNormal1, myNormal2, myNormal3);
    
		log_p_st1   = log_likelihood_mu(x_t_1, k, K, sigma1k, sigma2k, Z1sum, Z2sum, N1, N2, g1k, g2k, RHO, alpha, muK, myNormal1, myNormal2, myNormal3);
    
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
	
//     delete myNormal1;
//     delete myNormal2;
//     delete myNormal3;
    
    int selected 	= (rand() % (nSamples-1 -0+1))+ 0;
     
    return samples[selected];
    
}

//update_mu(mu_new_in, MU, G1, G2, SIGMA1, SIGMA2, RHO, K, ALPHA, Z1sum, Z2sum, T, N1, N2)

void update_mu(double* mu_new_in, double* MU, double* G1, double* G2, double* SIGMA1, double* SIGMA2, double RHO, int K, 
        double ALPHA, double* Z1sum, double* Z2sum, int T, int N1, int N2 )
{
    double mu_k_minus  = 0;
    double mu_k        = 0;
    double mu_k_plus   = 0;
    
    double mu_KK       = (double) *(MU + K-1);
    
    // returned value of MH
    double mu_k_tmp    = 0;
    
    // for each K 
    // should be parallel
    for (int k = 0; k < K; k++){
                
        if (k == 0){
            mu_k_minus     = 1;            
        }else{
            mu_k_minus     = (double) *(mu_new_in + (k-1));
            
        }
        
        if (k == (K-1)){
            mu_k_plus      = 0;            
        }else{
            mu_k_plus      = (double) *(MU + (k+1) );            
        }
        
        mu_k   = (double) *(MU + k);
               
        // MH_mu(k, K, mu_k_minus, mu_k, mu_k_plus, muK, z1sum, z2sum, g1k, g2k, sigma1k, sigma2k, N1, N2, rho, alpha);
        mu_k_tmp = MH_mu(k, K, mu_k_minus, mu_k, mu_k_plus, mu_KK, (int) *(Z1sum + k), (int) *(Z2sum + k), 
                        (double) *(G1 + k), (double) *(G2 + k), (double) *(SIGMA1 + k), (double) *(SIGMA2 + k), N1, N2, RHO, ALPHA);
        
        *(mu_new_in + k)  = mu_k_tmp;
                
    }
       
}

// [mu]  = Update_mu(mu, g, Sigma, rho, K_max, alpha, Z1, Z2, T, N1, N2)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double *MU, *G1, *G2, *SIGMA1, *SIGMA2;
    
    double RHO, C0, ALPHA; 
                
    double *Z1, *Z2;
    
    int K, T, N1, N2;
    
    
    /*  collect input external formate variables */
    MU      = mxGetPr(prhs[0]);
    G1       = mxGetPr(prhs[1]);
    G2       = mxGetPr(prhs[2]);
    SIGMA1   = mxGetPr(prhs[3]);
    SIGMA2   = mxGetPr(prhs[4]);
    
    RHO     = (double) mxGetScalar(prhs[5]);    
    K       = (int) mxGetScalar(prhs[6]);    
    ALPHA   = (double) mxGetScalar(prhs[7]);    
    
    Z1      = mxGetPr(prhs[8]);  
    Z2      = mxGetPr(prhs[9]);  
    
    T       = (int) mxGetScalar(prhs[10]);
    N1      = (int) mxGetScalar(prhs[11]);  
    N2      = (int) mxGetScalar(prhs[12]);  
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    double *MU_NEW; 
    
    /* output variables internal formate */
    double *mu_new_in; 
            
    mu_new_in   = (double *) mxCalloc( K , sizeof( double ));
    
        
    /* run the model */
    
    // mex_update_mu(mu, g(1, :), g(2, :), Sigma(:, 1, 1), Sigma(:, 2, 2), rho, K_max, alpha, sum(Z{1}), sum(Z{2}), T, N(1), N(2))
    
    update_mu(mu_new_in, MU, G1, G2, SIGMA1, SIGMA2, RHO, K, ALPHA, Z1, Z2, T, N1, N2);
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ] = mxCreateDoubleMatrix( 1, K, mxREAL );
    
    MU_NEW    = mxGetPr( plhs[ 0 ] );
    
    for (int k=0; k<K; k++) {        
        *(MU_NEW + k )    =  *(mu_new_in + k );                
    }
    
    //
    mxFree(mu_new_in);
    
}








