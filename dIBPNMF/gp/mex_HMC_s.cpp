
// s_new(k) = HMC_s(s(k), T, sigma, s_para, g(:, k))


#include "mex.h" 

//#include "stdafx.h" 
#include <numeric> 
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
//#include <random>
#include <cmath>
#include <cstring>

using namespace std;

// #include "ranlib.hpp"
// #include "rnglib.hpp"


#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include "storage_adaptors.hpp"

using namespace boost::numeric::ublas;
//namespace bnu = boost::numeric::ublas;

/* Matrix inversion routine.
 Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
template<class T>
bool InvertMatrix(const matrix<T>& input, matrix<T>& inverse)
{
	typedef permutation_matrix<std::size_t> pmatrix;

	// create a working copy of the input
	matrix<T> A(input);

	// create a permutation matrix for the LU-factorization
	pmatrix pm(A.size1());

	// perform LU-factorization
	int res = lu_factorize(A, pm);
	if (res != 0)
		return false;

	// create identity matrix of "inverse"
	inverse.assign(identity_matrix<T> (A.size1()));

	// backsubstitute to get the inverse
	lu_substitute(A, pm, inverse);

	return true;
}

int determinant_sign(const permutation_matrix<std::size_t>& pm)
{
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
}
 
double determinant( matrix<double>& m ) {
    permutation_matrix<std::size_t> pm(m.size1());
    double det = 1.0;
    if( lu_factorize(m,pm) ) {
        det = 0.0;
    } else {
        for(int i = 0; i < m.size1(); i++) 
            det *= m(i,i); // multiply by elements on diagonal
        det = det * determinant_sign( pm );
    }
    return det;
}

double dU_func(double x0, int T, double sigma, double S_PARA, double* G)
{
    matrix<double> sigmak (T, T);
    matrix<double> dsigmak (T, T);
    
    for (int i = 0 ;i < T; i++){
        for (int j = 0; j < T; j++){
         
            sigmak(i, j) = sigma * exp(-1 * pow(*(G+i) - *(G+j), 2) * pow(x0, -2));
            dsigmak(i, j) = sigma * exp(-1 * pow(*(G+i) - *(G+j), 2) * pow(x0, -2))
                                * (2 * pow(*(G+i) - *(G+j), 2) * pow(x0, -3));
            
        }        
    }
        
    matrix<double> inv_sigmak (T, T);
    
    identity_matrix<double> eye (T);
    
    matrix<double> refined_sigmak (T, T);
    
    refined_sigmak = sigmak+eye*0.0001;
    
    InvertMatrix(refined_sigmak, inv_sigmak);
    
    double f;
    
    matrix<double> gk (T, 1);
    
    gk = make_matrix_from_pointer(T, 1, G);
    
    matrix<double> inv_sigmak_dsigmak (T, T);
    
    inv_sigmak_dsigmak = prod(inv_sigmak, dsigmak);
    
    matrix_vector_range< matrix<double> > diag(inv_sigmak_dsigmak, range (0,T), range (0,T)); 
       
   
    matrix<double> prodct (1, 1);
    matrix<double> prodct1 (1, 2);
    matrix<double> prodct2 (2, 1);
    
    prodct1 = prod(trans(gk), inv_sigmak_dsigmak);
    
    prodct2 = prod(inv_sigmak, gk);
    
    prodct  = prod(prodct1, prodct2);
    
    f  = 0.5 * sum(diag) - 0.5 * prodct(1,1) - (S_PARA-1)/x0 + 1;
       
    return f;
}

double U_func(double x0, int T, double sigma, double S_PARA, double* G)
{
    
    matrix<double> sigmak (T, T);
    
    for (int i = 0 ;i < T; i++){
        for (int j = 0; j < T; j++){
         
            sigmak(i, j) = sigma * exp(-1 * pow(*(G+i) - *(G+j), 2) / pow(x0, 2));
            
        }        
    }
        
    matrix<double> inv_sigmak (T, T);
    
    identity_matrix<double> eye (T);
    
    matrix<double> refined_sigmak (T, T);
    
    refined_sigmak = sigmak+eye*0.0001;
    
    InvertMatrix(refined_sigmak, inv_sigmak);
    
    double f;
    
    matrix<double> gk (T, 1);
    
    gk = make_matrix_from_pointer(T, 1, G);
    
    matrix<double> prodct (1, 1);
    matrix<double> prodct1 (1, 2);
    
    prodct1 = prod(trans(gk), inv_sigmak);
    
    prodct  = prod(prodct1, gk);
    
    
    f  = 0.5 * log(determinant(sigmak) + 0.00000000001) + 0.5 * prodct(1, 1) 
            - (S_PARA - 1)*log(x0 + 0.00000000001) + x0;
        
    
    return f;
}

double hmc_s(double S, int T, double sigma, double S_PARA, double* G)
{
    double delta    = 0.3;
    int nSamples    = 50;
    int burnin      = nSamples * 0.2;
    int L           = 10;
    
    //
    typedef boost::mt19937                     ENG;    // Mersenne Twister
    typedef boost::normal_distribution<double> DIST;   // Normal Distribution
    typedef boost::variate_generator<ENG,DIST> GEN;    // Variate generator
 
    ENG  eng;
    DIST dist(0,1);
    GEN  gen(eng,dist);
    
    //
    double x[nSamples];
    double x0 = pow(S, 0.5);
    x[0] = x0;
        
    double p0, pStar, xStar, U0, UStar, K0, KStar, alpha, u;
    
    int t = 0;
    int num = 0;
    
    double x_tmp = x[t];
    
    while(num < nSamples){
       
        t       = t + 1;
        
        x0      = x_tmp;
        
        p0      = gen();
        
        pStar   = p0 - delta/2 * dU_func(x0, T, sigma, S_PARA, G);
        
        xStar   = x0 + delta*pStar;
        
        for (int jL = 0; jL < L-1; jL++){
         
            if(isnan(xStar)){
                xStar = 0.0000000001;
            }
            
            pStar = pStar - delta*dU_func(xStar, T, sigma, S_PARA, G);
            
            xStar = xStar + delta*pStar;
            
        }
        
        pStar = pStar - delta /2 *dU_func(xStar, T, sigma, S_PARA, G);
        
        if(isnan(xStar)){
             xStar = 0.0000000001;           
        }
        
        U0      = U_func(x0, T, sigma, S_PARA, G);
        UStar   = U_func(xStar, T, sigma, S_PARA, G);
        
        K0      = pow(p0, 2)/2;
        KStar   = pow(pStar, 2)/2;
        
        
        alpha   = exp( (U0 + K0) - (UStar + KStar) );
        
        if (alpha > 1)
            alpha = 1;
        
        u       = rand() /double(RAND_MAX);
    
		if (u < alpha){
			x_tmp           = xStar;	            	
		}else{
			x_tmp           = x0;            
        }	
        
        if (t > burnin){            
            x[num] = x_tmp;
            num++;
        }
        
    }
    
    //
    int selected 	= (rand() % (nSamples-1 -0+1))+ 0;
     
    return pow(x[selected], 2);
    
    
}

// s_new(k) = HMC_s(s(k), T, sigma, s_para, g)


void mexFunction(
        int nlhs,
        mxArray *plhs[], //output
        int nrhs,
        const mxArray *prhs[])  //Input
{
    /* input variables external formate */
    double S, S_PARA, SIGMA;
    
    double* G; 
               
    int T;
    
    
    /*  collect input external formate variables */
    
    S       = (double) mxGetScalar(prhs[0]);    
    T       = (int) mxGetScalar(prhs[1]);    
    SIGMA   = (double) mxGetScalar(prhs[2]);    
    S_PARA  = (double) mxGetScalar(prhs[3]); 
    G       = mxGetPr(prhs[4]);    
    
    
    /* convert variables to internal formate */
    
    /* output variables external formate */
    
    /* output variables internal formate */
    double s_new_in; 
                    
    /* run the model */
    
    s_new_in = hmc_s(S, T, SIGMA, S_PARA, G);
    
        
    /* output: convert varibale to external formate*/
    plhs[ 0 ] = mxCreateDoubleMatrix( 1, 1, mxREAL );
    
    double* S_NEW  = mxGetPr( plhs[ 0 ] );
    
    *(S_NEW)       = s_new_in;
    
}








