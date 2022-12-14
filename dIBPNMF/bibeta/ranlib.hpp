// #ifndef RANLIB_HPP
// #define RANLIB_HPP


char ch_cap ( char ch );
float genbet ( float aa, float bb );
float genchi ( float df );
float genexp ( float av );
float genf ( float dfn, float dfd );
float gengam ( float a, float r );
float *genmn ( float parm[] );
int *genmul ( int n, float p[], int ncat );
int genmulx ( int n, float p[], int ncat );
float gennch ( float df, float xnonc );
float gennf ( float dfn, float dfd, float xnonc );
float gennor ( float av, float sd );
void genprm ( int iarray[], int n );
float genunf ( float low, float high );
int i4_max ( int i1, int i2 );
int i4_min ( int i1, int i2 );
int ignbin ( int n, float pp );
int ignnbn ( int n, float p );
int ignpoi ( float mu );
int ignuin ( int low, int high );
int lennob ( char *s );
void phrtsd ( char *phrase, int &seed1, int &seed2 );
void prcomp ( int maxobs, int p, float mean[], float xcovar[], float answer[] );
float r4_exp ( float x );
float r4_exponential_sample ( float lambda );
float r4_max ( float x, float y );
float r4_min ( float x, float y );
float r4vec_covar ( int n, float x[], float y[] );
double r8_exponential_sample ( double lambda );
double r8_max ( double x, double y );
double r8_min ( double x, double y );
double r8vec_covar ( int n, double x[], double y[] );
int s_eqi ( char *s1, char *s2 );
float sdot ( int n, float dx[], int incx, float dy[], int incy );
float *setcov ( int p, float var[], float corr );
void setgmn ( float meanv[], float covm[], int p, float parm[] );
float sexpo ( );
float sgamma ( float a );
float snorm ( );
int spofa ( float a[], int lda, int n );
void stats ( float x[], int n, float &av, float &var, float &xmin, float &xmax );
void trstat ( string pdf, float parin[], float &av, float &var );

// #endif