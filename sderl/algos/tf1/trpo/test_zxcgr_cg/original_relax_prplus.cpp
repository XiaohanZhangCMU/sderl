// Last Modified : Mon Apr  6 23:29:08 2009

//////////////////////////////////////////////////////
// Stand alone conjugate gradient relaxiation program
//
//    transplanted from LAMMPS min_cg.cpp and min_linesearch.cpp
//
//  writen by Keonwook Kang at Stanford, Mar 16 2009
//
//////////////////////////////////////////////////////

//#include "general.h"
//#include "mplib.h"
//#include "scparser.h"
#include "float.h"

#ifndef _GENERAL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define ASSERT(a) ((void)0)
#define ERROR(s) puts(s)
#define Max(a,b) ((a) > (b)? (a):(b))
#define Min(a,b) ((a) > (b)? (b):(a))

#endif

// ALPHA_MAX : max. alpha allowed to avoid long backtracks in unit of A^2/eV
//             Sould be divided by L^2 later, where L is the box length.
// ALPHA_REDUCE : reduction ratio, [0.5, 1)
// BACKTRACK_SLPE : (0, 0.5]
// EPS_ENERGY : min. normalization for energy tolerance
// IDEAL_TOL : ideal energy tolerance for backtracking
#define ALPHA_MAX  1.0
#define ALPHA_REDUCE 0.5
#define BACKTRACK_SLOPE 0.4
#define EPS_ENERGY 1.0e-8
//#define IDEAL_TOL  1.0e-10
#define IDEAL_TOL  1.0e-18

enum{MAXITER, MAXEVAL, ETOL, FTOL, DOWNHILL, ZEROALPHA, ZEROFORCE, ZEROQUAD, EMPTY};

/* Special Interface to potential_wrapper in md.h md2d.h */
class MDFrame *__conj_simframe;
//class SIHMDFrame *__conj_sihmdframe;
//class SimFrame2d *__conj_simframe2d;
//class Morse2d *__conj_morse2d;
//class SpFrame2d *__conj_spframe2d;
/* End of Special Interface */

static double *s,    //Search direction
       *rss,*rsg,    //CG Restart procedure data
       *g_old,       //Gradient at start of iteration
       *xopt,*gopt;  //Optimal position

#ifdef __cplusplus
extern "C" void CGRelax_PRplus(void (*func)(int,double*,double *, double*),
            int n,double etol,double ftol,int max_iter,int max_eval,double maxdist,
            double x[],double g[],double *f, double *buffer);
#endif

#define MP_Free(s) free(s)

void CGRelax_PRplus(void (*func)(int,double*,double *, double*),
             int n,double etol,double ftol,int max_iter,int max_eval,double maxdist,
             double x[],double g[],double *f, double *buffer)
{
/* what is default value of maxdist?
   In Lammps, dmax = 0.1 Angstrom
   In MD++, conj_dfpred = 0.001 in no unit
   what about force tolerance ftol?
   In Lammps, ftol is given in eV/Angstrom, e.g ftol = 1e-6 eV/Angstrom
   in MD++, ftol is in unit of eV.
 */
    register int i; 
    int iterc;          // Iteration count
    int ncalls;         // Call count
    int fixbox;
    int return_value, n0, n1;
    double sum, temp, de_ideal, L;
    register double beta,f_old,fmin,gdotdir;
    register double gnorm2,gnorm2_init,gnorm2_old,gnorm2_mid,alpha,alpha_max;
 
    /* initialize return_value */
    return_value = EMPTY;
    alpha = 0;

    n0 = 0; n1 = n;
    //Use the passed buffer if it's not NULL
    if(buffer==NULL)
    {
        /* allocate work space if empty */
        s=(double *)realloc(s,sizeof(double)*n*6);
        if(s==NULL){ ERROR("Out of memory in CGRelax"); abort(); }
    }
    else s=buffer;

    temp = 0;
    for(i=n0;i<n1;i++)
        temp = Max(x[i],temp);
    //INFO_Printf("Max(x[i]) = %f\n",temp);
    if (temp > 0.5)
        fixbox = 1;
    else
        fixbox = 0;

    rss=s+n;         // Not Used
    rsg=rss+n;       // Not Used
    g_old=rsg+n;     // the address at which the initial gradient of size n will be stored.
    xopt=g_old+n;    // the address at which optimal position x of size n will be stored. Not used.
    gopt=xopt+n;     // the address at which optimal gradient g of size n will be stored. Not used.

    iterc=0;
    
     /* Call a potential function for the 1st time,
       and evaluate the potential energy f and its gradient g */
    func(n,x,f,g); ncalls=1; /* func has to be sync-ed in the end */
    INFO("relax: 1st potential call finished.");

    /* Initialize the search direction, s[i].
       The initial search direction be minus the gradient vector. (Steepest descent) */
    for(i=n0;i<n1;i++) { s[i]=-g[i]; g_old[i]=g[i]; }

    /* Calculate the initial value of the squared norm of the gradient g[i]
       and store it to gnorm2_init */
    for(sum=temp=0,i=n0;i<n1;i++)
        sum+=g[i]*g[i];
    gnorm2_init = sum; 

    if (fixbox) 
    {
        L = 1.0;
        INFO("##############################################################");
        INFO("iteration     neval    energy (eV)       |gradient|^2 (eV/A)^2");
        INFO("##############################################################");
    } else {
        // Need to update for a general non-orthogonal simulation box 
        //h.set(x); //INFO("h = [ "<<h<<" ]");
        //hinv = h.inv(); hinvtran = hinv.tran(); gh = hinv*hinvtran;  
        double Lx = x[0]; double Ly = x[4]; double Lz = x[8];
        L = Max(Lx,Ly); L = Max(L,Lz);
        INFO("############################################################");
        INFO("iteration     neval    energy (eV)       |gradient|^2 (eV^2)");
        INFO("############################################################");
    }    
    INFO_Printf("%10d %10d %20.14e %20.14e\n",iterc,ncalls,*f,sum);

    /* Store the initial position and gradient */
    for(i=n0;i<n1;i++) { xopt[i]=x[i]; gopt[i]=g[i]; }
    
    gnorm2_old = gnorm2_init;
    /***********************************************************/
    // The outer loop tries different search directions, s[i]
    /***********************************************************/   
    for(iterc=1;iterc<=max_iter;iterc++)
    {
        for(gdotdir=0,i=n0;i<n1;i++)
            gdotdir+=g[i]*s[i];
//        gdotdir/=((n-9)/3);  

        /* Check if the search direction is uphill */
        if (gdotdir >= 0.0) {return_value = DOWNHILL; goto l_return; }
        
        /* Check if the search direction is valid */
        for(temp=0, i=n0; i<n1; i++)
            temp = Max(fabs(s[i]),temp); 
        if (temp == 0) {return_value = ZEROFORCE; goto l_return;}
        
        alpha_max = maxdist/temp;
        alpha = Min(ALPHA_MAX/(L*L),alpha_max); f_old=*f;
        //INFO_Printf("alpha = min(%20.14e, %20.14e)\n",ALPHA_MAX/(L*L),alpha_max);
        /***********************************************************/
        // The inner loop aims to find the optimal step length alpha
        // for a given search direction.
        /***********************************************************/
        for(;;)
        {/* Backtracking Line Search
            Jorge Nocedal & Stephen J. Wright, "Numerical Optimization"
            Ch 3. Line Search Methods

            [NOTE]
            Backtracking method does not guarantee that a function f
            minimizes at the obtained alpha. Rather, it inexactly minimizes f
            by finding an alpha at which the function f sufficiently decreases.
            This decision can be justified if it is undesirable to exactly
            minimize a function f or if we don't want to spend too much
            computation time to find an exact alpha.
         */
//            del_alpha = alpha - alpha_previous;
            for(i=n0;i<n1;i++)
                x[i]=xopt[i]+alpha*s[i];
            
            func(n,x,f,g); ncalls++;

//            for(sum=0,i=n0;i<n1;i++)
//                sum+=g[i]*g[i];      
            
            /* Print out iteration information */
//            INFO_Printf("%10d %10d %20.14e %20.14e %12.8e\n",iterc,ncalls,*f,sum,alpha*L*L);
            de_ideal = BACKTRACK_SLOPE*alpha*gdotdir;
            if (*f <= f_old + de_ideal)
            {
                fmin = *f;
                for(i=n0;i<n1;i++) { xopt[i]=x[i]; gopt[i]=g[i]; }
                break;
            }

            // reduce alpha
            alpha *= ALPHA_REDUCE;

            if (alpha <= 0.0 || de_ideal >= -IDEAL_TOL)
            {// if alpha becomes effectively zero,
//                INFO_Printf("alpha=%f, de_ideal=%12.8e, IDEAL_TOL=%12.8e\n",alpha,de_ideal,IDEAL_TOL);
                return_value = ZEROALPHA; goto l_return;
            }
                
//            /* Print out iteration information */
//            INFO_Printf("%10d %10d %20.14e %20.14e\n",iterc,ncalls,*f,sum);
        }
        /*******************************************/
        /*          END OF THE INNTER LOOP         */
        /*******************************************/

        // ENSURE THAT F, X AND G ARE OPTIMAL.
        *f=fmin;
        for(i=n0;i<n1;i++) { x[i]=xopt[i]; g[i]=gopt[i]; }

        for(gnorm2=0,i=n0;i<n1;i++)
            gnorm2+=g[i]*g[i];
        /* Print out iteration information */
        INFO_Printf("%10d %10d %20.14e %20.14e\n",iterc,ncalls,*f,gnorm2);
        
        // Function evaluation criterion 
        if (ncalls >= max_eval) { return_value = MAXEVAL; goto l_return; }

        // Energy tolerance criterion
        if (fabs(*f-f_old) < etol*0.5*(fabs(*f) + fabs(f_old) + EPS_ENERGY))
        {
            return_value = ETOL; goto l_return;
        }

        // Force tolerance criterion
        if (gnorm2 < ftol*ftol) { return_value = FTOL; goto l_return; }
        
        for(gnorm2_mid=0,i=n0;i<n1;i++)
            gnorm2_mid+=g[i]*g_old[i];

        //CALCULATE THE VALUE OF BETA THAT OCCURS IN THE NEW SEARCH DIRECTION.
        /* Polak-Ribiere plus method
           Jorge Nocedal & Stephen J. Wright, "Numerical Optimization"
           Ch 5.2 Nonlinear Conjugate Gradient Methods */
        beta=Max(0.0,(gnorm2-gnorm2_mid)/gnorm2_old);
        // If we chhose beta as
        // beta=gnorm2/gnorm2_old;
        // then it is Fletcher-Reeves method.

        // Reinitialize CG every n iterations by setting beta = 0;
        if ((iterc+1)%n==0) beta = 0.0;
        
        gnorm2_old = gnorm2;

        /* Update the search direction, s[i].
           When beta=0, the search direction is set to be negative gradient
           just as in the steepest descent method. */
        for(i=n0;i<n1;i++)
        {
            s[i]=-g[i]+beta*s[i];
            g_old[i]=g[i];
        }

        /* If new search direction s[i] is not downhill,
           set s[i] be minus the gradient vector. (Steepest descent)*/
        for(gdotdir=0,i=n0;i<n1;i++)
            gdotdir+=g[i]*s[i];
        if (gdotdir >= 0.0)
        {
            for(i=n0;i<n1;i++) s[i]=-g[i];
        }
    }
    /*******************************************/
    /*          END OF THE OUTER LOOP          */
    /*******************************************/
    
    if (iterc == max_iter)
        return_value = MAXITER;

    
l_return:
    switch (return_value)
    {
    case MAXITER:
        ERROR("CGRelax: Max iterations reached"); break;
    case MAXEVAL:
        ERROR("CGRelax: Max of calling the potential function"); break;
    case ETOL:
        INFO("CGRelax: enerigy tolerance was obtained."); break;
    case FTOL:
        INFO("CGRelax: force tolerance was obtained."); break;
    case DOWNHILL:
        ERROR("CGRelax: search direction is not downhill."); break;
    case ZEROFORCE:
        ERROR("CGRelax: forces are zero."); break;
    case ZEROALPHA:
        //ERROR("CGRelax: linesearch alpha is zero."); 
        INFO_Printf("CGRelax: linesearch alpha is zero: alpha=%20.14e\n",alpha); 
        break;
    default:
        INFO("CGRelax: Ends");
    }
    if(buffer==NULL) MP_Free(s);
}

