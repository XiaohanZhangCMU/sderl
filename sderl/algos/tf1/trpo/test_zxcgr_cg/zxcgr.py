import numpy as np

MAXLIN =5 #Maximum iterations within a line search
MXFCON =2 #Maximum iterations before F can be decreased
MAXREPEAT =10 #Maximum iterations when energy and gradient are identical

def conjugate_gradient(f_ax, b_vec, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    conjugate gradient calculation (Ax = b), bases on
    https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel p 312

    :param f_ax: (function) The function describing the Matrix A dot the vector x
                 (x being the input parameter of the function)
    :param b_vec: (numpy float) vector b, where Ax = b
    :param cg_iters: (int) the maximum number of iterations for converging
    :param callback: (function) callback the values of x while converging
    :param verbose: (bool) print extra information
    :param residual_tol: (float) the break point if the residual is below this value
    :return: (numpy float) vector x, where Ax = b
    """
    first_basis_vect = b_vec.copy()  # the first basis vector
    residual = b_vec.copy()  # the residual
    x_var = np.zeros_like(b_vec)  # vector x, where Ax = b
    residual_dot_residual = residual.dot(residual)  # L2 norm of the residual

    fmt_str = "%10i %10.3g %10.3g"
    title_str = "%10s %10s %10s"
    if verbose:
        print(title_str % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x_var)
        if verbose:
            print(fmt_str % (i, residual_dot_residual, np.linalg.norm(x_var)))
        z_var = f_ax(first_basis_vect)
        v_var = residual_dot_residual / first_basis_vect.dot(z_var)
        x_var += v_var * first_basis_vect
        residual -= v_var * z_var
        new_residual_dot_residual = residual.dot(residual)
        mu_val = new_residual_dot_residual / residual_dot_residual
        first_basis_vect = residual + mu_val * first_basis_vect

        residual_dot_residual = new_residual_dot_residual
        if residual_dot_residual < residual_tol:
            break

    if callback is not None:
        callback(x_var)
    if verbose:
        print(fmt_str % (i + 1, residual_dot_residual, np.linalg.norm(x_var)))
    return x_var

def conjugate_gradient_zxcgr(f_ax, b_vec, maxfn=20, dfpred=0.02, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    nrepeat = 0
    n = len(b_vec)
    n0=0
    n1=n

    s = np.zeros(n)
    rss = np.zeros(n)
    rsg = np.zeros(n)
    ginit = np.zeros(n)
    xopt = np.zeros(n)
    gopt = np.zeros(n)

    ddspln=gamden=0 #The line is purely to avoid warning of var unused

    iterc=iterfm=iterrs=0

    x = np.zeros(b_vec.shape) # b_vec.copy()  # The first basis vector

    g = f_ax(x)-b_vec
    f = x.dot(g)

    ncalls=1

    s -= g

    summ = np.sum(g**2)
    print("       ############################################################")
    print("       iteration     neval    energy (eV)       |gradient|^2 (eV^2)")
    print("       ############################################################")
    print("{:10d} {:10d} {:20.14e} {:20.14e}\n".format(iterc,ncalls,f,summ))

    f_old = f
    sum_old = summ

    if summ<=residual_tol:
        return x, f, g

    gnew=-summ
    fmin=f
    gsqrd=summ
    nfopt=ncalls
    xopt = x
    gopt = g
    dfpr=dfpred
    stmin=dfpred/gsqrd

    while True:
        iterc += 1
        finit = f
        ginit = g

        gfirst = s.dot(g)
        if gfirst >=0:
            print("CGRelax: Search direction is uphill.")
            return x, f, g

        gmin=gfirst
        sbound=-1
        nfbeg=ncalls
        iretry=-1
        stepch=np.fmin(stmin, np.fabs(dfpr/gfirst))
        stmin=0


        while True:
            step=stmin+stepch
            x = xopt + stepch * s
            temp=0;
            temp = np.fabs(x-xopt).max()
            if temp <= 0:
                if ncalls > nfbeg+1 or np.fabs(gmin/gfirst) > 0.2:
                    print("Error: CGRelax: Line search aborted, "
                          "possible error in gradient.")
                    return x, f, g
                else:
                    break
            ncalls += 1

            g = f_ax(x)-b_vec
            f = x.dot(g)
            gnew = s.dot(g)
            summ = g.dot(g)

            if f < fmin or (f==fmin and gnew/gmin>=-1):
                fmin=f
                gsqrd=summ
                nfopt=ncalls
                xopt = x
                gopt = g

            #Print out iteration information
            print("{:10d} {:10d} {:20.14e} {:20.14e}\n".format(iterc,ncalls,f,summ))

            #Getting the code out of "infinite" loop
            if f_old==f and sum_old==sum:
                nrepeat +=1
            else:
                nrepeat = 0
                f_old = f
                sum_old = summ

            if nrepeat>=MAXREPEAT:
                print("Error: CGRelax: getting stuck, stop...")
                return x, f, g

            if f<=fmin and summ <= residual_tol:
                return x, f, g # Successful return

            if ncalls>=maxfn:
                print("Error: CGRelax: Too many iterations, stop...");
                return x, f, g

            temp=(f+f-fmin-fmin)/stepch-gnew-gmin
            ddspln=(gnew-gmin)/stepch
            if ncalls > nfopt:
                sbound=step
            else:
                if gmin*gnew <= 0:
                    sbound=stmin
                stmin=step
                gmin=gnew
                stepch=-stepch

            if(f!=fmin):
                ddspln+=(temp+temp)/stepch;

            if gmin==0:
                break
            if ncalls >= nfbeg+1:
                if np.fabs(gmin/gfirst) <=0.2:
                    break

            if ncalls >= nfopt+MAXLIN:
                print("Error: CGRelax: Line search aborted, "
                      "possible error in gradient.");
                return x, f, g

            stepch=0.5*(sbound-stmin)
            if sbound < -0.5:
                stepch=9*stmin
            gspln=gmin+stepch*ddspln
            if gmin*gspln<0:
                stepch*=gmin/(gmin-gspln)


        #ENSURE THAT F, X AND G ARE OPTIMAL.
        if ncalls!=nfopt:
            f=fmin
            x = xopt
            g = gopt

        summ = g.dot(ginit)
        beta=(gsqrd-summ)/(gmin-gfirst);
        if np.fabs(beta*gmin) > 0.2*gsqrd:
            iretry += 1
            if iretry<=0:
                if ncalls >= nfopt+MAXLIN:
                    print("Error: CGRelax: Line search aborted, "
                          "possible error in gradient.");
                    return x, f, g

                stepch=0.5*(sbound-stmin)
                if sbound < -0.5:
                    stepch=9*stmin
                gspln=gmin+stepch*ddspln
                if gmin*gspln<0:
                    stepch*=gmin/(gmin-gspln)

                print('Error: I am not sure how to jump back to restart')
                return x, f, g



        if f<finit:
            iterfm=iterc
        elif iterc >= iterfm+MXFCON:
            print("Error: CGRelax: Cannot reduce value of F, aborting...")
            return x, f, g

        dfpr=stmin*gfirst
        if iretry>0: # Restart since we need to retry
            s -= g
            iterrs=0
            continue

        if iterrs!=0 and (iterc-iterrs<n) and np.fabs(summ)<0.2*gsqrd:
            gama = g.dot(rsg)
            summ = g.dot(rss)
            gama/=gamden
            if np.fabs(beta*gmin+gama*summ) < 0.2*gsqrd:
                s = -g + beta * s + gama * rss
                continue;

        #APPLY THE RESTART PROCEDURE.
        gamden=gmin-gfirst

        rss = s
        rsg = g - ginit
        s = -g + beta*s
        iterrs=iterc

    return x, f,g

if __name__ == '__main__':

    #def f_ax(x):
    #    return np.array([[1.,2.,3.], [2.,5.,0], [3.,0,9.]]).dot(x)
    def f_ax(x):
        return np.array([[1.,0.,0.], [0.,1.,0], [0.,0,1.]]).dot(x)

    b_vec = np.array([1.,2.,1.])
    x, f, g = conjugate_gradient_zxcgr(f_ax, b_vec, cg_iters=10)
    print('x = ')
    print(x)
    print('f = ')
    print(f)
    print('g = ')
    print(g)


    x = conjugate_gradient(f_ax, b_vec, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10)
    print('cg rl baselines')
    print(x)



