import numpy as np
import tensorflow as tf
from termcolor import colored, cprint
from utils import *

class QuadMesher:
    def quadMesher(self, nEx, nEy, Lx, Ly):
        nnelem, nNodes, nElements = 4, (nEx+1)*(nEy+1), nEx*nEy

        rn0 = np.zeros((nNodes,2)) # Nodes ...
        for j in range(nEy+1):
            for i in range(nEx+1):
             rn0[i+j*(nEx+1),:] = [(i/nEx-0.5)*Lx,(j/nEy-0.5)*Ly];

        elements = [[ 0 for _ in range(nnelem)] for _ in range(nElements)] # Elements ...
        for j in range(nEy):
            for i in range(nEx):
                elements[i+j*nEx] =[ a+i+j*(nEx+1) for a in [0, 1, nEx+2, nEx+1]];

        ind_LE = np.arange(0,nNodes-nEx,nEx+1) # B.C. ...
        ind_BE = np.arange(0,nEx+1)
        ind_RE = np.arange(nEx,nNodes,nEx+1)
        ind_TE = np.arange(nNodes-nEx-1,nNodes)

        return rn0, elements, [ind_LE,ind_BE,ind_RE,ind_TE]

    def plot(self, X, x, conn, plotseq = [0,1,2,3,0]):
        import matplotlib.pyplot as plt
        dim, nnelem = X.shape[1], len(conn[0])
        for e in conn: plt.plot(X[e][plotseq,0],X[e][plotseq,1],'k:',x[e][plotseq,0],x[e][plotseq,1],'b-');
        plt.show()

class Quad4:  # Quad, Linear Type Elements
    def sample_element(self,e1,e2, Xe, dim=2, elem_type='Q4'):
        if dim == 2 and elem_type == 'Q4':
            N1 = 1./4*(1-e1)*(1-e2); N2 = 1./4*(1+e1)*(1-e2);
            N3 = 1./4*(1+e1)*(1+e2); N4 = 1./4*(1-e1)*(1+e2);
            N1xi = -1./4*(1-e2); N1eta = -1./4*(1-e1);
            N2xi =  1./4*(1-e2); N2eta = -1./4*(1+e1);
            N3xi =  1./4*(1+e2); N3eta =  1./4*(1+e1);
            N4xi = -1./4*(1+e2); N4eta =  1./4*(1-e1);
            N =np.array([[N1,0,N2,0,N3,0,N4,0],[0,N1,0,N2,0,N3,0,N4]]);
            dN_parent=np.array([[N1xi,N2xi,N3xi,N4xi],[N1eta,N2eta,N3eta,N4eta]])
            J = tf.tensordot(dN_parent, Xe, axes=1)
            je = tf.linalg.det(J);
            JTinv = tf.linalg.inv(tf.transpose(J));
            Nd = tf.transpose(tf.tensordot(dN_parent.T, JTinv, axes=1))
        return N, Nd, je

    def gaussian(self, nig1d):
        return { 1: np.array([[0],[2]]), 2: np.stack((np.array([-1,1])*np.sqrt(1/3), np.array([1,1]))),
                 3: np.stack((np.array([-1,0,1])*np.sqrt(3/5), np.array([5,8,5])/9.0))}[nig1d]

@timeit
def fem(X, disp, conn,  mask, fe=Quad4(), nig1d=2):
    dim, nnelem = X.shape[1], len(conn[0])

    # Reference configuration as input to computation graph
    X_ph = tf.placeholder(shape=X.shape, dtype=np.float64)
    all_phs = [X_ph]

    # Potential energy as output to computation graph then minimize
    pe = tf.Variable(0,dtype=np.float64)

    # Define constants
    V0, MU = 1, 1 # ?????? V0 should not be 1 but computed on the fly
    Q = fe.gaussian(nig1d)

    # Initialize current configuration and apply b.c. with mask
    rn0 = np.zeros(X.shape, dtype=np.float64) + X + disp
    x_ = tf.Variable(rn0)
    mask_h = tf.abs(mask-1)
    x = tf.stop_gradient(mask_h*x_)+ mask*x_  # Don`t omit x_. Otherwise RunErr!

    for e in conn: # Energy integral
        xe, Xe = tf.gather(x,e), tf.gather(X_ph,e)
        for I in range(Q.shape[1]): # Gaussian integral
            for J in range(Q.shape[1]):
                N, Nd, je = fe.sample_element(Q[0,I],Q[0,J],Xe=Xe)
                F = tf.eye(dim,dtype=np.float64) + tf.tensordot(Nd, xe-Xe, axes=1)
                Ft = tf.transpose(F)
                B = tf.tensordot(F,Ft,axes=1)
                dW = (MU/2.0*(tf.linalg.trace(B)+1.0/tf.linalg.det(B)-3)) * V0
                pe+=0.25*dW*Q[1,I]*Q[1,J] # tf.assign_add not work ???

    # Optimization setup
    lr = 1e-3  # step size
    tol = 1e-9 # convergence tol
    MaxGDsteps = 1<<20

    optimizer = tf.train.AdamOptimizer(lr)
    #grads_and_vars = optimizer.compute_gradients(tf.reduce_max(pe*pe))
    grads_and_vars = optimizer.compute_gradients(pe*pe)
    op = optimizer.apply_gradients(grads_and_vars,global_step=tf.Variable(0, trainable=False))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    inputs = {k:v for k,v in zip(all_phs, [X])}

    pe_old = 1e5
    for step in range(MaxGDsteps): # Minimize pe^2 til |pe-pe_old|<tol
        sess.run(op, feed_dict=inputs)
        result = sess.run([x, pe,grads_and_vars], feed_dict={X_ph:X})
        nrm = np.max(np.abs(np.array(result[2][0][0]))) #??????
        err = np.abs(result[1] - pe_old)
        pe_old = result[1] # Record last step

        if step % 100 == 0:
            print("Step = {:5d}; E={: 5.10E}; |G|={: 5.10E}; Err={: 5.10E};".format(step, result[1], nrm, err))
            np.save('x'+str(step)+'.npy',result[0])
        if err < 1e-9:
            cprint("Converged in {:d} steps!!!".format(step), 'green', 'on_red')
            print("E={: 5.10E}. |G|={: 5.10E}; Err={: 5.10E};".format(result[1], nrm, err))
            return result

    cprint("Failed to converged in {:d} steps!!!".format(step), 'green', 'on_red')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='fem')
    args = parser.parse_args()

    mesher = QuadMesher()
    X, conn, bids = mesher.quadMesher(nEx=10, nEy=4, Lx=0.5, Ly=0.1)

    # Specify displacement b.c. using mask
    u, m = np.zeros(X.shape), np.ones(X.shape)
    u[bids[2],0] = u[bids[2],0] + 0.05
    m[bids[2],:] = 0
    m[bids[0],:] = 0

    result = fem(X, fe=Quad4(), disp=u, conn=conn, mask=m, nig1d=2) # PDE solve

    mesher.plot(X, result[0],conn) # Plot final configuration
