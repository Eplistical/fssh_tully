import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import time 
from datetime import datetime


def cal_H(x):
    '''calc Hamiltonian
    '''

    # Tully 1
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    H = np.zeros([2,2])
    if x > 0.0:
        H[0,0] = A * (1 - np.exp(-B * x));
    else:
        H[0,0] = -A * (1 - np.exp(B * x));

    H[1,1] = -H[0,0]
    H[0,1] = C * np.exp(-D * x * x)
    H[1,0] = H[0,1]

    return H



def exact_1d(xI, kxI, init_s):
    '''1-D exact quantum dynamics
    '''

    enable_plot = True

    c0 = np.sqrt(1 - init_s)
    c1 = np.sqrt(init_s)

    N = 2
    L = 20
    M = 256

    mass = 2000
    sigmax = 1.0

    dt = 0.1
    Nstep = 15000
    tgraph = 100

    # grids
    x0, dx = np.linspace(-L/2, L/2, M, retstep=True)
    dkx = 2 * np.pi / M / dx;
    kx0 = np.arange(int(-M/2), int(M/2)) * dkx;

    # construct TU on k grid
    T = kx0**2 / 2 / mass;
    TU = np.exp(-1j * dt * T);
    TU = np.fft.fftshift(TU);

    # construct VU
    VU = np.zeros([M,N,N], dtype=np.complex)
    Hs = np.zeros([M,N,N], dtype=np.float)
    evas = np.zeros([M,N,N], dtype=np.float)
    for i, x in enumerate(x0):
        H = cal_H(x)
        VU[i,:,:] = la.expm(-1j * dt / 2 * H)
        Hs[i,:,:] = H

        eva, evt = la.eigh(H)
        evas[i,:,:] = np.diag(eva)

    # Initial Wavefunction -- Gaussian wavepacket
    psi0 = np.zeros([M,N], dtype=np.complex)
    psi0[:,0] = c0 * np.exp(1j*(kxI*x0)) * np.exp(-(x0-xI)**2/sigmax**2);
    psi0[:,1] = c1 * np.exp(1j*(kxI*x0)) * np.exp(-(x0-xI)**2/sigmax**2);
    psi0 /= np.sqrt(np.sum(np.abs(psi0)**2))

    # Propagate WF
    psi_k = np.zeros([M,N], dtype=np.complex)
    psi = psi0

    for t in range(Nstep):
        # exp(-iVdt/2) * |Psi> in diab repr
        psi_k[:,0] = VU[:,0,0]*psi[:,0] + VU[:,0,1]*psi[:,1];
        psi_k[:,1] = VU[:,1,0]*psi[:,0] + VU[:,1,1]*psi[:,1];
        # exp(-iTdt) * psi
        psi_k[:,0] = TU * np.fft.fft(psi_k[:,0]);
        psi_k[:,1] = TU * np.fft.fft(psi_k[:,1]);
        # exp(-iVdt/2) * psi
        psi_k[:,0] = np.fft.ifft(psi_k[:,0]);
        psi_k[:,1] = np.fft.ifft(psi_k[:,1]);
        psi[:,0] = VU[:,0,0]*psi_k[:,0] + VU[:,0,1]*psi_k[:,1];
        psi[:,1] = VU[:,1,0]*psi_k[:,0] + VU[:,1,1]*psi_k[:,1];

        # analysis & report
        if t % tgraph == 0:

            # plot
            if enable_plot:
                print(t)
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(x0, evas[:,0,0], '-r')
                ax.plot(x0, evas[:,1,1], '-g')
                ax.fill_between(x0, evas[:,0,0],  evas[:,0,0] + 0.2*np.abs(psi[:,0])**2, color='#eec881')
                ax.fill_between(x0, evas[:,1,1],  evas[:,1,1] + 0.2*np.abs(psi[:,1])**2, color='#81d0ee')

                ax.set_xlim([-L/2, L/2])
                ax.set_ylim([-0.015,0.025])
                fig.savefig('%04d.png' % (int(t / tgraph)))
                plt.close()





def run():
    '''Main program entrance
    '''
    exact_1d(-8, 20, 1)


if __name__ == '__main__':
    t0 = time.time()
    run()
    t1 = time.time()
    print('Total Time Elapsed = ', t1 - t0)
