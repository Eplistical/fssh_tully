import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import time 
from datetime import datetime

'''
Absorbing Boundary Condition SE 
'''


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


def exact_1d(xI, kxI, init_s, Nstep):
    '''1-D exact quantum dynamics
    '''

    enable_plot = True

    c0 = np.sqrt(1 - init_s)
    c1 = np.sqrt(init_s)

    N = 2
    L = 16
    M = 256

    mass = 2000
    sigmax = 1.0

    dt = 0.2
    tgraph = 1000

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
    evts = np.zeros([M,N,N], dtype=np.float)
    for i, x in enumerate(x0):
        H = cal_H(x)
        VU[i,:,:] = la.expm(-1j * dt / 2 * H)
        Hs[i,:,:] = H

        eva, evt = la.eigh(H)
        evas[i,:,:] = np.diag(eva)

        if i > 0:
            phase0 = evts[i-1,0,0] * evt[0,0] + evts[i-1,1,0] * evt[1,0]
            phase1 = evts[i-1,0,1] * evt[0,1] + evts[i-1,1,1] * evt[1,1]
            if phase0 < 0.0:
                evt[:,0] *= -1
            if phase1 < 0.0:
                evt[:,1] *= -1

        evts[i,:,:] = evt

    # Initial Wavefunction -- Gaussian wavepacket on adiabats
    psiad0 = np.zeros([M,N], dtype=np.complex)
    psiad0[:,0] = c0 * np.exp(1j*(kxI*x0)) * np.exp(-(x0-xI)**2/sigmax**2);
    psiad0[:,1] = c1 * np.exp(1j*(kxI*x0)) * np.exp(-(x0-xI)**2/sigmax**2);
    psiad0 /= np.sqrt(np.sum(np.abs(psiad0)**2))

    # convert to diabats
    psi0 = np.zeros([M,N], dtype=np.complex)
    psi0[:,0] = evts[:,0,0] * psiad0[:,0] + evts[:,0,1] * psiad0[:,1]
    psi0[:,1] = evts[:,1,0] * psiad0[:,0] + evts[:,1,1] * psiad0[:,1]

    # Propagate WF
    psi_k = np.zeros([M,N], dtype=np.complex)
    psi = psi0

    # ABC
    x_to_left = np.abs(x0 - x0[0])
    x_to_right = np.abs(x0 - x0[-1])
    x_to_bound = np.minimum(x_to_left, x_to_right)
    U0 = 1.0
    alpha = 10
    gamma = U0 / np.cosh(alpha * x_to_bound)

    #plt.plot(x0, gamma); plt.show(); dslfkj;

    accu_trans = np.zeros(2)
    accu_refl = np.zeros(2)

    print('%20s%20s%20s%20s%20s%20s' % ('t','T0', 'T1', 'R0', 'R1', 'Nrest'))
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

        # adiab
        psiad = np.zeros([M,N], dtype=np.complex)
        psiad[:,0] = np.conj(evts[:,0,0]) * psi[:,0] + np.conj(evts[:,1,0]) * psi[:,1];
        psiad[:,1] = np.conj(evts[:,0,1]) * psi[:,0] + np.conj(evts[:,1,1]) * psi[:,1];

        # ABC
        for k in range(2):
            tmp = np.abs(psiad[:,k])**2 * (2 * gamma * dt - gamma**2 * dt**2)
            accu_trans[k] += np.sum(tmp[int(M/2):])
            accu_refl[k] += np.sum(tmp[:int(M/2)])

        psiad[:,0] *= (1 - gamma*dt)
        psiad[:,1] *= (1 - gamma*dt)

        # back to diab
        psi[:,0] = evts[:,0,0] * psiad[:,0] + evts[:,0,1] * psiad[:,1];
        psi[:,1] = evts[:,1,0] * psiad[:,0] + evts[:,1,1] * psiad[:,1];


        # analysis & report
        if t % tgraph == 0:
            #psiad = np.zeros([M,N], dtype=np.complex)
            #psiad[:,0] = np.conj(evts[:,0,0]) * psi[:,0] + np.conj(evts[:,1,0]) * psi[:,1];
            #psiad[:,1] = np.conj(evts[:,0,1]) * psi[:,0] + np.conj(evts[:,1,1]) * psi[:,1];

            psiad_k = np.zeros([M,N], dtype=np.complex)
            psiad_k[:,0] = np.fft.fft(psiad[:,0]);
            psiad_k[:,1] = np.fft.fft(psiad[:,1]);
            psiad_k = psiad_k / np.sqrt(np.sum(np.abs(psiad_k)**2)) * np.sqrt(np.sum(np.abs(psiad)**2))

            T0 = accu_trans[0] + np.sum(np.abs(psiad_k[:int(M/2),0])**2)
            T1 = accu_trans[1] + np.sum(np.abs(psiad_k[:int(M/2),1])**2)
            R0 = accu_refl[0] + np.sum(np.abs(psiad_k[int(M/2):,0])**2)
            R1 = accu_refl[1] + np.sum(np.abs(psiad_k[int(M/2):,1])**2)
            Nrest = 1.0 - np.sum(accu_trans + accu_refl)
            print('%20.10f%20.10f%20.10f%20.10f%20.10f%20.4e' % (t * dt, T0, T1, R0, R1, Nrest))

            # plot
            if enable_plot:
                ampl = 1.0
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(x0, evas[:,0,0], '-r')
                ax.plot(x0, evas[:,1,1], '-g')
                ax.fill_between(x0, evas[:,0,0],  evas[:,0,0] + ampl*np.abs(psiad[:,0])**2, color='#eec881')
                ax.fill_between(x0, evas[:,1,1],  evas[:,1,1] + ampl*np.abs(psiad[:,1])**2, color='#81d0ee')

                ax.set_xlim([-L/2, L/2])
                ax.set_xlabel('x')
                ax.set_ylim([-0.05,0.05])
                #ax.set_title('Tully #1')
                fig.savefig('%04d.png' % (int(t / tgraph)))
                plt.close()

                if t == 0:
                    fig = plt.figure()
                    ax = fig.gca()
                    ax.plot(x0, evas[:,0,0], '-r')
                    ax.plot(x0, evas[:,1,1], '-g')

                    ax.set_xlim([-L/2, L/2])
                    ax.set_xlabel('x')
                    ax.set_ylim([-0.015,0.025])
                    ax.set_ylim([-0.05,0.05])
                    fig.savefig('surf.png')
                    plt.close()

            # check end
            if Nrest < 1e-12:
                print('check end ', Nrest)
                return


def run():
    '''Main program entrance
    '''
    exact_1d(-4, 8.5, 0, 2000000)


if __name__ == '__main__':
    t0 = time.time()
    run()
    t1 = time.time()
    print('Total Time Elapsed = ', t1 - t0)
