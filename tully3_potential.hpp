#ifndef _TULLY3_POTENTIAL_HPP
#define _TULLY3_POTENTIAL_HPP


#include <cstdlib>
#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include "misc/crasher.hpp"
#include "misc/randomer.hpp"
#include "misc/vector.hpp"
#include "misc/matrixop.hpp"
#include "misc/ioer.hpp"


/*
 ****************************************************************
 * TULLY 3 POTENTIAL 
 *
 * H00 = A
 * H11 = -A
 * H12 = B exp(Cx), x < 0
 *       B (2 - exp(-Cx)), x > 0
 * 
 *
 ****************************************************************
 *  API INFO
 ****************************************************************
 *
 *  - void output_potenial_params() 
 *      
 *      THIS FUNCTION OUTPUTS POTENTIAL PARAMETRS
 *
 *  - vector< complex<double> > cal_H(const double x)
 *
 *      THIS FUNCTION CALCULATES HAMILTONIAN FOR A GIVEN POSITION
 *
 *  - vector< complex<double> > cal_nablaH(const double x)
 *
 *      THIS FUNCTION CALCULATES HAMILTONIAN GRADIANT FOR A GIVEN POSITION
 *
 *  - void cal_info_nume(const double x, vector<double>& eva, vector< complex<double> >& dc, vector< complex<double> >& F, vector< complex<double> >& lastevt)
 *
 *      THIS FUNCTION CALCULATES EIGENVALUES, EIGENVECTORS, FORCE MATRIX AND DERIVATIVE COUPLINGS FOR A GIVEN POSITION
 *  
 ****************************************************************
 */

namespace {
    using std::vector;
    using std::complex;

    double param_A = 6e-4;
    double param_B = 0.1;
    double param_C = 0.9;

    void output_potential_params() {
        ioer::info("# Tully3 Parameters: ", 
                    " A     = ", param_A,
                    " B     = ", param_B,
                    " C     = ", param_C,
                    ""
                    );
    }

    void set_potenial_params(const std::vector<double>& params) {
        misc::crasher::confirm(params.size() >= 3, 
                "set_potenial_params: potential paramter vector size must be >= 3");
        param_A = params[0];
        param_B = params[1];
        param_C = params[2];
    }


    vector< complex<double> > cal_H(const double x) {
        const double A = param_A;
        const double B = param_B;
        const double C = param_C;

        vector< complex<double> > H(4, 0.0);
        H[0+0*2] = A;
        H[1+1*2] = -A;
        if (x < 0.0) {
            H[0+1*2] = B * exp(C * x);
        }
        else {
            H[0+1*2] = B * (2.0 - exp(-C * x));
        }
        H[1+0*2] = conj(H[0+1*2]);

        return H;
    }

    vector< complex<double> > cal_nablaH(const double x) {
        const double A = param_A;
        const double B = param_B;
        const double C = param_C;

        vector< complex<double> > nablaH(4);
        vector< complex<double> >& Hx = nablaH;

        Hx.resize(4);
        Hx[0+0*2] = 0.0;
        Hx[1+1*2] = 0.0;
        if (x < 0.0) {
            Hx[0+1*2] = B * C * exp(C * x);
        }
        else {
            Hx[0+1*2] = B * C * exp(-C * x);
        }
        Hx[1+0*2] = conj(Hx[0+1*2]);

        return nablaH;
    }

    void cal_info_nume(const double x,
            vector<double>& eva, 
            vector< complex<double> >& dc,
            vector< complex<double> >& F,
            vector< complex<double> >& lastevt)
    {
        const int edim = 2;

        vector< complex<double> > evt;
        matrixop::hdiag(cal_H(x), eva, evt);
        // correct phase
        if (not lastevt.empty()) {
            auto tmp = matrixop::matCmat(lastevt, evt, edim);
            for (int j = 0; j < edim; ++j) {
                complex<double> eip = tmp[j+j*edim] / abs(tmp[j+j*edim]);
                for (int k = 0; k < edim; ++k) {
                    evt[k+j*edim] /= eip;
                }
            }
        }

        // F, dc
        dc = matrixop::matCmatmat(evt, cal_nablaH(x), evt, edim, edim);
        F.assign(edim * edim, 0.0);

        for (int j = 0; j < edim; ++j) {
            for (int k = 0; k < edim; ++k) {
                F[j+k*edim] = -dc[j+k*edim];
                if (j == k) {
                    dc[j+k*edim] = 0.0;
                }
                else {
                    dc[j+k*edim] /= (eva[k] - eva[j]);
                }
            }
        }

        // save evt to lastevt
        lastevt = std::move(evt);
    }
};

#endif 
