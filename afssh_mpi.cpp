#include <cstdlib>
#include <cmath>
#include <complex>
#include <algorithm>
#include <string>
#include "misc/fmtstring.hpp"
#include "misc/ioer.hpp"
#include "misc/crasher.hpp"
#include "misc/randomer.hpp"
#include "misc/vector.hpp"
#include "misc/timer.hpp"
#include "misc/matrixop.hpp"
#include "misc/MPIer.hpp"
#include "boost/numeric/odeint.hpp"
#include "boost/math/special_functions/erf.hpp"
#include "boost/program_options.hpp"
#include "tully3_potential.hpp"
//#include "tully1_potential.hpp"


// -- GLOBAL -- //


using namespace std;
namespace po = boost::program_options;
using state_t = vector< complex<double> >;

enum {
    HOP_UP,
    HOP_DN,
    HOP_RJ,
    HOP_FR
};

// const paramters
const complex<double> zI(0.0, 1.0);
const int ndim = 1;
const int edim = 2;

const double mass = 2000.0;

const double sigma_x = 0.0;
const double sigma_p = 0.0;

// tunable parameters
double init_x = -12.0;
double init_p = 20.0;
double init_s = 0.0;

double dt = 1.0;
int Nstep = 10000;
int output_step = 100;
int Ntraj = 5000;
bool enable_hop = true;
bool enable_deco = true;
bool enable_prev = true;
int seed = 0;

// cache
vector<double> eva(edim);
vector< complex<double> > lastevt;
vector< complex<double> > F(edim * edim);
vector< complex<double> > dc(edim * edim);


// -- FUNCTIONS -- //


bool argparse(int argc, char** argv) 
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("init_x", po::value<double>(&init_x), "init x")
        ("init_p", po::value<double>(&init_p), "potential para init_p")
        ("init_s", po::value<double>(&init_s), "init surface")
        ("Ntraj", po::value<int>(&Ntraj), "# traj")
        ("Nstep", po::value<int>(&Nstep), "# step")
        ("output_step", po::value<int>(&output_step), "# step for output")
        ("dt", po::value<double>(&dt), "single time step")
        ("enable_hop", po::value<bool>(&enable_hop), "enable hopping")
        ("enable_deco", po::value<bool>(&enable_deco), "enable decoherence")
        ("enable_prev", po::value<bool>(&enable_prev), "enable momentum reversal")
        ("seed", po::value<int>(&seed), "random seed. Default 0")
        ;
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return false;
    }
    return true;
}

void init_state(state_t& state) {
    // state = r(1), p(1), c(2), s(1), rmom(4), pmom(4)
    state.resize(13, matrixop::ZEROZ);

    state[0].real(randomer::normal(init_x, sigma_x)); 
    state[1].real(randomer::normal(init_p, sigma_p)); 
    state[2].real(sqrt(1.0 - init_s));
    state[3].real(sqrt(init_s));
    state[4].real((randomer::rand() < init_s) ? 1.0 : 0.0);
}

bool check_end(const state_t& state) {
    /*
    double x = state[0].real();
    double p = state[1].real();
    if (x < -15.0 and p < 0.0) {
        return true;
    }
    if (x > 15.0 and p > 0.0) {
        return true;
    }
    */
    return false;
}

void integrator(state_t& state, const double dt) {
    // extract info
    // state = r(1), p(1), c(2), s(1), rmom(4), pmom(4)
    double x = state[0].real();
    double p = state[1].real();
    vector< complex<double> > c(state.begin() + 2, state.begin() + 4);
    int s = static_cast<int>(state[4].real());
    vector< complex<double> > rmom(state.begin() + 5, state.begin() + 9);
    vector< complex<double> > pmom(state.begin() + 9, state.begin() + 13);

    // ele part, RK4

    double v = p / mass;

    vector< complex<double> > k1, k2, k3, k4; // c

    k1 = dt * (-zI * c * eva - v * matrixop::matvec(dc, c));
    k2 = dt * (-zI * (c + 0.5 * k1) * eva - v * matrixop::matvec(dc, c + 0.5 * k1));
    k3 = dt * (-zI * (c + 0.5 * k2) * eva - v * matrixop::matvec(dc, c + 0.5 * k2));
    k4 = dt * (-zI * (c + k3) * eva - v * matrixop::matvec(dc, c + k3));

    c += k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0;
    copy(c.begin(), c.end(), state.begin() + 2);

    // moments part, RK4
    if (enable_deco) {
        vector< complex<double> > l1, l2, l3, l4; // rmom
        vector< complex<double> > m1, m2, m3, m4; // pmom
        vector< complex<double> > TR, TP, dF, sigma, dFsigma;

        vector< complex<double> > Emat(edim * edim);
        for (int i(0); i < edim; ++i) {
            Emat[i+i*edim] = eva[i];
        }

        dF = F - F[s+s*edim] * matrixop::eye(edim);
        sigma.resize(edim * edim);
        for (int j(0); j < edim; ++j) {
            for (int k(0); k < edim; ++k) {
                sigma[j+k*edim] = c[j] * conj(c[k]);
            }
        }
        dFsigma = matrixop::anticommutator(dF, sigma);

        l1 = dt * (-zI * matrixop::commutator(Emat, rmom) + pmom / mass -  v * matrixop::commutator(dc, rmom));
        m1 = dt * (-zI * matrixop::commutator(Emat, pmom) + 0.5 * dFsigma - v * matrixop::commutator(dc, pmom));
        l2 = dt * (-zI * matrixop::commutator(Emat, rmom + 0.5 * l1) + (pmom + 0.5 * m1) / mass -  v * matrixop::commutator(dc, rmom + 0.5 * l1));
        m2 = dt * (-zI * matrixop::commutator(Emat, pmom + 0.5 * m1) + 0.5 * dFsigma - v * matrixop::commutator(dc, pmom + 0.5 * m1));
        l3 = dt * (-zI * matrixop::commutator(Emat, rmom + 0.5 * l2) + (pmom + 0.5 * m2) / mass -  v * matrixop::commutator(dc, rmom + 0.5 * l2));
        m3 = dt * (-zI * matrixop::commutator(Emat, pmom + 0.5 * m2) + 0.5 * dFsigma - v * matrixop::commutator(dc, pmom + 0.5 * m2));
        l4 = dt * (-zI * matrixop::commutator(Emat, rmom + l3) + (pmom + m3) / mass -  v * matrixop::commutator(dc, rmom + l3));
        m4 = dt * (-zI * matrixop::commutator(Emat, pmom + m3) + 0.5 * dFsigma - v * matrixop::commutator(dc, pmom + m3));
        
        rmom += l1 / 6.0 + l2 / 3.0 + l3 / 3.0 + l4 / 6.0;
        pmom += m1 / 6.0 + m2 / 3.0 + m3 / 3.0 + m4 / 6.0;

        copy(rmom.begin(), rmom.end(), state.begin() + 5);
        copy(pmom.begin(), pmom.end(), state.begin() + 9);
    }

    // nuclear part, VV

    p += 0.5 * dt * F[s+s*edim].real();
    x += dt * p / mass;
    cal_info_nume(x, eva, dc, F, lastevt);
    p += 0.5 * dt * F[s+s*edim].real();

    state[0].real(x);
    state[1].real(p);
}

int hopper(state_t& state) {
    // extract info
    // state = r(1), p(1), c(2), s(1), rmom(4), pmom(4)
    double x = state[0].real();
    double p = state[1].real();
    vector< complex<double> > c(state.begin() + 2, state.begin() + 4);
    int s = static_cast<int>(state[4].real());
    vector< complex<double> > rmom(state.begin() + 5, state.begin() + 9);
    vector< complex<double> > pmom(state.begin() + 9, state.begin() + 13);

    // calc hop prob
    double g = -2 * dt * (c[s] * conj(c[1-s]) * (p / mass * dc[1-s+s*2])).real() / (c[s] * conj(c[s])).real();
    double dE = eva[1-s] - eva[s];
    // hop
    if (randomer::rand() < g) {
        double tmp = p * p - 2 * mass * dE;
        if (tmp > 0.0) {
            // hop accepted
            double pnew = sqrt(tmp);
            state[1] = pnew * (p < 0.0 ? -1 : 1); 
            state[4].real(1.0 - s); 

            // moments
            if (enable_deco) {
                complex<double> ptmp = pmom[1-s+(1-s)*edim];
                complex<double> rtmp = rmom[1-s+(1-s)*edim];
                for (int i(0); i < edim; ++i) {
                    rmom[i+i*edim] -= rtmp;
                    pmom[i+i*edim] -= ptmp;
                }

                copy(rmom.begin(), rmom.end(), state.begin() + 5);
                copy(rmom.begin(), rmom.end(), state.begin() + 9);
            }

            return (s == 0) ? HOP_UP : HOP_DN;
        }
        else {
            // hop frustrated
            if (enable_prev) {
                double F0d01 = (F[0+0*edim] * dc[0+1*edim]).real();
                double F1d01 = (F[1+1*edim] * dc[0+1*edim]).real();
                double vd01 = p / mass * dc[0+1*2].real();
                if (F0d01 * F1d01 < 0.0 and vd01 * F1d01 < 0.0)
                {
                    // momentum reversal
                    state[1] *= -1.0;
                }
            }
            return HOP_FR; 
        }
    }   
    // hop rejected
    return HOP_RJ;
}

void decoherencer(state_t& state, const double xi = 1.0) {
    // extract info
    // state = r(1), p(1), c(2), s(1), rmom(4), pmom(4)
    double x = state[0].real();
    double p = state[1].real();
    vector< complex<double> > c(state.begin() + 2, state.begin() + 4);
    int s = static_cast<int>(state[4].real());
    vector< complex<double> > rmom(state.begin() + 5, state.begin() + 9);
    vector< complex<double> > pmom(state.begin() + 9, state.begin() + 13);

    // collapse
    const double Pcollapse = 0.5 * dt * (F[0+0*edim] - F[1+1*edim]).real() * (rmom[0+0*edim] - rmom[1+1*edim]).real() * 
                ( (rmom[0+0*edim] - rmom[1+1*edim]).real() * (pmom[0+0*edim] - pmom[1+1*edim]).real() > 0.0 ? 1.0 : -1.0);

    if (randomer::rand() < Pcollapse) {
        c[s] = 1.0;
        c[1-s] = 0.0;
        rmom.assign(edim * edim, matrixop::ZEROZ);
        pmom.assign(edim * edim, matrixop::ZEROZ);

        copy(c.begin(), c.end(), state.begin() + 2);
        copy(rmom.begin(), rmom.end(), state.begin() + 5);
        copy(rmom.begin(), rmom.end(), state.begin() + 9);
    }
}


void afssh_1d_mpi() {

    // potential para

    // assign jobs
    const vector<int> my_jobs = MPIer::assign_job(Ntraj);
    const int my_Ntraj = my_jobs.size();

    // initialize
    vector<state_t> state(my_Ntraj);
    for (auto& st : state) {
        init_state(st);
    }

    // statistics
    double hopup = 0.0, hopdn = 0.0, hopfr = 0.0, hoprj = 0.0;
    vector<double> hop_count(my_Ntraj, 0.0);
    vector<double> hop_count_summary(50, 0.0);

    double n0t = 0.0, n1t = 0.0;
    double n0r = 0.0, n1r = 0.0;
    double KE = 0.0, PE = 0.0;

    // recorder
    int Nrec = Nstep / output_step;
    map<string, vector<double> > record {
        { "n0t", vector<double>(Nrec, 0.0) },
        { "n1t", vector<double>(Nrec, 0.0) },
        { "n0r", vector<double>(Nrec, 0.0) },
        { "n1r", vector<double>(Nrec, 0.0) },
        { "KE", vector<double>(Nrec, 0.0) },
        { "PE", vector<double>(Nrec, 0.0) },
    };


    //  ----  MAIN LOOP  ---- //


    vector< vector< complex<double> > > lastevt_save(my_Ntraj);
    for (int istep(0); istep < Nstep; ++istep) {
        for (int itraj(0); itraj < my_Ntraj; ++itraj) {
            if (check_end(state[itraj]) == false) {
                // assign last evt
                lastevt = move(lastevt_save[itraj]);
                // calc info
                cal_info_nume(state[itraj][0].real(), eva, dc, F, lastevt);
                // hopper
                if (enable_hop) {
                    int hopflag = hopper(state[itraj]);
                    switch (hopflag) {
                        case HOP_UP : hopup += 1.0; hop_count[itraj] += 1.0; break;
                        case HOP_DN : hopdn += 1.0; hop_count[itraj] += 1.0; break;
                        case HOP_FR : hopfr += 1.0; break;
                        case HOP_RJ : hoprj += 1.0; break;
                        default : break;
                    }
                }
                // decoherence
                if (enable_deco) {
                    decoherencer(state[itraj]);
                }
                // integrater t -> t + dt
                integrator(state[itraj], dt);
                // save last evt
                lastevt_save[itraj] = move(lastevt);
            }
        }

        if (istep % output_step == 0) {
            // record data
            n0t = n1t = 0.0;
            n0r = n1r = 0.0;
            KE = PE = 0.0;
            for_each(state.begin(), state.end(), 
                    [&n0t, &n1t, &n0r, &n1r, &KE, &PE](const state_t& st) { 
                        double x = st[0].real();
                        double p = st[1].real();
                        int s = static_cast<int>(st[4].real());

                        if (p > 0.0) {
                            if (s == 0) n0t += 1.0;
                            if (s == 1) n1t += 1.0;
                        }
                        else {
                            if (s == 0) n0r += 1.0;
                            if (s == 1) n1r += 1.0;
                        }

                        // energy
                        KE += p * p / mass / 2;

                        vector<double> evatmp;
                        matrixop::hdiag(cal_H(x), evatmp);
                        PE += evatmp[s]; 
                    });

            const int irec = istep / output_step;
            record["n0t"][irec] = n0t;
            record["n1t"][irec] = n1t;
            record["n0r"][irec] = n0r;
            record["n1r"][irec] = n1r;
            record["KE"][irec] = KE;
            record["PE"][irec] = PE;
        }
    }
    MPIer::barrier();


    // ----  COLLECT DATA  ---- //


    for (int r(1); r < MPIer::size; ++r) {
        if (MPIer::rank == r) {
            MPIer::send(0, record["n0t"]);
            MPIer::send(0, record["n1t"]);
            MPIer::send(0, record["n0r"]);
            MPIer::send(0, record["n1r"]);
            MPIer::send(0, record["KE"]);
            MPIer::send(0, record["PE"]);
        }
        else if (MPIer::master) {
            vector<double> vbuf;
            MPIer::recv(r, vbuf); record["n0t"] += vbuf;
            MPIer::recv(r, vbuf); record["n1t"] += vbuf;
            MPIer::recv(r, vbuf); record["n0r"] += vbuf;
            MPIer::recv(r, vbuf); record["n1r"] += vbuf;
            MPIer::recv(r, vbuf); record["KE"] += vbuf;
            MPIer::recv(r, vbuf); record["PE"] += vbuf;
        }
        MPIer::barrier();
    }
    MPIer::barrier();


    // ----  PROCESS & COLLECT HOP STATISTICS DATA  ---- //


    for_each(hop_count.begin(), hop_count.end(),
            [&hop_count_summary](double x) { hop_count_summary[static_cast<int>(x)] += 1.0; });
    MPIer::barrier();

    for (int r = 1; r < MPIer::size; ++r) {
        if (MPIer::rank == r) {
            MPIer::send(0, hopup, hopdn, hopfr, hoprj, hop_count_summary);
        }
        else if (MPIer::master) {
            double buf;
            vector<double> vbuf;

            MPIer::recv(r, buf); hopup += buf;
            MPIer::recv(r, buf); hopdn += buf;
            MPIer::recv(r, buf); hopfr += buf;
            MPIer::recv(r, buf); hoprj += buf;
            MPIer::recv(r, vbuf); hop_count_summary += vbuf;
        }
        MPIer::barrier();
    }
    MPIer::barrier();


    // ----  OUTPUT  ---- //


    if (MPIer::master) {
        // para & header
        output_potential_params();
        ioer::info("# FSSH para: ", " Ntraj = ", Ntraj, " Nstep = ", Nstep, " dt = ", dt, " output_step = ", output_step,
                    " mass = ", mass, 
                    " init_x = ", init_x, " init_p = ", init_p, 
                    " sigma_x = ", sigma_x, " sigma_p = ", sigma_p, 
                    " init_s = ", init_s,
                    " enable_hop = ", enable_hop,
                    " enable_deco = ", enable_deco,
                    " enable_prev = ", enable_prev,
                    " seed = ", seed
                );
        ioer::tabout('#', "t", "n0t", "n1t", "n0r", "n1r", "KE", "PE", "Etot");

        for (int irec = 0; irec < Nrec; ++irec) {
            ioer::tabout(
                    "#", irec * output_step * dt, 
                    record["n0t"][irec] / Ntraj, 
                    record["n1t"][irec] / Ntraj, 
                    record["n0r"][irec] / Ntraj, 
                    record["n1r"][irec] / Ntraj, 
                    record["KE"][irec] / Ntraj, 
                    record["PE"][irec] / Ntraj, 
                    (record["KE"][irec] + record["PE"][irec]) / Ntraj
                    );
        }

        ioer::tabout(
                "", init_p,
                record["n0t"][Nrec-1] / Ntraj, 
                record["n1t"][Nrec-1] / Ntraj, 
                record["n0r"][Nrec-1] / Ntraj, 
                record["n1r"][Nrec-1] / Ntraj, 
                record["KE"][Nrec-1] / Ntraj, 
                record["PE"][Nrec-1] / Ntraj, 
                (record["KE"][Nrec-1] + record["PE"][Nrec-1]) / Ntraj
                );

        // output hop info
        ioer::info("# hop statistics: ");
        ioer::info("# hopup = ", hopup, " hopdn = ", hopdn, " hopfr = ", hopfr, " hopfr_rate = ", hopfr / (hopup + hopdn + hopfr));
        ioer::info("# hop count: ", hop_count_summary);
    }
    MPIer::barrier();
}

void check() {
    // check surf
    for (double x = -10; x < 10; x += 0.1) {
        cal_info_nume(x, eva, dc, F, lastevt);
        ioer::tabout(x, eva, F[0+0*edim].real(), F[1+1*edim].real(), dc[0+1*edim].real(), dc[0+1*edim].imag());
    }
}


// -- MAIN -- //


int main(int argc, char** argv) {
    MPIer::setup();
    if (argc < 2) {
        if (MPIer::master) 
            ioer::info("use --help for detailed info");
    }
    else if (string(argv[1]) == "check") {
        if (MPIer::master) 
            check();
    }
    else {
        if (argparse(argc, argv) == true) {
            randomer::seed(MPIer::assign_random_seed(seed));
            if (MPIer::master) timer::tic();
            afssh_1d_mpi();
            if (MPIer::master) ioer::info("# ", timer::toc());
        }
    }
    MPIer::barrier();
    MPIer::finalize();
    return 0;
}
