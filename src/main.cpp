#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "OsqpEigen/OsqpEigen.h"
#include "logging-utils-lib/logging.h"
#include "logging-utils-lib/filesystem.h"
#include "logging-utils-lib/yaml.h"
#include "progress_bar.h"
#include <string>
#include <vector>
#include <math.h>

Eigen::SparseMatrix<double, ColMajor>    leastSquares_P(const unsigned int n, const unsigned int m);
Eigen::SparseMatrix<double, ColMajor>    leastSquares_A(const unsigned int n, const unsigned int m,
                                                        const Eigen::MatrixXd &A_d);
Eigen::VectorXd leastSquares_q(const unsigned int n, const unsigned int m);
Eigen::VectorXd leastSquares_l(const unsigned int n, const unsigned int m,
                                                        const Eigen::MatrixXd &b, const Eigen::MatrixXd &lb);
Eigen::VectorXd leastSquares_u(const unsigned int n, const unsigned int m,
                                                        const Eigen::MatrixXd &b, const Eigen::MatrixXd &ub);

template<typename derived>
void solveMuxMuyProblem(const unsigned int &num_datapoints,
                        const Eigen::MatrixXd &phi_data,
                        const Eigen::MatrixXd &tht_data,
                        const Eigen::MatrixXd &p_data,
                        const Eigen::MatrixXd &q_data,
                        const Eigen::MatrixXd &r_data,
                        const Eigen::MatrixXd &u_data,
                        const Eigen::MatrixXd &u_dot_data,
                        const Eigen::MatrixXd &v_data,
                        const Eigen::MatrixXd &v_dot_data,
                        const Eigen::MatrixXd &w_data,
                        const double &m_, const double &g_,
                        const double &mu_max, Eigen::MatrixBase<derived> &sol);

int main(int argc, char *argv[])
{
    std::vector<std::string> program_args;
    if (argc > 1)
        program_args.assign(argv + 1, argv + argc);
    else
        return 1;

    // Progress bar
    ProgressBar pb;

    // Read config file values
    std::string configfname = program_args[0];
    std::string logdirname;
    logging::get_yaml_node<std::string>("log_dir", configfname, logdirname);
    double m_, g_, mu_max;
    logging::get_yaml_node<double>("m", configfname, m_);
    logging::get_yaml_node<double>("g", configfname, g_);
    logging::get_yaml_node<double>("mu_max", configfname, mu_max);

    // Extract log data
    std::cout << "Reading logs..." << std::endl;
    Eigen::MatrixXd phi_data, tht_data, p_data, q_data, r_data, u_data,
                    u_dot_data, v_data, v_dot_data, w_data;
    logging::logToMatrix(logdirname + "/phi.log",     phi_data, 2);
    logging::logToMatrix(logdirname + "/tht.log",     tht_data, 2);
    logging::logToMatrix(logdirname + "/p.log",         p_data, 2);
    logging::logToMatrix(logdirname + "/q.log",         q_data, 2);
    logging::logToMatrix(logdirname + "/r.log",         r_data, 2);
    logging::logToMatrix(logdirname + "/u.log",         u_data, 2);
    logging::logToMatrix(logdirname + "/u_dot.log", u_dot_data, 2);
    logging::logToMatrix(logdirname + "/v.log",         v_data, 2);
    logging::logToMatrix(logdirname + "/v_dot.log", v_dot_data, 2);
    logging::logToMatrix(logdirname + "/w.log",         w_data, 2);

    // Solve a bunch of osqp problems
    unsigned int N = static_cast<unsigned int>(phi_data.cols());
    pb.init(static_cast<int>(N), 50);
    std::cout << "Solving a bunch of least squares problems from n = 1 -> n = " << N << "..." << std::endl;
    Eigen::MatrixXd sols; sols.resize(2, N+1);

    sols.col(0)[0] = m_;
    sols.col(0)[1] = g_;
    pb.print(0);
    for (unsigned int i = 1; i <= N; i++)
    {
        auto sol = sols.col(i);
        solveMuxMuyProblem(i, phi_data, tht_data, p_data, q_data, r_data, u_data, u_dot_data, v_data,
                           v_dot_data, w_data, m_, g_, mu_max, sol);
        pb.print(static_cast<int>(i));
    }

    std::cout << std::endl << "DONE. Writing log file to " << logdirname << "..." << std::endl;

    logging::matrixToLog(logdirname + "/SOLS.log", sols);

    return 0;
}

template<typename derived>
void solveMuxMuyProblem(const unsigned int &num_datapoints,
                        const Eigen::MatrixXd &phi_data,
                        const Eigen::MatrixXd &tht_data,
                        const Eigen::MatrixXd &p_data,
                        const Eigen::MatrixXd &q_data,
                        const Eigen::MatrixXd &r_data,
                        const Eigen::MatrixXd &u_data,
                        const Eigen::MatrixXd &u_dot_data,
                        const Eigen::MatrixXd &v_data,
                        const Eigen::MatrixXd &v_dot_data,
                        const Eigen::MatrixXd &w_data,
                        const double &m_, const double &g_,
                        const double &mu_max, Eigen::MatrixBase<derived> &sol)
{
    unsigned int di = static_cast<unsigned int>(phi_data.cols() / num_datapoints);
    unsigned int n = 2;
    unsigned int m = 2 * num_datapoints;
    Eigen::MatrixXd A_d, b, mu_lower, mu_upper;
    mu_lower.resize(n,1); mu_lower.setZero();
    mu_upper.resize(n,1); mu_upper.setOnes(); mu_upper *= mu_max;
    A_d.resize(m, n);
    b.resize(m, 1);

    for (unsigned int i = 0; i < num_datapoints; i++)
    {
        unsigned int i_idx = i * di;
        double phi_i = phi_data.col(i_idx)[1];
        double tht_i = tht_data.col(i_idx)[1];
        double p_i = p_data.col(i_idx)[1];
        double q_i = q_data.col(i_idx)[1];
        double r_i = r_data.col(i_idx)[1];
        double u_i = u_data.col(i_idx)[1];
        double v_i = v_data.col(i_idx)[1];
        double w_i = w_data.col(i_idx)[1];
        double u_dot_i = u_dot_data.col(i_idx)[1];
        double v_dot_i = v_dot_data.col(i_idx)[1];

        A_d(2*i+0,0) = -u_i/m_; A_d(2*i+0,1) =    0.0;
        A_d(2*i+1,0) =    0.0; A_d(2*i+1,1) = -v_i/m_;

        b(2*i+0,0) = -g_ * sin(tht_i) + (v_i*r_i - w_i*q_i) - u_dot_i;
        b(2*i+1,0) = g_ * sin(phi_i) * cos(tht_i) + (w_i*p_i - u_i*r_i) - v_dot_i;
    }

    Eigen::SparseMatrix<double> P, A;
    Eigen::VectorXd q, l, u;

    P = leastSquares_P(n, m);
    A = leastSquares_A(n, m, A_d);
    q = leastSquares_q(n, m);
    l = leastSquares_l(n, m, b, mu_lower);
    u = leastSquares_u(n, m, b, mu_upper);

    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(false);
    solver.data()->setNumberOfVariables(static_cast<int>(n+m));
    solver.data()->setNumberOfConstraints(static_cast<int>(n+m));
    solver.data()->setHessianMatrix(P);
    solver.data()->setGradient(q);
    solver.data()->setLinearConstraintsMatrix(A);
    solver.data()->setLowerBound(l);
    solver.data()->setUpperBound(u);
    solver.initSolver();
    solver.solve();

    sol = solver.getSolution().topRows(n);
}

Eigen::SparseMatrix<double, ColMajor> leastSquares_P(const unsigned int n, const unsigned int m)
{
    Eigen::SparseMatrix<double, ColMajor> P;
    P.resize(m+n,m+n);
    P.setZero();
    for (unsigned int i = 0; i < m; i++)
        P.insert(n+i,n+i) = 1.0;
    return P;
}

Eigen::SparseMatrix<double, ColMajor> leastSquares_A(const unsigned int n, const unsigned int m,
                                                     const Eigen::MatrixXd &A_d)
{
    Eigen::SparseMatrix<double, ColMajor> A;
    A.resize(n+m,n+m);
    A.setZero();
    for (unsigned int i = 0; i < m; i++)
    {
        A.insert(i, n+i) = -1.0;
        for (unsigned int j = 0; j < n; j++)
            A.insert(i, j) = A_d(i, j);
    }
    for (unsigned int k = 0; k < n; k++)
        A.insert(m+k, k) = 1.0;
    return A;
}

Eigen::VectorXd leastSquares_q(const unsigned int n, const unsigned int m)
{
    Eigen::VectorXd q;
    q.resize(m+n,1);
    q.setZero();
    return q;
}

Eigen::VectorXd leastSquares_l(const unsigned int n, const unsigned int m,
                                                        const Eigen::MatrixXd &b, const Eigen::MatrixXd &lb)
{
    Eigen::VectorXd l;
    l.resize(m+n,1);
    l.topRows(m) = b;
    l.bottomRows(n) = lb;
    return l;
}

Eigen::VectorXd leastSquares_u(const unsigned int n, const unsigned int m,
                                                        const Eigen::MatrixXd &b, const Eigen::MatrixXd &ub)
{
    Eigen::VectorXd u;
    u.resize(m+n,1);
    u.topRows(m) = b;
    u.bottomRows(n) = ub;
    return u;
}
