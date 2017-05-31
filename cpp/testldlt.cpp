#include <iostream>
using namespace std;

#include "Eigen/Dense"
using namespace Eigen;

#include "utils/random.h"

int main( int argc, char *argv[] ) {
    int N = 5;
    Eigen::MatrixXd A_(N,N);
    seed(0);
    for( int i = 0; i < N; i++ ) {
        for( int j = 0; j < N; j++ ) {
            A_(i,j) = nextDouble();
        } 
    } 
    Eigen::MatrixXd A = A_.transpose() * A_;
    cout << "A:\n" << A << endl;
    LDLT<MatrixXd> ldlt = A.ldlt();
    MatrixXd ldltL = ldlt.matrixL();
    //cout << "ldlt lower: \n" << ldltL << endl;
    VectorXd ldltD = ldlt.vectorD();
    //cout << " vectorD \n" << ldltD << endl;
    //cout << "sqrtD:\n";
    for( int i = 0; i < N; i++ ) {
      //  cout << sqrt(ldltD(i)) << endl;
    }
    LLT<MatrixXd> llt = A.llt();
    Eigen::MatrixXd lltLower = llt.matrixL();
    //cout << "lltLower: \n" << lltLower << endl;
    //cout << "LL': \n" << (lltLower * lltLower.transpose() ) << endl;
    MatrixXd Dmatrix(N,N);
    MatrixXd Dsqrtmatrix(N,N);
    for( int i =0 ; i < N; i++ ) {
        for( int j = 0; j < N; j++ ) {
            Dmatrix(i,j) = 0;
            if( i == j ) {
                Dmatrix(i,i) = ldltD(i);
                Dsqrtmatrix(i,i) = sqrt(ldltD(i));
            }
        }
    }
//    cout << "transpositionsP:\n" << ldlt.transpositionsP().indices() << endl;
    //cout << "D\n" << Dmatrix << endl;
    cout << "LL' L:\n" << lltLower << endl;
    cout << "Lsqrt(D) \n" << (ldltL * Dsqrtmatrix ) << endl;
    MatrixXd ptlsqrtd = ldlt.transpositionsP().transpose()*ldltL * Dsqrtmatrix;
    cout << "P'Lsqrt(D) \n" << (ptlsqrtd ) << endl;
    cout << "ptlsqrtd * ptlsqrtd' \n" << (ptlsqrtd * ptlsqrtd.transpose() ) << endl;
    //cout << "Lsqrt(D)sqrt(D)'L' \n" << (ldltL * Dsqrtmatrix * Dsqrtmatrix.transpose() * ldltL.transpose() ) << endl;
    cout << "LDL' \n" << (ldltL * Dmatrix * ldltL.transpose() ) << endl;
    MatrixXd ldltLDLT = ldltL * Dmatrix * ldltL.transpose();
    cout << "P'LDL'P \n" << (ldlt.transpositionsP().transpose()*ldltLDLT*ldlt.transpositionsP() ) << endl;
    cout << "PAP'\n" << (ldlt.transpositionsP()* A * ldlt.transpositionsP().transpose() ) << endl;

    MatrixXd probe(N,N);
    for( int i = 0; i < N; i++ ) {
        for( int j = 0; j < N; j++ ) {
            probe(i,j) = i*N+j;
        }
    }
//    cout << "probe\n" << probe << endl;
//    cout << "trans * probe\n" << (ldlt.transpositionsP()*probe) << endl;
    cout << "probe*trans\n" << (probe*ldlt.transpositionsP()) << endl;
//    cout << "trans' * probe\n" << (ldlt.transpositionsP().transpose()*probe) << endl;
//    cout << "probe*trans'\n" << (probe*ldlt.transpositionsP().transpose()) << endl;

    MatrixXd Ainv = A.inverse();
    cout << "Ainv: \n" << Ainv << endl;
    {
        MatrixXd LrootD = ldlt.matrixL() * Dsqrtmatrix;
        MatrixXd P = ldlt.transpositionsP() * MatrixXd::Identity(N,N);
        MatrixXd M = LrootD.triangularView<Eigen::Lower>().solve(P).transpose();
        MatrixXd MMT = M * M.transpose();
        cout << "MMT:\n" << MMT << endl;
    }

    {
        MatrixXd LrootD(N,N);
        for( int j = 0; j < N; j++ ) {
            double sqrtdj = sqrt(ldltD(j));
            for( int i = 0; i < N; i++ ) {
                LrootD(i,j) = ldltL(i,j) * sqrtdj;
            }
        }
        MatrixXd P = ldlt.transpositionsP() * MatrixXd::Identity(N,N);
        MatrixXd M = LrootD.triangularView<Eigen::Lower>().solve(P).transpose();
        MatrixXd MMT = M * M.transpose();
        cout << "MMT:\n" << MMT << endl;
    }

    return 0;
}

