#include <mpi.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>

const size_t N = 512;
const size_t d = 512;   //Matrix Size N*d

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> h_Q, h_K, h_V;

    h_K.resize(N * d);
    h_V.resize(N * d);

    if (rank == 0) {
        h_Q.resize(N * d);

        std::ifstream infile("../data.in");
        if (!infile) {
            std::cerr << "无法打开文件！" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 顺序读取所有数据
        for (size_t i = 0; i < N * d; ++i) infile >> h_Q[i];
        for (size_t i = 0; i < N * d; ++i) infile >> h_K[i];
        for (size_t i = 0; i < N * d; ++i) infile >> h_V[i];
        
        infile.close();
        std::cout << "文件读取完成。" << std::endl;
    }

    int local_n = N / size;
    std::vector<double> local_q(local_n * d);

    MPI_Scatter(h_Q.data(), local_n * d, MPI_DOUBLE, local_q.data(), local_n * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(h_K.data(), N * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_V.data(), N * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    using MatrixRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<MatrixRM> Q_local(local_q.data(), local_n, d);
    Eigen::Map<MatrixRM> K_gloabl(h_K.data(), N, d);
    Eigen::Map<MatrixRM> V_gloabl(h_V.data(), N, d);

    // 计算softmax

    MatrixRM scores = Q_local * K_gloabl.transpose() / std::sqrt(static_cast<double>(d));
    Eigen::VectorXd row_maxes = scores.rowwise().maxCoeff();
    scores.colwise() -= row_maxes;
    scores = scores.array().exp().matrix();

    Eigen::VectorXd row_sums = scores.rowwise().sum();
    scores.array().colwise() /= row_sums.array();

    MatrixRM local_output = scores * V_gloabl;

    std::vector<double> h_output;
    if (rank == 0) {
        h_output.resize(N * d);
    }
    
    MPI_Gather(local_output.data(), local_n * d, MPI_DOUBLE, h_output.data(), local_n * d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Attention 并行计算圆满完成！" << std::endl;
        std::cout << "--- 对拍数据 ---" << std::endl;
        for (size_t j = 0; j < 2; j ++ ) {
            for (size_t i = 0; i < 20; i ++ ) {
                std::cout << h_output[j * d + i] << ' ' ;
            }
            std::cout << '\n';
        }
        std::cout << "---------------" << std::endl;
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}