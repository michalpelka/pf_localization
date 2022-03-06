#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
namespace save_mat_util{
    Eigen::Matrix4d loadMat(const std::string& fn){
        Eigen::Matrix4d m;
        std::ifstream ifss(fn);
        for (int i =0; i < 16; i++) {
            ifss >> m.data()[i];
        }
        ifss.close();
        std::cout << m.transpose() << std::endl;
        return m.transpose();;
    }
    void saveMat(const std::string& fn, const Eigen::Matrix4d& mat){
        Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "", "");
        std::ofstream fmt_ofs (fn);
        fmt_ofs << mat.format(HeavyFmt);
        fmt_ofs.close();
    }

}