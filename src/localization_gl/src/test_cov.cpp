#include <iostream>
#include <thread>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <GL/freeglut.h>
#include <mutex>
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include "structs.h"
#include "pf.h"

#include "mirror_converter.h"
#include "save_mat.h"
#include "transformations.h"
#include "point_to_point_source_to_landmark_tait_bryan_wc_jacobian.h"
#include "point_to_point_source_to_landmark_tait_bryan_wc_cov.h"

bool imgui_log_trajectory{false};
const unsigned int window_width = 1920;
const unsigned int window_height = 1080;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -60.0;
float translate_x = 0.0;
float translate_y = 0.0;
bool gui_mouse_down{false};
int windows_handle = -1;

pcl::PointCloud<pcl::PointXYZ> input_data_obstacles;
pcl::PointCloud<pcl::PointXYZ> input_data_floor;


pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_traversability(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr scan(new pcl::PointCloud<pcl::PointXYZ>);
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
Sophus::SE3f pose {Eigen::Matrix4f::Identity()};
Eigen::Matrix3d covPose {Eigen::Matrix3d::Identity()};
Eigen::Matrix<double,6,6> Sigma;
float scale = 100;
void display();
void reshape(int w, int h);
void mouse(int glut_button, int state, int x, int y);
void motion(int x, int y);
bool initGL(int *argc, char **argv);
void processSpecialKeys(unsigned char key, int x, int y);
struct imgui_data{
    int resample_type = 1;
    int sensor_type= 1;
}imgui_data;

struct PointMeanCov{
//    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
    Eigen::Vector3d coords;
};

void calculate_ICP_COV(std::vector<PointMeanCov>& data_pi,
                       std::vector<PointMeanCov>& model_qi, Eigen::Affine3d transform, Eigen::MatrixXd& ICP_COV)
{
    TaitBryanPose pose = pose_tait_bryan_from_affine_matrix(transform);

    Eigen::MatrixXd d2sum_dbeta2(6,6);
    d2sum_dbeta2 = Eigen::MatrixXd::Zero(6,6);

    for (size_t s = 0; s < data_pi.size(); ++s )
    {
        double pix = data_pi[s].coords.x();
        double piy = data_pi[s].coords.y();
        double piz = data_pi[s].coords.z();
        double qix = model_qi[s].coords.x();
        double qiy = model_qi[s].coords.y();
        double qiz = model_qi[s].coords.z();

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> d2sum_dbeta2i;
        point_to_point_source_to_landmark_tait_bryan_wc_d2sum_dbeta2(d2sum_dbeta2i, pose.px, pose.py, pose.pz, pose.om,
                                                                     pose.fi, pose.ka, pix, piy, piz, qix, qiy, qiz);

        Eigen::MatrixXd d2sum_dbeta2_temp(6,6);
        d2sum_dbeta2_temp << d2sum_dbeta2i;
        d2sum_dbeta2 = d2sum_dbeta2 + d2sum_dbeta2_temp;
    }

    int n = data_pi.size();
    Eigen::MatrixXd d2sum_dbetadx(6,6*n);
    for (int k = 0; k < n ; ++k)
    {
        double pix = data_pi[k].coords.x();
        double piy = data_pi[k].coords.y();
        double piz = data_pi[k].coords.z();
        double qix = model_qi[k].coords.x();
        double qiy = model_qi[k].coords.y();
        double qiz = model_qi[k].coords.z();

        Eigen::MatrixXd d2sum_dbetadx_temp(6,6);
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> d2sum_dbetadxi;
        point_to_point_source_to_landmark_tait_bryan_wc_d2sum_dbetadx(d2sum_dbetadxi, pose.px, pose.py, pose.pz, pose.om,
                                                                      pose.fi, pose.ka, pix, piy, piz, qix, qiy, qiz);

        d2sum_dbetadx_temp << d2sum_dbetadxi;
        d2sum_dbetadx.block<6,6>(0,6*k) = d2sum_dbetadx_temp;
    }

    Eigen::MatrixXd cov_x(6*n,6*n);
    cov_x = 0.0 * Eigen::MatrixXd::Identity(6*n,6*n);

    for(size_t i = 0; i < n ; i ++){
        int row = i * 6;
        int col = i * 6;

        cov_x(row, col + 0) = data_pi[i].cov(0,0);
        cov_x(row, col + 1) = data_pi[i].cov(0,1);
        cov_x(row, col + 2) = data_pi[i].cov(0,2);

        cov_x(row + 1, col + 0) = data_pi[i].cov(1,0);
        cov_x(row + 1, col + 1) = data_pi[i].cov(1,1);
        cov_x(row + 1, col + 2) = data_pi[i].cov(1,2);

        cov_x(row + 2, col + 0) = data_pi[i].cov(2,0);
        cov_x(row + 2, col + 1) = data_pi[i].cov(2,1);
        cov_x(row + 2, col + 2) = data_pi[i].cov(2,2);

        cov_x(row + 3, col + 3 + 0) = model_qi[i].cov(0,0);
        cov_x(row + 3, col + 3 + 1) = model_qi[i].cov(0,1);
        cov_x(row + 3, col + 3 + 2) = model_qi[i].cov(0,2);

        cov_x(row + 4, col + 3 + 0) = model_qi[i].cov(1,0);
        cov_x(row + 4, col + 3 + 1) = model_qi[i].cov(1,1);
        cov_x(row + 4, col + 3 + 2) = model_qi[i].cov(1,2);

        cov_x(row + 5, col + 3 + 0) = model_qi[i].cov(2,0);
        cov_x(row + 5, col + 3 + 1) = model_qi[i].cov(2,1);
        cov_x(row + 5, col + 3 + 2) = model_qi[i].cov(2,2);
    }
    ICP_COV =  d2sum_dbeta2.inverse() * d2sum_dbetadx * cov_x * d2sum_dbetadx.transpose() * d2sum_dbeta2.inverse();
}

void draw_confusion_ellipse(const Eigen::Matrix3d& covar, Eigen::Vector3d& mean, Eigen::Vector3f color, float nstd  = 3)
{

    Eigen::LLT<Eigen::Matrix<double,3,3> > cholSolver(covar);
    Eigen::Matrix3d transform = cholSolver.matrixL();
//    std::cout << "transform " << std::endl;
//    std::cout << transform << std::endl;
    const double pi = 3.141592;
    const double di =0.02;
    const double dj =0.04;
    const double du =di*2*pi;
    const double dv =dj*pi;
    glColor3f(color.x(), color.y(),color.z());

    for (double i = 0; i < 1.0; i+=di)  //horizonal
        for (double j = 0; j < 1.0; j+=dj)  //vertical
        {
            double u = i*2*pi;      //0     to  2pi
            double v = (j-0.5)*pi;  //-pi/2 to pi/2

            const Eigen::Vector3d pp0( cos(v)* cos(u),cos(v) * sin(u),sin(v));
            const Eigen::Vector3d pp1(cos(v) * cos(u + du) ,cos(v) * sin(u + du) ,sin(v));
            const Eigen::Vector3d pp2(cos(v + dv)* cos(u + du) ,cos(v + dv)* sin(u + du) ,sin(v + dv));
            const Eigen::Vector3d pp3( cos(v + dv)* cos(u),cos(v + dv)* sin(u),sin(v + dv));
            Eigen::Vector3d tp0 = transform * (nstd*pp0) + mean;
            Eigen::Vector3d tp1 = transform * (nstd*pp1) + mean;
            Eigen::Vector3d tp2 = transform * (nstd*pp2) + mean;
            Eigen::Vector3d tp3 = transform * (nstd*pp3) + mean;

            glBegin(GL_LINE_LOOP);
            glVertex3dv(tp0.data());
            glVertex3dv(tp1.data());
            glVertex3dv(tp2.data());
            glVertex3dv(tp3.data());
            glEnd();
        }
}

struct NN{
    unsigned int indx1;
    unsigned int indx2;
    Eigen::Vector3f ps; //point from scan, global
    Eigen::Vector3f psL; //point from scan, local
    Eigen::Vector3f pm;  //point from map, global
};
std::vector<NN> nearest_neighborhood;
int gl_main(int argc, char *argv[]){
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(processSpecialKeys);
    glutMainLoop();
}

void computeNN()
{
    nearest_neighborhood.clear();
    for (unsigned int i = 0; i<scan->size(); i++ )
    {
        pcl::PointXYZ pppt;
        const Eigen::Vector4f pp =  (*scan)[i].getArray4fMap();
        const Eigen::Vector4f ppt = pose.matrix() * pp;
        pppt.getArray3fMap() = ppt.head<3>();
        const int K = 1;
        std::vector<int> pointIdxKNNSearch(K);
        std::vector<float> pointKNNSquaredDistance(K);

        if ( kdtree.nearestKSearch (pppt, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0 )
        {
            const unsigned int j (pointIdxKNNSearch.front());
            NN n{i,j,  ppt.head<3>(), pp.head<3>(), (*map_cloud)[j].getArray3fMap()};
            nearest_neighborhood.emplace_back(n);
        }
    }
}
int main (int argc, char *argv[])
{

    //ToDo data.host_map -> load map from file

    pcl::io::loadPCDFile("/media/michal/ext/garaz2/CAD/p7p_cloud_clean_2d.pcd", *map_cloud);
    pcl::io::loadPCDFile("/media/michal/ext/garaz2/CAD/p7p_cloud_clean_2d_traversability.pcd", *map_traversability);
    pcl::io::loadPCDFile("/home/michal/code/livox_ws/src/pf_localization/test_data/test_wining_particle.pcd", *scan);
    Eigen::Matrix4d m = save_mat_util::loadMat("/home/michal/code/livox_ws/src/pf_localization/test_data/test_wining_particle.txt");
    kdtree.setInputCloud (map_cloud);

    pose = Sophus::SE3f(m.cast<float>());
    gl_main(argc, argv);
    glutDestroyWindow(windows_handle);

}

void display() {
    std::vector<Point> points;
    double timestamp = 0;



    ImGuiIO& io = ImGui::GetIO();
    glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.0, 1.0);

    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1, 0.0f, 0.0f);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1, 0.0f);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1);
    glEnd();

    glPointSize(10);


    glBegin(GL_POINTS);
    for (int i=0; i< points.size();i++)
    {
        const auto &p = points[i];
        if (p.label == PointType::obstacle)
        {
            glColor3f(0.6, 1, 0.1);
        }else{
            glColor3f(1.0, 1.0, 1.0);
        }

        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
    glPointSize(1);



    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.0f, 0.0f);
//    //ToDo data -> render map from file
    for (const auto &p : *map_cloud)
    {
        glVertex3f(p.x, p.y,p.z);
    }
    glEnd();
    ///////////////////////

    glPointSize(10);
    glBegin(GL_POINTS);
    for (const auto &p : *scan)
    {

        glColor3f(1.0, 1.0, 1.0);
        const Eigen::Vector4f pt =  p.getArray4fMap();
        const Eigen::Vector4f ppt = pose.matrix() * pt;
        glVertex3f(ppt.x(), ppt.y(), ppt.z());
    }
    glEnd();
    Eigen::Vector3d tt=pose.translation().cast<double>();
    draw_confusion_ellipse((double)scale*Sigma.block<3,3>(0,0),tt, Eigen::Vector3f(1,1,0));

    for (const auto &nn : nearest_neighborhood){
            glBegin(GL_LINES);
            glColor3f(1.0f, 1.0f, 1.0f);
            glVertex3fv(nn.ps.data());
            glVertex3fv(nn.pm.data());
            glEnd();
    }
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::Begin("Demo Window1");

    std::stringstream  ss;
    ss << Sigma;
    ImGui::Text(ss.str().c_str());

    if(ImGui::Button("nn")){
        computeNN();
    }


    if(ImGui::Button("icp")) {
        computeNN();
        std::vector<Eigen::Triplet<double>> tripletListA;
        std::vector<Eigen::Triplet<double>> tripletListP;
        std::vector<Eigen::Triplet<double>> tripletListB;

        TaitBryanPose _pose = pose_tait_bryan_from_affine_matrix(Eigen::Affine3d(pose.matrix().cast<double>()));

        Eigen::MatrixXd Err(3,nearest_neighborhood.size());
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> hessian_sum;
        hessian_sum.setZero();
        for (int i = 0; i < nearest_neighborhood.size(); i++)
        {
            const auto & nn = nearest_neighborhood[i];
            // todo check!!
            Eigen::Vector3d p_1 = nn.psL.cast<double>();
            Eigen::Vector3d p_l = nn.pm.cast<double>();

            double delta_x;
            double delta_y;
            double delta_z;
            point_to_point_source_to_landmark_tait_bryan_wc(delta_x, delta_y, delta_z,
                                                            _pose.px,_pose.py,_pose.pz,
                                                            _pose.om,_pose.fi,_pose.ka,
                                                            p_1.x(),p_1.y(),p_1.z(),
                                                            p_l.x(),p_l.y(),p_l.z());
            Eigen::Matrix<double, 3, 6, Eigen::RowMajor> jacobian;
            point_to_point_source_to_landmark_tait_bryan_wc_jacobian(jacobian,
                                                           _pose.px,_pose.py,_pose.pz,
                                                           _pose.om,_pose.fi,_pose.ka,
                                                            p_1.x(),p_1.y(),p_1.z());
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> hessian;

            point_to_point_source_to_landmark_tait_bryan_wc_hessian(hessian,
                                                           _pose.px,_pose.py,_pose.pz,
                                                           _pose.om,_pose.fi,_pose.ka,
                                                            p_1.x(),p_1.y(),p_1.z(),
                                                            p_l.x(),p_l.y(),p_l.z());
            hessian_sum+=hessian;
            Err(0,i) = delta_x;
            Err(1,i) = delta_y;
            Err(2,i) = delta_z;

            int ir = tripletListB.size();
            int ic_1 = 0;
            tripletListA.emplace_back(ir     , ic_1    , -jacobian(0,0));
            tripletListA.emplace_back(ir     , ic_1 + 1, -jacobian(0,1));
            tripletListA.emplace_back(ir     , ic_1 + 2, -jacobian(0,2));
            tripletListA.emplace_back(ir     , ic_1 + 3, -jacobian(0,3));
            tripletListA.emplace_back(ir     , ic_1 + 4, -jacobian(0,4));
            tripletListA.emplace_back(ir     , ic_1 + 5, -jacobian(0,5));

            tripletListA.emplace_back(ir + 1 , ic_1    , -jacobian(1,0));
            tripletListA.emplace_back(ir + 1 , ic_1 + 1, -jacobian(1,1));
            tripletListA.emplace_back(ir + 1 , ic_1 + 2, -jacobian(1,2));
            tripletListA.emplace_back(ir + 1 , ic_1 + 3, -jacobian(1,3));
            tripletListA.emplace_back(ir + 1 , ic_1 + 4, -jacobian(1,4));
            tripletListA.emplace_back(ir + 1 , ic_1 + 5, -jacobian(1,5));

            tripletListA.emplace_back(ir + 2 , ic_1    , -jacobian(2,0));
            tripletListA.emplace_back(ir + 2 , ic_1 + 1, -jacobian(2,1));
            tripletListA.emplace_back(ir + 2 , ic_1 + 2, -jacobian(2,2));
            tripletListA.emplace_back(ir + 2 , ic_1 + 3, -jacobian(2,3));
            tripletListA.emplace_back(ir + 2 , ic_1 + 4, -jacobian(2,4));
            tripletListA.emplace_back(ir + 2 , ic_1 + 5, -jacobian(2,5));

            tripletListP.emplace_back(ir    , ir    ,  1);
            tripletListP.emplace_back(ir + 1, ir + 1,  1);
            tripletListP.emplace_back(ir + 2, ir + 2,  1);

            tripletListB.emplace_back(ir    , 0,  delta_x);
            tripletListB.emplace_back(ir + 1, 0,  delta_y);
            tripletListB.emplace_back(ir + 2, 0,  delta_z);
        }

        Eigen::SparseMatrix<double> matA(tripletListB.size(), 6);
        Eigen::SparseMatrix<double> matP(tripletListB.size(), tripletListB.size());
        Eigen::SparseMatrix<double> matB(tripletListB.size(), 1);

        matA.setFromTriplets(tripletListA.begin(), tripletListA.end());
        matP.setFromTriplets(tripletListP.begin(), tripletListP.end());
        matB.setFromTriplets(tripletListB.begin(), tripletListB.end());

        Eigen::SparseMatrix<double> AtPA(6, 6);
        Eigen::SparseMatrix<double> AtPB(6, 1);

        {
            Eigen::SparseMatrix<double> AtP = matA.transpose() * matP;
//            AtPA = (AtP) * matA;
            AtPA = hessian_sum.sparseView();
            AtPB = (AtP) * matB;
        }

        std::cout << "AtPA.size: " << AtPA.size() << std::endl;
        std::cout << "AtPB.size: " << AtPB.size() << std::endl;

        std::cout << "start solving AtPA=AtPB" << std::endl;
        Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver(AtPA);

        std::cout << "x = solver.solve(AtPB)" << std::endl;
        Eigen::SparseMatrix<double> x = solver.solve(AtPB);

        std::vector<double> h_x;

        for (int k=0; k<x.outerSize(); ++k){
            for (Eigen::SparseMatrix<double>::InnerIterator it(x,k); it; ++it){
                h_x.push_back(it.value());
            }
        }


        // From cannonical form to moment form
        auto J = (matB.transpose()*matB).toDense();
        std::cout << "J " << J.rows() << " " << J.cols() << std::endl;
        Sigma =  2.0*(J(0,0))/(nearest_neighborhood.size()-3)*Eigen::Matrix<double,6,6>(AtPA).inverse();
        // marginalize out only translation
        covPose = Sigma.block<3,3>(0,0);


        std::cout << "covPose " << std::endl;
        std::cout << covPose << std::endl;

        _pose.px += h_x[0];
        _pose.py += h_x[1];
        _pose.pz += h_x[2];
        _pose.om += h_x[3];
        _pose.fi += h_x[4];
        _pose.ka += h_x[5];

        pose = Sophus::SE3f(affine_matrix_from_pose_tait_bryan(_pose).matrix().cast<float>());
    }
    if(ImGui::Button("implicit")) {
        computeNN();
        std::vector<Eigen::Triplet<double>> tripletListA;
        std::vector<Eigen::Triplet<double>> tripletListP;
        std::vector<Eigen::Triplet<double>> tripletListB;

        TaitBryanPose _pose = pose_tait_bryan_from_affine_matrix(Eigen::Affine3d(pose.matrix().cast<double>()));

        Eigen::MatrixXd Err(3,nearest_neighborhood.size());
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> hessian_sum;
        hessian_sum.setZero();
        for (int i = 0; i < nearest_neighborhood.size(); i++)
        {
            const auto & nn = nearest_neighborhood[i];
            // todo check!!
            Eigen::Vector3d p_1 = nn.psL.cast<double>();
            Eigen::Vector3d p_l = nn.pm.cast<double>();

            double delta_x;
            double delta_y;
            double delta_z;
            point_to_point_source_to_landmark_tait_bryan_wc(delta_x, delta_y, delta_z,
                                                            _pose.px,_pose.py,_pose.pz,
                                                            _pose.om,_pose.fi,_pose.ka,
                                                            p_1.x(),p_1.y(),p_1.z(),
                                                            p_l.x(),p_l.y(),p_l.z());
            Eigen::Matrix<double, 3, 6, Eigen::RowMajor> jacobian;
            point_to_point_source_to_landmark_tait_bryan_wc_jacobian(jacobian,
                                                                     _pose.px,_pose.py,_pose.pz,
                                                                     _pose.om,_pose.fi,_pose.ka,
                                                                     p_1.x(),p_1.y(),p_1.z());
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> hessian;

            point_to_point_source_to_landmark_tait_bryan_wc_hessian(hessian,
                                                                    _pose.px,_pose.py,_pose.pz,
                                                                    _pose.om,_pose.fi,_pose.ka,
                                                                    p_1.x(),p_1.y(),p_1.z(),
                                                                    p_l.x(),p_l.y(),p_l.z());
            hessian_sum+=hessian;
            Err(0,i) = delta_x;
            Err(1,i) = delta_y;
            Err(2,i) = delta_z;

            int ir = tripletListB.size();
            int ic_1 = 0;
            tripletListA.emplace_back(ir     , ic_1    , -jacobian(0,0));
            tripletListA.emplace_back(ir     , ic_1 + 1, -jacobian(0,1));
            tripletListA.emplace_back(ir     , ic_1 + 2, -jacobian(0,2));
            tripletListA.emplace_back(ir     , ic_1 + 3, -jacobian(0,3));
            tripletListA.emplace_back(ir     , ic_1 + 4, -jacobian(0,4));
            tripletListA.emplace_back(ir     , ic_1 + 5, -jacobian(0,5));

            tripletListA.emplace_back(ir + 1 , ic_1    , -jacobian(1,0));
            tripletListA.emplace_back(ir + 1 , ic_1 + 1, -jacobian(1,1));
            tripletListA.emplace_back(ir + 1 , ic_1 + 2, -jacobian(1,2));
            tripletListA.emplace_back(ir + 1 , ic_1 + 3, -jacobian(1,3));
            tripletListA.emplace_back(ir + 1 , ic_1 + 4, -jacobian(1,4));
            tripletListA.emplace_back(ir + 1 , ic_1 + 5, -jacobian(1,5));

            tripletListA.emplace_back(ir + 2 , ic_1    , -jacobian(2,0));
            tripletListA.emplace_back(ir + 2 , ic_1 + 1, -jacobian(2,1));
            tripletListA.emplace_back(ir + 2 , ic_1 + 2, -jacobian(2,2));
            tripletListA.emplace_back(ir + 2 , ic_1 + 3, -jacobian(2,3));
            tripletListA.emplace_back(ir + 2 , ic_1 + 4, -jacobian(2,4));
            tripletListA.emplace_back(ir + 2 , ic_1 + 5, -jacobian(2,5));

            tripletListP.emplace_back(ir    , ir    ,  1);
            tripletListP.emplace_back(ir + 1, ir + 1,  1);
            tripletListP.emplace_back(ir + 2, ir + 2,  1);

            tripletListB.emplace_back(ir    , 0,  delta_x);
            tripletListB.emplace_back(ir + 1, 0,  delta_y);
            tripletListB.emplace_back(ir + 2, 0,  delta_z);
        }

        Eigen::SparseMatrix<double> matA(tripletListB.size(), 6);
        Eigen::SparseMatrix<double> matP(tripletListB.size(), tripletListB.size());
        Eigen::SparseMatrix<double> matB(tripletListB.size(), 1);

        matA.setFromTriplets(tripletListA.begin(), tripletListA.end());
        matP.setFromTriplets(tripletListP.begin(), tripletListP.end());
        matB.setFromTriplets(tripletListB.begin(), tripletListB.end());

        Eigen::SparseMatrix<double> AtPA(6, 6);
        Eigen::SparseMatrix<double> AtPB(6, 1);

        {
            Eigen::SparseMatrix<double> AtP = matA.transpose() * matP;
//            AtPA = (AtP) * matA;
            AtPA = hessian_sum.sparseView();
            AtPB = (AtP) * matB;
        }

        std::cout << "AtPA.size: " << AtPA.size() << std::endl;
        std::cout << "AtPB.size: " << AtPB.size() << std::endl;

        std::cout << "start solving AtPA=AtPB" << std::endl;
        Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver(AtPA);

        std::cout << "x = solver.solve(AtPB)" << std::endl;
        Eigen::SparseMatrix<double> x = solver.solve(AtPB);

        std::vector<double> h_x;

        for (int k=0; k<x.outerSize(); ++k){
            for (Eigen::SparseMatrix<double>::InnerIterator it(x,k); it; ++it){
                h_x.push_back(it.value());
            }
        }



        std::cout << "covPose " << std::endl;
        std::cout << covPose << std::endl;

        _pose.px += h_x[0];
        _pose.py += h_x[1];
        _pose.pz += h_x[2];
        _pose.om += h_x[3];
        _pose.fi += h_x[4];
        _pose.ka += h_x[5];

        pose = Sophus::SE3f(affine_matrix_from_pose_tait_bryan(_pose).matrix().cast<float>());

        std::vector<PointMeanCov> data_pi;
        std::vector<PointMeanCov> model_qi;

        Eigen::Matrix3d cov = 1.0*Eigen::Matrix3d::Identity();

        for (const auto &nn : nearest_neighborhood)
        {
            PointMeanCov pi;
            //pi.mean = nn.psL.cast<double>();
            pi.coords =  nn.psL.cast<double>();
            pi.cov = cov;

            PointMeanCov qi;
            qi.coords = nn.pm.cast<double>();
            qi.cov = cov;
            data_pi.push_back(pi);
            model_qi.push_back(qi);
        }
        Eigen::MatrixXd ICP_COV;
        calculate_ICP_COV(data_pi, model_qi,Eigen::Affine3d(pose.matrix().cast<double>()), ICP_COV);
        Sigma = ICP_COV;

    }
    ImGui::InputFloat("scale", &scale);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glutSwapBuffers();
    glutPostRedisplay();
}

void mouse(int glut_button, int state, int x, int y) {
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
    int button = -1;
    if (glut_button == GLUT_LEFT_BUTTON) button = 0;
    if (glut_button == GLUT_RIGHT_BUTTON) button = 1;
    if (glut_button == GLUT_MIDDLE_BUTTON) button = 2;
    if (button != -1 && state == GLUT_DOWN)
        io.MouseDown[button] = true;
    if (button != -1 && state == GLUT_UP)
        io.MouseDown[button] = false;

    if (!io.WantCaptureMouse)
    {
        if (state == GLUT_DOWN) {
            mouse_buttons |= 1 << glut_button;
        } else if (state == GLUT_UP) {
            mouse_buttons = 0;
        }
        mouse_old_x = x;
        mouse_old_y = y;
    }

}

void motion(int x, int y) {
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);

    if (!io.WantCaptureMouse)
    {
        float dx, dy;
        dx = (float) (x - mouse_old_x);
        dy = (float) (y - mouse_old_y);
        gui_mouse_down = mouse_buttons>0;
        if (mouse_buttons & 1) {
            rotate_x += dy * 0.2f;
            rotate_y += dx * 0.2f;
        } else if (mouse_buttons & 4) {
            translate_z += dy * 0.05f;
        } else if (mouse_buttons & 3) {
            translate_x += dx * 0.05f;
            translate_y -= dy * 0.05f;
        }
        mouse_old_x = x;
        mouse_old_y = y;
    }
    glutPostRedisplay();
}

void reshape(int w, int h) {
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) w / (GLfloat) h, 0.01, 10000.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

bool initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    windows_handle = glutCreateWindow("perspective_camera_ba");
    glutDisplayFunc(display);
    glutMotionFunc(motion);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) window_width / (GLfloat) window_height, 0.01,
                   10000.0);
    glutReshapeFunc(reshape);
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL2_Init();
    return true;
}
void processSpecialKeys(unsigned char key, int x, int y) {
    Sophus::Vector6f update = Sophus::Vector6f::Zero();
    if (key == 'w'){
        update[0] = 0.1;
    }
    if (key == 's'){
        update[0] = -0.1;
    }
    if (key == 'a'){
        update[1] = -0.1;
    }
    if (key == 'd'){
        update[1] = 0.1;
    }
    if (key == 'q'){
        update[5] = -0.1;
    }
    if (key == 'e') {
        update[5] = 0.1;
    }
    pose =    pose * Sophus::SE3f::exp(update) ;
}