#include "mirror_converter.h"
pcl::PointCloud<pcl::PointXYZI> catoptric_livox::converterLivoxMirror(const std::vector<catoptric_livox::Mirror>& mirrors, const livox_ros_driver::CustomMsg::ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.reserve(msg->points.size());

    Eigen::Matrix4f m_rot = Eigen::Matrix4f::Identity();
    m_rot.block<3,3>(0,0) = Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitY()).toRotationMatrix();
    Eigen::Matrix4f m_rot2 = Eigen::Matrix4f::Identity();
    m_rot2.block<3,3>(0,0) = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    for (const livox_ros_driver::CustomPoint &livox_p : msg->points){
        const Eigen::Vector3d ds (livox_p.x,livox_p.y,livox_p.z);
        int mirror_id =-1;
        for (int i = 0; i < mirrors.size(); i++) {
            if (mirrors[i].checkIfRayIntersectMirror(Eigen::Vector3d(0, 0, 0),
                                                     Eigen::Vector3d(ds))) {
                mirror_id = i;
                break;
            }
        }
        if (mirror_id!=-1) {
            Eigen::Vector3f c = catoptric_livox::mirror_colors[mirror_id];
            Eigen::Vector3d dir{ds};
            const double l = dir.norm();
            dir = dir / l;
//                Eigen::Vector4d plane = catoptric_livox::getPlaneCoefFromSE3(mirrors[mirror_id].getTransformationOptimized());
            Eigen::Vector4d plane = mirrors[mirror_id].getABCDofPlane();

            Eigen::Vector4d p = Eigen::Vector4d::Ones();
            p.head(3)= catoptric_livox::getMirroredRay(dir, l, plane);
            pcl::PointXYZI pp;
            pp.getArray4fMap() =m_rot2 * m_rot * p.cast<float>();
            pp.intensity = livox_p.reflectivity;
            cloud.push_back(pp);
        }
    }
    auto h = msg->header;
    h.frame_id = "base_link";;
    //h.stamp = ros::Time::now();
    pcl_conversions::toPCL(h, cloud.header);
    //std::cout << msg->header << std::endl;
    cloud.header.frame_id = "livox_mirror";
    return cloud;
}