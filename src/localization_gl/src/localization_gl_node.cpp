#include <iostream>
#include <thread>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

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

std::vector<catoptric_livox::Mirror> mirrors;
std::deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> aggregated_pointcloud_odom;
ros::Publisher aggregated_livox;

const unsigned int window_width = 1920;
const unsigned int window_height = 1080;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;
float translate_x, translate_y = 0.0;
bool gui_mouse_down{false};
int windows_handle = -1;
std::mutex mtx_input_data;
pcl::PointCloud<pcl::PointXYZI> input_data;
Eigen::Affine3d odometry {Eigen::Affine3d::Identity()};
Eigen::Affine3d last_odometry {Eigen::Affine3d::Identity()};

Eigen::Affine3d odometry_increment{Eigen::Affine3d::Identity()};
HostDeviceData host_device_data;

void display();
void reshape(int w, int h);
void mouse(int glut_button, int state, int x, int y);
void motion(int x, int y);
bool initGL(int *argc, char **argv);


int gl_main(int argc, char *argv[]){
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMainLoop();
}

void pointcloud_callback(const pcl::PointCloud<pcl::PointXYZI>::Ptr  msg){
    Eigen::Affine3f transform(Eigen::Affine3f::Identity());
    transform.translation().x() = 0.2;
    transform.rotate(Eigen::Quaternionf(0.96593,0,0,-0.25882));
    pcl::ApproximateVoxelGrid<pcl::PointXYZI> filter;
    filter.setInputCloud(msg);
    pcl::PointCloud<pcl::PointXYZI> filtered;
    filter.setLeafSize(0.25,0.25,0.25);
    filter.filter(filtered);
    std::lock_guard<std::mutex> lck(mtx_input_data);

    pcl::PointCloud<pcl::PointXYZI> filtered_cut_z;
    filtered_cut_z.reserve(filtered.size());
    for(const auto &ref:filtered){
    	if(fabs(ref.z)< 0.3){
    		filtered_cut_z.push_back(ref);
    	}
    }

    pcl::transformPointCloud(filtered_cut_z, input_data, transform);
}

void pointcloud_callback_lvx(const livox_ros_driver::CustomMsg::ConstPtr& msg){
    pcl::PointCloud<pcl::PointXYZI> cloud = catoptric_livox::converterLivoxMirror(mirrors, msg);
    std::cout << "cloud " << cloud.size() << std::endl;
    pcl::transformPointCloud(cloud, cloud, odometry.matrix().cast<float>());
    aggregated_pointcloud_odom.push_back(cloud.makeShared());

    if (aggregated_pointcloud_odom.size()>10){
        aggregated_pointcloud_odom.pop_front();
    }
    Eigen::Affine3f transform(Eigen::Affine3f::Identity());
    transform.translation().x() = 0.2;


    Eigen::Affine3f odom_inv = odometry.inverse().cast<float>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr aggregated(new  pcl::PointCloud<pcl::PointXYZI>);
    for (auto &pc : aggregated_pointcloud_odom) {
        for (int i = 0; i < pc->size(); i += 1) {
            pcl::PointXYZI pt;
            pt.getArray3fMap() = odom_inv* transform *(*pc)[i].getArray3fMap();
            pt.intensity = (*pc)[i].intensity;
            aggregated->push_back(pt);
        }
    }

    pcl::ApproximateVoxelGrid<pcl::PointXYZI> filter;
    filter.setInputCloud(aggregated);
    pcl::PointCloud<pcl::PointXYZI> filtered;
    filter.setLeafSize(0.25,0.25,0.25);
    filter.filter(filtered);

    pcl::PointCloud<pcl::PointXYZI> filtered_cut_z;
    filtered_cut_z.reserve(filtered.size());
    for(const auto &ref:filtered){
        if(fabs(ref.z-0.8)< 0.3 && ref.x*ref.x + ref.y*ref.y > 1.0){
            filtered_cut_z.push_back(ref);
        }
    }

    input_data = filtered_cut_z;
    aggregated->header.frame_id ="base_link";
    pcl_conversions::toPCL(ros::Time::now(), input_data.header.stamp);
    if(aggregated_livox.getNumSubscribers()>0){
        aggregated_livox.publish(aggregated);
    }
    //pcl::io::savePCDFile("/tmp/test_pcd.pcd", *aggregated);
}

void odometry_callback(const nav_msgs::Odometry::ConstPtr odo)
{
    Eigen::Affine3d transform(Eigen::Affine3d::Identity());

    transform.translation() = Eigen::Vector3d{odo->pose.pose.position.x,odo->pose.pose.position.y,odo->pose.pose.position.z};
    transform.rotate(Eigen::Quaterniond{odo->pose.pose.orientation.w,odo->pose.pose.orientation.x,
                                        odo->pose.pose.orientation.y,odo->pose.pose.orientation.z});
    std::lock_guard<std::mutex> lck(mtx_input_data);
    odometry = transform;
}
int main (int argc, char *argv[])
{

	//ToDo data.host_map -> load map from file
	pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile("//media/michal/ext/p7p2_2d_clean.pcd", *map_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_traversability(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("//media/michal/ext/p7p2_2d_traversability.pcd", *map_traversability);

    host_device_data.host_map.resize(map_cloud->size());
    std::transform(map_cloud->begin(),map_cloud->end(), host_device_data.host_map.begin(),
                   //[](const pcl::PointXYZI&p){return Point{p.x,p.y,p.z, PointType::obstacle };});
    		[](const pcl::PointXYZI&p){return Point{p.x,p.y,0, PointType::obstacle };});

    if (map_traversability) {
        host_device_data.host_travesability.resize(map_traversability->size());
        std::transform(map_traversability->begin(), map_traversability->end(), host_device_data.host_travesability.begin(),
                //[](const pcl::PointXYZI&p){return Point{p.x,p.y,p.z, PointType::obstacle };});
                       [](const pcl::PointXYZ &p) { return Point{p.x, p.y, 0, PointType::obstacle}; });
    }
	initialize_host_device_data(host_device_data);

    mirrors = catoptric_livox::loadMirrorFromPLY("/home/michal/code/livox_ws/src/test_lustro_hex_mid70.ply");

    Sophus::SE3d intruments_lever_arm;
    catoptric_livox::loadCFG(mirrors, intruments_lever_arm, "/home/michal/code/livox_ws/src/pf_localization/test_lustro_hex_mid_70_calib.ini");

    map_cloud = nullptr;
    ros::init(argc, argv, "localization_gl");
    ros::NodeHandle n;
    //ros::Subscriber sub1 = n.subscribe("/velodyne_points", 1, pointcloud_callback);
    ros::Subscriber sub1 = n.subscribe("/livox/lidar", 1, pointcloud_callback_lvx);

    ros::Subscriber sub2 = n.subscribe("/odometry/filtered", 1, odometry_callback);
    aggregated_livox = n.advertise<pcl::PointCloud<pcl::PointXYZI>>("/livox/lidar_mirror",1);

    std::thread gl_th(gl_main, argc, argv);
    ros::spin();
    glutDestroyWindow(windows_handle);
    gl_th.join();

}

void display() {
	std::vector<Point> points;
	Pose pose_update;
    {
        std::lock_guard<std::mutex> lck(mtx_input_data);
        odometry_increment =  last_odometry.inverse()* odometry;
        last_odometry = odometry;
        points.resize(input_data.size());
        std::transform(input_data.begin(),input_data.end(), points.begin(),
                       [](const pcl::PointXYZI&p){return Point{p.x,p.y,0, PointType::obstacle };});
        pose_update = get_pose(odometry_increment);

    }
    //std::cout << pose_update.p.x <<"\t" << pose_update.p.y <<"\t" << pose_update.o.z_angle_rad << std::endl;
	particle_filter_step(host_device_data, pose_update, points);
    Eigen::Affine3d best_pose;
    if(host_device_data.particles.size()> 0){
		std::cout << "best pose" << std::endl;
        best_pose = get_matrix(host_device_data.particles[0].pose);
		std::cout << best_pose.matrix() << std::endl;
		std::cout << "w: " <<host_device_data.particles[0].W << " " << host_device_data.particles[0].nW << std::endl;
	}


    if (!ros::ok()){
        return;
    }

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
    glColor3f(0.6, 1, 0.1);
    glBegin(GL_POINTS);
    for (int i=0; i< points.size();i++)
    {
        const auto &p = points[i];
        const Eigen::Vector3d pp{p.x, p.y,p.z};
        const Eigen::Vector3d ppt = best_pose * pp;
        glVertex3f(ppt.x(), ppt.y(), ppt.z());
    }
    glEnd();
    glPointSize(1);

    glColor3f(0.7,0.7,0.7);
    glBegin(GL_POINTS);
#ifdef ROTATION_TB
    for(size_t i = 0 ; i < host_device_data.particles.size(); i++){
    	glVertex3f(host_device_data.particles[i].pose.p.x,
    			host_device_data.particles[i].pose.p.y,
    			host_device_data.particles[i].pose.p.z);
    }
#endif
#ifdef ROTATION_SE3
    for(size_t i = 0 ; i < host_device_data.particles.size(); i++){
        glVertex3f(host_device_data.particles[i].pose.p.x(),
                   host_device_data.particles[i].pose.p.y(),
                   host_device_data.particles[i].pose.p.z());
    }
#endif
    glEnd();

    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 0.0f);
    //ToDo data -> render map from file
    for (const auto &p : host_device_data.host_map)
    {
        glVertex3f(p.x, p.y,p.z);
    } glEnd();
    ///////////////////////

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::Begin("Demo Window1");
    ImGui::Text("Elapsed %.3f", host_device_data.step_time);


    if(ImGui::Button("reset"))
    {
        host_device_data.particles.clear();
    	std::vector<Particle> exploration_particles = choose_random_exploration_particles(host_device_data);
        host_device_data.particles.clear();
    	host_device_data.particles.insert(host_device_data.particles.end(), exploration_particles.begin(), exploration_particles.end());

        std::cout << "bar\n";
    }

    ImGui::End();

    ImGui::Begin("Demo Window2");
    ImGui::Text("Text");
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
