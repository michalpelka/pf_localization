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
#include <pcl/common/transforms.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

#include "structs.h"
#include "pf.h"

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

float imgui_co_size{1.0f};
bool imgui_draw_co{true};

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
    pcl::transformPointCloud(filtered, input_data, transform);
}
void odometry_callback(const nav_msgs::Odometry::ConstPtr odo)
{
    Eigen::Affine3d transform(Eigen::Affine3d::Identity());
    std::lock_guard<std::mutex> lck(mtx_input_data);
    transform.translation() = Eigen::Vector3d{odo->pose.pose.position.x,odo->pose.pose.position.y,odo->pose.pose.position.z};
    transform.rotate(Eigen::Quaterniond{odo->pose.pose.orientation.w,odo->pose.pose.orientation.x,
                                        odo->pose.pose.orientation.y,odo->pose.pose.orientation.z});
}
int main (int argc, char *argv[])
{
	//ToDo data.host_map -> load map from file
	pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::io::loadPCDFile("/media/michal/ext/p7p2_2d_clean.pcd", *map_cloud);

    host_device_data.host_map.resize(map_cloud->size());
    std::transform(map_cloud->begin(),map_cloud->end(), host_device_data.host_map.begin(),
                   [](const pcl::PointXYZI&p){return Point{p.x,p.y,p.z, PointType::obstacle };});

	initialize_host_device_data(host_device_data);

    map_cloud = nullptr;
    ros::init(argc, argv, "localization_gl");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/velodyne_points", 1, pointcloud_callback);

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
        odometry_increment = odometry * last_odometry.inverse();
        last_odometry = odometry;
        points.resize(input_data.size());
        std::transform(input_data.begin(),input_data.end(), points.begin(),
                       [](const pcl::PointXYZI&p){return Point{p.x,p.y,p.z, PointType::obstacle };});
        pose_update = get_pose(odometry_increment);
    }
	particle_filter_step(host_device_data, pose_update, points);

	if(host_device_data.particles.size()> 0){
		std::cout << "best pose" << std::endl;
		Eigen::Affine3d m = get_matrix(host_device_data.particles[0].pose);
		std::cout << m.matrix() << std::endl;
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

    if (imgui_draw_co) {
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(imgui_co_size, 0.0f, 0.0f);

        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, imgui_co_size, 0.0f);

        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, imgui_co_size);
        glEnd();
    }

    glBegin(GL_POINTS);
    for (int i=0; i< points.size();i++)
    {
        const auto &p = points[i];
        glVertex3f(p.x, p.y,p.z);
    }
    glEnd();


    glColor3f(0.7,0.7,0.7);
    glBegin(GL_POINTS);
    for(size_t i = 0 ; i < host_device_data.particle_filter_initial_guesses.size(); i++){
    	glVertex3f(host_device_data.particle_filter_initial_guesses[i].pose.p.x,
    			host_device_data.particle_filter_initial_guesses[i].pose.p.y,
    			host_device_data.particle_filter_initial_guesses[i].pose.p.z);
    }
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
    ImGui::Text("Text");
    ImGui::SliderFloat("co_size", &imgui_co_size, 0.1f, 10.0f, "co_size: %.0f" );
    ImGui::Checkbox("co_draw", &imgui_draw_co);
    if(ImGui::Button("foo"))
    {
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
