#include <iostream>
#include <thread>


#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

#include "structs.h"
#include "pf.h"


#include <iostream>
#include <thread>

#include <GL/freeglut.h>
#include <mutex>
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/common/transforms.h>

#include "structs.h"
#include "pf.h"


bool imgui_log_trajectory{false};
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
pcl::PointCloud<pcl::PointXYZI> input_data_obstacles;
pcl::PointCloud<pcl::PointXYZI> input_data_floor;



HostDeviceData host_device_data;
pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_traversability(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr measurment(new pcl::PointCloud<pcl::PointXYZ>);

Eigen::Affine3d best_pose (Eigen::Affine3d::Identity());
int imgui_particles_count = host_device_data.max_particles;
void display();
void reshape(int w, int h);
void mouse(int glut_button, int state, int x, int y);
void motion(int x, int y);
bool initGL(int *argc, char **argv);
struct imgui_data{
    int resample_type = 1;
    int sensor_type= 1;
}imgui_data;

int gl_main(int argc, char *argv[]){
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMainLoop();
    return 0;
}

void Bresenham3D(int ix_start, int iy_start, int iz_start, int ix_stop, int iy_stop, int iz_stop){
    int dx = std::abs(ix_stop - ix_start);
    int dy = std::abs(iy_stop - iy_start);
    int dz = std::abs(iz_stop - iz_start);
    int xs= (ix_stop > ix_start) ? 1 : -1;
    int ys= (iy_stop > iy_start) ? 1 : -1;
    int zs= (iz_stop > iz_start) ? 1 : -1;
    if (dx >= dy && dx >= dz){
        int p1 = 2.0 * dy - dx;
        int p2 = 2.0 * dz - dx;
        while (ix_start != ix_stop){
            ix_start+= xs;
            if (p1 >= 0){
                iy_start += ys;
                p1 -= 2 * dx;
            }
            if (p2 >= 0) {
                iz_start += zs;
                p2 -= 2 * dx;
            }
            p1 += 2 * dy;
            p2 += 2 * dz;
            std::cout << ix_start << ", " << iy_start << ", " << iz_start << std::endl;
        }
    }
    else if (dy >= dx && dy >= dz){
        int p1 = 2 * dx - dy;
        int p2 = 2 * dz - dy;
        while (iy_start != iy_stop) {
            iy_start += ys;
            if (p1 >= 0) {
                ix_start += xs;
                p1 -= 2 * dy;
            }
            if (p2 >= 0) {
                iz_start += zs;
                p2 -= 2 * dy;
            }
            p1 += 2 * dx;
            p2 += 2 * dz;
            std::cout << ix_start << ", " << iy_start << ", " << iz_start << std::endl;
        }
    }
    else{
        int p1 = 2 * dy - dz;
        int p2 = 2 * dx - dz;
        while (iz_start != iz_stop) {
            iz_start += zs;
            if (p1 >= 0) {
                iy_start += ys;
                p1 -= 2 * dz;
            }
            if (p2 >= 0) {
                ix_start += xs;
                p2 -= 2 * dz;
            }
            p1 += 2 * dy;
            p2 += 2 * dx;
            std::cout << ix_start << ", " << iy_start << ", " << iz_start << std::endl;
        }
    }
}

int main (int argc, char *argv[])
{
//
//    Bresenham3D(-1, 1, 1, 5, 3, -1);
//
//    return 0;
    Eigen::Matrix4d m;
    m << 0.960016,  0.279946,         0,    12.329,
        -0.279946,  0.960016,         0,  -16.3503,
                0,         0,         1,         0,
                0,         0,         0,         1;

    best_pose = Eigen::Affine3d (m);
    std::string path{"/home/michal/ros2_ws/src/pf_localization/src/localization_gl"};
    pcl::io::loadPCDFile(path+"/data/p7p_cloud_clean_2d.pcd", *map_cloud);
    pcl::io::loadPCDFile(path+"/data/p7p_cloud_clean_2d_traversability.pcd", *map_traversability);
    pcl::io::loadPCDFile(path+"/data/frame_248.pcd", *measurment);
    host_device_data.host_map.resize(map_cloud->size()+map_traversability->size());

    std::transform(map_cloud->begin(),map_cloud->end(), host_device_data.host_map.begin(),
            //[](const pcl::PointXYZI&p){return Point{p.x,p.y,p.z, PointType::obstacle };});
                   [](const pcl::PointXYZI&p){return Point{p.x,p.y,0, PointType::obstacle };});

    std::transform(map_traversability->begin(),map_traversability->end(), host_device_data.host_map.begin()+map_cloud->size()-1,
            //[](const pcl::PointXYZI&p){return Point{p.x,p.y,p.z, PointType::obstacle };});
                   [](const pcl::PointXYZ&p){return Point{p.x,p.y,0, PointType::ground };});

    if (map_traversability) {
        host_device_data.host_travesability.resize(map_traversability->size());
        std::transform(map_traversability->begin(), map_traversability->end(), host_device_data.host_travesability.begin(),
                //[](const pcl::PointXYZI&p){return Point{p.x,p.y,p.z, PointType::obstacle };});
                       [](const pcl::PointXYZ &p) { return Point{p.x, p.y, 0, PointType::obstacle}; });
    }
    initialize_host_device_data(host_device_data);

    gl_main(argc, argv);
    glutDestroyWindow(windows_handle);
//    gl_th.join();

}

void display() {

    ImGuiIO& io = ImGui::GetIO();
    if (io.KeysDown['Q'])
    {
        Eigen::Affine3d  r {Eigen::Affine3d::Identity()};
        r.rotate(Eigen::AngleAxisd (M_PI/180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix());
        best_pose = best_pose *r ;
    }
    else if (io.KeysDown['E'])
    {
        Eigen::Affine3d  r {Eigen::Affine3d::Identity()};
        r.rotate(Eigen::AngleAxisd (-M_PI/180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix());
        best_pose =  best_pose * r;
    }
    else if (io.KeysDown['W'])
    {
        Eigen::Affine3d  r {Eigen::Affine3d::Identity()};
        Eigen::Vector3d v {0.1,0,0};
        r.translation() = best_pose.rotation() *v;
        best_pose = r *best_pose;
    }
    else if (io.KeysDown['S'])
    {
        Eigen::Affine3d  r {Eigen::Affine3d::Identity()};
        Eigen::Vector3d v {-0.1,0,0};
        r.translation() = best_pose.rotation() *v;
        best_pose =  r * best_pose;
    }


    std::vector<Point> points;

    points.resize(measurment->size());
    Pose pose_update;
    {
        std::transform(measurment->begin(), measurment->end(), points.begin(),
                       [](const pcl::PointXYZ&p){return Point{p.x,p.y,0, PointType::obstacle };});
//        std::transform(input_data_floor.begin(), input_data_floor.end(), points.begin()+input_data_obstacles.size()-1,
//                       [](const pcl::PointXYZI&p){return Point{p.x,p.y,-1, PointType::ground };});


    }
    // add
    for (auto &p : *measurment){
        Eigen::Vector3f pn = p.getVector3fMap();
        float len = pn.norm();
        pn = pn / len;
        for (float i =0; i < len ; i+=0.25){
            points.push_back(Point{pn.x()*i,pn.y()*i,pn.z()*i, PointType::ground });
        }
    }

    double timestamp = 0;
    //std::cout << pose_update.p.x <<"\t" << pose_update.p.y <<"\t" << pose_update.o.z_angle_rad << std::endl;
    //particle_filter_step(host_device_data, pose_update, points);
    host_device_data.particles.resize(1);
    host_device_data.particles[0].pose = MatrixToPose(best_pose.matrix());
    compute_overlaps(host_device_data, points, false);

    std::cout <<host_device_data.particles[0].overlap << std::endl;
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
        if (p.label == obstacle) {
            glColor3f(1.0f, 1.0f, 0.0f);
        }else{
            glColor3f(1.0f, 1.0f, 1.0f);
        }
        glVertex3f(p.x, p.y, p.z);
    } glEnd();
    ///////////////////////

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();
    ImGui::Begin("Demo Window1");
    ImGui::InputInt("number of samples", & host_device_data.max_particles, 100,1000);
    ImGui::InputInt("number of exploration", & host_device_data.number_of_replicatations_motion_model, 100,1000);

    const char* const items1[] = {"Velodyne", "Livox", "Failed"};
    ImGui::Combo("Sensor Type" , &imgui_data.sensor_type, items1 ,3);

    const char* const items2[] = {"Standard", "Stratified resampling"};
    ImGui::Combo("Resample Algorithm" , &imgui_data.resample_type, items2 ,2);

    ImGui::Text("Elapsed %.3f", host_device_data.step_time);

    if (ImGui::Button("rot")){
        std::cout << best_pose.matrix() << std::endl;
    }
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
