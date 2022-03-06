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

const unsigned int window_width = 1920;
const unsigned int window_height = 1080;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -60.0;
float translate_x=-44;
float translate_y = -10.0;
bool gui_mouse_down{false};
int windows_handle = -1;

void display();
void reshape(int w, int h);
void mouse(int glut_button, int state, int x, int y);
void motion(int x, int y);
bool initGL(int *argc, char **argv);

std::map<double,Sophus::SE3d> trajectorygt;
std::vector<std::map<double,Sophus::SE3d>> trajectory2;


int gl_main(int argc, char *argv[]){
    initGL(&argc, argv);
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMainLoop();
}
std::map<double,Sophus::SE3d> loadTrajectory(const std::string& fn, double off= 0){
    std::map<double,Sophus::SE3d> trajectory;
    std::ifstream fss(fn);
    while (!fss.eof()){
        double ts;
        Sophus::Vector6d  t;
        fss >> ts;
        fss >> t[0];
        fss >> t[1];
        fss >> t[2];
        fss >> t[3];
        fss >> t[4];
        fss >> t[5];
        ts += off;
        trajectory[ts] = Sophus::SE3d::exp(t);
        trajectory[ts].translation().z() = 0;
        trajectory[ts].normalize();
    }
    return trajectory;
}


int main (int argc, char *argv[])
{
    trajectorygt = loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_gt.txt",0);
//    trajectory2 = loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_filter_calib.txt");
//    trajectory2 = loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory2/trajectory_velo_100k.txt");

//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_lvx/trajectory_filter_5k.txt"));
//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_lvx/trajectory_filter_10k.txt"));
    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_lvx/trajectory_filter_50k.txt"));
//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_lvx/trajectory_filter_100k.txt"));

//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory2/trajectory_velo_5k.txt"));
//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory2/trajectory_velo_10k.txt"));
//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory2/trajectory_velo_50k.txt"));
//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory2/trajectory_velo_100k.txt"));

//trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_lvx_non_calib/trajectory_filter_5k.txt"));
//trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_lvx_non_calib/trajectory_filter_10k.txt"));
//trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_lvx_non_calib/trajectory_filter_50k.txt"));
//


//    trajectory2.push_back(loadTrajectory("/media/michal/ext/garaz2/trajectory/trajectory_wheel.txt"));


    std::thread gl_th(gl_main, argc, argv);
    gl_th.join();

}

void display() {
    ImGuiIO& io = ImGui::GetIO();
    glViewport(0, 0, (GLsizei)io.DisplaySize.x, (GLsizei)io.DisplaySize.y);
    glClearColor(1.0,1.0,1.0,1.0);
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

    for (int i=-200;i<200; i++){
        glLineWidth(1);
        glBegin(GL_LINES);
        glColor3f(0.8f, 0.8f, 0.8f);
        glVertex3f(i,-100.0, 0.0f);
        glVertex3f(i, 100.0, 0.0f);
        glEnd();
    }

    for (int i=-200;i<200; i++){
        glLineWidth(1);
        glBegin(GL_LINES);
        glColor3f(0.8f, 0.8f, 0.8f);
        glVertex3f(-100.0,i, 0.0f);
        glVertex3f( 100.0,i, 0.0f);
        glEnd();
    }


    glLineWidth(2);
    glBegin(GL_LINE_STRIP);
    glColor3f(0.0f, 1.0f, 0.0f);
    for (const auto &t : trajectorygt){
        const auto &dr = t.first;
        const auto &tr = t.second;
        glVertex3f(tr.translation().x(),tr.translation().y(),tr.translation().z());
    }
    glEnd();

    glBegin(GL_LINE_STRIP);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (int i =0; i < trajectory2.size(); i++) {
        const auto & trajectory = trajectory2[i];

        for (const auto &t : trajectory) {
            const auto &dr = t.first;
            const auto &tr = t.second;
            glVertex3f(tr.translation().x(), tr.translation().y(), tr.translation().z());
        }
    }
    glEnd();
    glLineWidth(1);
    glBegin(GL_LINES);
    glColor3f(0.5f, 0.5f, 0.5f);
    std::vector<double> ate_f1n;
    for (int i =0; i < trajectory2.size(); i++)
    {
        const auto & trajectory = trajectory2[i];

        for (const auto &p : trajectory){
            const auto &timestamp = p.first;
            const auto &tr = p.second;
            const auto it = trajectorygt.lower_bound(timestamp);
            const double err = std::abs(it->first - timestamp);
            if (err < 0.5) {

                auto P = tr;
                auto Q = it->second;
                double ate = (Q.inverse() * P).translation().squaredNorm();
                // if (ate < 3.0){
                glVertex3f(tr.translation().x(), tr.translation().y(), tr.translation().z());
                glVertex3f(it->second.translation().x(), it->second.translation().y(), it->second.translation().z());
                ate_f1n.push_back(ate);
                //}
//            std::cout << P.matrix() << std::endl;
//            std::cout << Q.matrix() << std::endl;
//            std::cout << "=================" << std::endl;
            }
        }
        double ate = std::sqrt(std::accumulate(ate_f1n.begin(),ate_f1n.end(),0.0)/ate_f1n.size());
        std::cout << i << " ATE " << ate << std::endl;
    }


    glEnd();
    //ImGui::Render();
    //ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
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
