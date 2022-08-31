from sympy import *
import sys
sys.path.insert(1, '.')
from tait_bryan_R_utils import *
#from rodrigues_R_utils import *
#from quaternion_R_utils import *

x_L, y_L, z_L = symbols('x_L y_L z_L')
x_s, y_s, z_s = symbols('x_s y_s z_s')
tx, ty, tz = symbols('tx ty tz')
om, fi, ka = symbols('om fi ka')
#sx, sy, sz = symbols('sx sy sz')
#q0, q1, q2, q3 = symbols('q0 q1 q2 q3')

position_symbols = [tx, ty, tz]
orientation_symbols = [om, fi, ka]
#orientation_symbols = [sx, sy, sz]
#orientation_symbols = [q0, q1, q2, q3]
landmark_symbols = [x_L, y_L, z_L]

all_symbols = position_symbols + orientation_symbols

R_wc = matrix44FromTaitBryan(tx, ty, tz, om, fi, ka)
R_cw=R_wc[:-1,:-1].transpose()
T_wc=Matrix([tx, ty, tz]).vec()
T_cw=-R_cw*T_wc
RT_cw=Matrix.hstack(R_cw, T_cw)
RT_cw=Matrix.vstack(RT_cw, Matrix([[0,0,0,1]]))

point_Landmark = Matrix([x_L, y_L, z_L]).vec()
point_source = Matrix([x_s, y_s, z_s, 1]).vec()
transformed_point_source = (  RT_cw * point_source)[:-1,:]
#transformed_point_s = (matrix44FromRodrigues(tx, ty, tz, sx, sy, sz) * point_source)[:-1,:]
#transformed_point_s = (matrix44FromQuaternion(tx, ty, tz, q0, q1, q2, q3) * point_source)[:-1,:]

target_value = Matrix([0,0,0]).vec()
model_function = transformed_point_source-point_Landmark
delta = target_value - model_function
delta_jacobian=delta.jacobian(all_symbols)
#delta_hessian=delta_jacobian.jacobian(all_symbols)


print(delta)
print(delta_jacobian)
err = Matrix([model_function[0]*model_function[0] +  model_function[1]*model_function[1] +  model_function[2]*model_function[2]])
#print(delta_hessian)
hessian = err.jacobian(all_symbols).jacobian(all_symbols)


with open("point_to_point_source_to_landmark_tait_bryan_cw_jacobian.h",'w') as f_cpp:
    f_cpp.write("inline void point_to_point_source_to_landmark_tait_bryan_cw(double &delta_x, double &delta_y, double &delta_z, double tx, double ty, double tz, double om, double fi, double ka, double x_s, double y_s, double z_s, double x_L, double y_L, double z_L)\n")
    f_cpp.write("{")
    f_cpp.write("delta_x = %s;\n"%(ccode(delta[0,0])))
    f_cpp.write("delta_y = %s;\n"%(ccode(delta[1,0])))
    f_cpp.write("delta_z = %s;\n"%(ccode(delta[2,0])))
    f_cpp.write("}")
    f_cpp.write("\n")
    f_cpp.write("inline void point_to_point_source_to_landmark_tait_bryan_cw_jacobian(Eigen::Matrix<double, 3, 6, Eigen::RowMajor> &j, double tx, double ty, double tz, double om, double fi, double ka, double x_s, double y_s, double z_s, double x_L, double y_L, double z_L)\n")
    f_cpp.write("{")
    for i in range (3):
        for j in range (6):
            f_cpp.write("j.coeffRef(%d,%d) = %s;\n"%(i,j, ccode(delta_jacobian[i,j])))
    f_cpp.write("}")

    f_cpp.write("inline void point_to_point_source_to_landmark_tait_bryan_cw_hessian(Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &j, double tx, double ty, double tz, double om, double fi, double ka, double x_s, double y_s, double z_s, double x_L, double y_L, double z_L)\n")
    f_cpp.write("{")
    for i in range (6):
        for j in range (6):
            f_cpp.write("j.coeffRef(%d,%d) = %s;\n"%(i,j, ccode(hessian[i,j])))
    f_cpp.write("}")


