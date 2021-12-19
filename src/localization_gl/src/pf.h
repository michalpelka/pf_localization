#ifndef _PF_H_
#define _PF_H_

#include "structs.h"

void initialize_host_device_data(HostDeviceData& data);
void compute_occupancy(std::vector<Point> &host_points, Grid3DParams params, std::vector<char> &host_occupancy_map);


#endif
