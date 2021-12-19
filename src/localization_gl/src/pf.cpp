#include "pf.h"
#include "rgd.cuh"

void initialize_host_device_data(HostDeviceData& data)
{
	cudaMalloc((void **)&data.device_map, sizeof(Point)*data.host_map.size());
	cudaMemcpy(data.device_map, data.host_map.data(), sizeof(Point)*data.host_map.size(), cudaMemcpyHostToDevice);
	data.device_map_size = data.host_map.size();

	data.map_grid3Dparams.resolution_X = 1.0;
	data.map_grid3Dparams.resolution_Y = 1.0;
	data.map_grid3Dparams.resolution_Z = 1000.0;
	data.map_grid3Dparams.bounding_box_extension = 0;

	throw_cuda_error(cudaCalculateParams3D(
			data.device_map,
			data.device_map_size,
			data.map_grid3Dparams), __FILE__, __LINE__);

	data.device_occupancy_map_size = data.map_grid3Dparams.number_of_buckets;
	data.host_occupancy_map.resize(data.device_occupancy_map_size);

	compute_occupancy(data.host_map, data.map_grid3Dparams, data.host_occupancy_map);

	cudaMalloc((void **)&data.device_occupancy_map, sizeof(char)*data.map_grid3Dparams.number_of_buckets);
	throw_cuda_error(cudaMemcpy(data.device_occupancy_map, &(data.host_occupancy_map[0]), sizeof(char)*data.device_occupancy_map_size, cudaMemcpyHostToDevice), __FILE__, __LINE__);

	data.particle_filter_state = ParticleFilterState::initial;

	data.initial_w = -0.2;

	for(size_t i = 0 ; i < 10000000; i++){
		Pose pose;
		pose.p.x = (((float(rand()%1000000))/1000000.0f) - 0.5) * 200.0;
		pose.p.y = (((float(rand()%1000000))/1000000.0f) - 0.5) * 200.0;
		pose.o.z_angle_rad = (((float(rand()%1000000))/1000000.0f) - 0.5) * 2.0 * M_PI;

		Particle p;
		p.is_tracking = false;
		p.pose = pose;
		p.W = data.initial_w;
		p.nW = 0;

		data.particle_filter_initial_guesses.push_back(p);
	}

#if 0


	global_structures.std_update.p.x = 0.05;
	global_structures.std_update.p.y = 0.05;
	global_structures.std_update.p.z = 0.05;
	global_structures.std_update.o.x_angle_rad = 0.005 * M_PI/180.0;
	global_structures.std_update.o.y_angle_rad = 0.005 * M_PI/180.0;;
	global_structures.std_update.o.z_angle_rad = 0.005 * M_PI/180.0;;

	global_structures.percent_particles_from_initial=0.9;
	global_structures.max_particles=30000;

	global_structures.particle_filter_state = initial;
	global_structures.min_dump_propability = -10;//-0.1;
	global_structures.min_dump_propability_tracking = -0.1;//-0.01;
	global_structures.min_dump_propability_no_bservations = -1.0;


	global_structures.initial_w_exploration_particles = -0.1;



	global_structures.initial_w_cylinder_particles = -0.2;

	global_structures.number_of_replicatations_cylinder = 10;
	global_structures.number_of_replicated_best_particles_cylinder = 1000;

	global_structures.std_cylinder_x = 2;
	global_structures.std_cylinder_y = 2;
	global_structures.std_cylinder_z = 5;
	global_structures.std_cylinder_x_angle_deg = 2;
	global_structures.std_cylinder_y_angle_deg = 2;
	//global_structures.std_cylinder_z_angle_deg = 5;
	global_structures.std_cylinder_z_angle_deg = 180;
	global_structures.std_cylinder_angle_deg = 15;



	global_structures.mean_initial_guess_x=0;
	global_structures.mean_initial_guess_y=0;
	global_structures.mean_initial_guess_z=30 - 100;
	global_structures.std_initial_guess_x=3;
	global_structures.std_initial_guess_y=3;
	global_structures.std_initial_guess_z=30;
	global_structures.mean_initial_guess_x_angle_deg=0;
	global_structures.mean_initial_guess_y_angle_deg=0;
	global_structures.mean_initial_guess_z_angle_deg=0;
	global_structures.std_initial_guess_x_angle_deg=1;
	global_structures.std_initial_guess_y_angle_deg=1;
	global_structures.std_initial_guess_z_angle_deg=90;
	global_structures.min_z=1 - 100;
	global_structures.max_z=150 - 100;


	global_structures.number_of_replicated_best_particles_motion_model = 1000;
	global_structures.number_of_replicatations_motion_model = 10;

	global_structures.std_autobus_x = 2;
	global_structures.std_autobus_y = 2;
	global_structures.std_autobus_z = 5;
	global_structures.std_autobus_x_angle_deg = 2;
	global_structures.std_autobus_y_angle_deg = 2;
	global_structures.std_autobus_z_angle_deg = 5;














	std::default_random_engine generator;
	std::normal_distribution<double> dist_x(global_structures.mean_initial_guess_x, global_structures.std_initial_guess_x);
	std::normal_distribution<double> dist_y(global_structures.mean_initial_guess_y, global_structures.std_initial_guess_y);
    Generator dist_z(global_structures.mean_initial_guess_z, global_structures.std_initial_guess_z,global_structures.min_z,global_structures.max_z);
	std::normal_distribution<double> dist_x_angle(global_structures.mean_initial_guess_x_angle_deg, global_structures.std_initial_guess_x_angle_deg);
	std::normal_distribution<double> dist_y_angle(global_structures.mean_initial_guess_y_angle_deg, global_structures.std_initial_guess_y_angle_deg);
	std::normal_distribution<double> dist_z_angle(global_structures.mean_initial_guess_z_angle_deg, global_structures.std_initial_guess_z_angle_deg);

	// for(size_t i = 0 ; i < 10000000; i++){
	// 	Particle particle;
	// 	particle.W = 1;
	// 	particle.nW = 0;
	// 	particle.overlap = 0;
	// 	particle.pose.p.x = dist_x(generator);
	// 	particle.pose.p.y = dist_y(generator);
	// 	particle.pose.p.z = dist_z();

	// 	particle.pose.o.x_angle_rad = dist_x_angle(generator);
	// 	particle.pose.o.y_angle_rad = dist_y_angle(generator);
	// 	particle.pose.o.z_angle_rad = dist_z_angle(generator);
	// 	printf("%lf,%lf,%lf,%lf,%lf,%lf\n",particle.pose.p.x,particle.pose.p.y,particle.pose.p.z,particle.pose.o.x_angle_rad,particle.pose.o.y_angle_rad,particle.pose.o.z_angle_rad);

	// 	global_structures.particle_filter_initial_guesses.push_back(particle);
	// }







	std::cout << "initialize_cuda_structures DONE" << std::endl;
#endif
}

void compute_occupancy(std::vector<Point> &host_points,	Grid3DParams params, std::vector<char> &host_occupancy_map)
{
	for (size_t ind=0; ind < host_occupancy_map.size(); ++ind)
	{
		host_occupancy_map[ind] = PointType::not_initialised;
	}

	for(size_t index_of_point_source = 0 ; index_of_point_source < host_points.size(); index_of_point_source++){

		Point &pSource = host_points[index_of_point_source];

		if(pSource.x < params.bounding_box_min_X || pSource.x > params.bounding_box_max_X)continue;
		if(pSource.y < params.bounding_box_min_Y || pSource.y > params.bounding_box_max_Y)continue;
		if(pSource.z < params.bounding_box_min_Z || pSource.z > params.bounding_box_max_Z)continue;


		long long unsigned int ix = (pSource.x - params.bounding_box_min_X) / params.resolution_X;
		long long unsigned int iy = (pSource.y - params.bounding_box_min_Y) / params.resolution_Y;
		long long unsigned int iz = (pSource.z - params.bounding_box_min_Z) / params.resolution_Z;


		long long unsigned int index_bucket = ix* static_cast<long long unsigned int>(params.number_of_buckets_Y)*
						static_cast<long long unsigned int>(params.number_of_buckets_Z) +
						iy* static_cast<long long unsigned int>(params.number_of_buckets_Z) + iz;

		if(index_bucket < host_occupancy_map.size()){
			host_occupancy_map[index_bucket] = pSource.label;
		}
	}
}
