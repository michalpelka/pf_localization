#ifndef _STRUCTS_H_
#define _STRUCTS_H_

struct Point{
	float x;
	float y;
	float z;
	char label;
};

struct GlobalStructures{
	std::vector<Point> map;
};


#endif
