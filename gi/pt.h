#pragma once

#include "libgi/algorithm.h"
#include "libgi/material.h"

#ifndef RTGI_SKIP_SIMPLE_PT
class simple_pt : public recursive_algorithm {
protected:
	int max_path_len = 10;
	int rr_start = 2;  // start RR after this many unrestricted bounces
	enum class bounce { uniform, cosine, brdf } bounce = bounce::brdf;

	virtual vec3 path(ray view_ray, int x, int y);
public:
	glm::vec3 sample_pixel(uint32_t x, uint32_t y) override;
	bool interprete(const std::string &command, std::istringstream &in) override;
};

#ifndef RTGI_SKIP_PT
class pt_nee : public simple_pt {
	vec3 path(ray view_ray, int x, int y) override;
	std::tuple<ray,vec3,float> sample_light(const diff_geom &hit);
// 	bool mis = true;
	bool mis = false;
public:
	bool interprete(const std::string &command, std::istringstream &in) override;
	void finalize_frame();
};
#endif
#endif
