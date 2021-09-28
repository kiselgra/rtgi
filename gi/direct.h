#pragma once

#include "libgi/algorithm.h"
#include "libgi/material.h"

#ifndef RTGI_SKIP_DIRECT_ILLUM
class direct_light : public recursive_algorithm {
	enum sampling_mode { sample_uniform, sample_cosine, sample_light, sample_brdf };
	enum sampling_mode sampling_mode = sample_light;

protected:
	vec3 sample_uniformly(const diff_geom &hit, const ray &view_ray);
	vec3 sample_lights(const diff_geom &hit, const ray &view_ray);
#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
	vec3 sample_cosine_weighted(const diff_geom &hit, const ray &view_ray);
	vec3 sample_brdfs(const diff_geom &hit, const ray &view_ray);
#endif
	
#ifndef RTGI_AXX
	sample_result full_mis(uint32_t x, uint32_t y, uint32_t samples);
#endif

public:
	gi_algorithm::sample_result sample_pixel(uint32_t x, uint32_t y, uint32_t samples) override;
	bool interprete(const std::string &command, std::istringstream &in) override;
};
#endif

#ifndef RTGI_A07
class direct_light_mis : public direct_light {
public:
	gi_algorithm::sample_result sample_pixel(uint32_t x, uint32_t y, uint32_t samples) override;
	bool interprete(const std::string &command, std::istringstream &in) override;
};
#else
#ifndef RTGI_A06
// todo: derive direct_light_mis from direct_light
#endif
#endif
