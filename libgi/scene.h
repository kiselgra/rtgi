#pragma once

#include "camera.h"
#include "intersect.h"
#include "material.h"
#ifndef RTGI_SKIP_LIGHT_SOURCE_SAMPLING
#include "discrete_distributions.h"
#endif

#include <vector>
#include <map>
#include <string>
#include <filesystem>

namespace wf {
	class batch_ray_tracer;
}

template <typename T>
struct texture2d {
	std::string name;
	std::filesystem::path path;
	unsigned w, h;
	T *texel = nullptr;
	~texture2d() {
		delete [] texel;
	}
	const T& sample(float u, float v) const {
		u = u - floor(u);
		v = v - floor(v);
		int x = (int)(u*w+0.5f);
		int y = (int)(v*h+0.5f);
		if (x == w) x = 0;
		if (y == h) y = 0;
		return texel[y*w+x];
	}
	const T& sample(vec2 uv) const {
		return sample(uv.x, uv.y);
	}
	const T& operator()(float u, float v) const {
		return sample(u, v);
	}
	const T& operator()(vec2 uv) const {
		return sample(uv.x, uv.y);
	}
	const T& value(int x, int y) const {
		return texel[y*w+x];
	}
	const T& operator[](glm::uvec2 xy) const {
		return value(xy.x, xy.y);
	}
	uint32_t size_in_bytes() {
		return w*h*sizeof(T);
	}
};

texture2d<vec3>* load_image3f(const std::filesystem::path &path, bool crash_on_error = true);
texture2d<vec4>* load_image4f(const std::filesystem::path &path, const std::filesystem::path *opacity_path = nullptr, bool crash_on_error = true);
texture2d<vec3>* load_hdr_image3f(const std::filesystem::path &path);


#ifndef RTGI_SKIP_BRDF
struct light {
	virtual ~light() {}
	virtual vec3 power() const = 0;
#ifndef RTGI_SKIP_LIGHT_SOURCE_SAMPLING
	virtual tuple<ray, vec3, float> sample_Li(const diff_geom &from, const vec2 &xis) const = 0;
// 	virtual float pdf(const ray &r) const = 0;
#endif
};

struct pointlight : public light {
	vec3 pos;
	vec3 col;
	pointlight(const vec3 pos, const vec3 col) : pos(pos), col(col) {}
	vec3 power() const override;
#ifndef RTGI_SKIP_LIGHT_SOURCE_SAMPLING
	tuple<ray, vec3, float> sample_Li(const diff_geom &from, const vec2 &xis) const override;
// 	float pdf(const ray &r) const override { return 0; }
#endif
};
#endif

#ifndef RTGI_SKIP_DIRECT_ILLUM
/*! Keeping the emissive triangles as seperate copies might seem like a strange design choice.
 *  It is. However, this way the BVH is allowed to reshuffle triangle positions (not vertex positions!)
 *  without having an effect on this.
 *
 *  We might have to re-think this at a point, we could as well provide an indirection-array for the BVH's
 *  triangles. I opted against that at first, as not to intorduce overhead, but any more efficient representation
 *  copy the triangle data on the GPU or in SIMD formats anyway.
 */
struct trianglelight : public light, public triangle {
	const ::scene& scene;
	trianglelight(const ::scene &scene, uint32_t i);
	vec3 power() const override;
#ifndef RTGI_SKIP_LIGHT_SOURCE_SAMPLING
	tuple<ray, vec3, float> sample_Li(const diff_geom &from, const vec2 &xis) const override;
	float pdf(const ray &r, const diff_geom &on_light) const;
#endif
};
#endif

#ifndef RTGI_SKIP_SKY
struct skylight : public light {
	texture2d<vec3> *tex = nullptr;
	float intensity_scale;
	distribution_2d *distribution = nullptr;
	float scene_radius;

	skylight(const std::filesystem::path &file, float intensity_scale) : intensity_scale(intensity_scale) {
		tex = load_hdr_image3f(file);
	}
	void build_distribution();
	void scene_bounds(aabb box);
	vec3 Le(const ray &ray) const;
	virtual vec3 power() const;
	virtual tuple<ray, vec3, float> sample_Li(const diff_geom &from, const vec2 &xis) const;
	float pdf_Li(const ray &ray) const;
};

#endif


/*  \brief The scene culminates all the geometric information that we use.
 *
 *  This naturally includes the surface geometry to be displayed, but also light sources and cameras.
 *
 *  The scene also holds a reference to the ray tracer as the tracer runs on the scene's data.
 *
 */
struct scene {
	std::vector<::vertex>    vertices;
	std::vector<::triangle>  triangles;
	std::vector<::material>  materials;
	std::vector<::texture2d<vec4>*>  textures;
#ifndef RTGI_SKIP_BRDF
	std::map<std::string, brdf*> brdfs;
#endif
#ifndef RTGI_SKIP_LOCAL_ILLUM
	std::vector<light*>      lights;
#endif
#ifndef RTGI_SKIP_DIRECT_ILLUM
	void find_light_geometry();
	void compute_light_distribution();
#endif
#ifndef RTGI_SKIP_LIGHT_SOURCE_SAMPLING
	distribution_1d *light_distribution;
#endif
#ifndef RTGI_SKIP_SKY
	skylight *sky = nullptr;
#endif
	std::map<std::string, ::camera> cameras;
	::camera camera;
	vec3 up;
	aabb scene_bounds;

	//! meshes with this material will not be loaded
	std::vector<std::string> mtl_blacklist;

	const ::material material(uint32_t triangle_index) const {
		return materials[triangles[triangle_index].material_id];
	}
	scene() : camera(vec3(0,0,-1), vec3(0,0,0), vec3(0,1,0), 65, 1280, 720) {
		modelpaths.push_back("");
	}
	~scene();
	void add(const std::filesystem::path &path, const std::string &name, const glm::mat4 &trafo = glm::mat4());
	
	std::vector<std::filesystem::path> modelpaths;
	void add_modelpath(const std::filesystem::path &p);
	void remove_modelpath(const std::filesystem::path &p);

	vec3 normal(const triangle &tri) const;

#ifndef RTGI_SKIP_WF
	//! This is only used for individual_ray_tracers
#endif
	individual_ray_tracer *rt = nullptr;
	
	//! The scene takes ownership of the RT, deletes it upon destruction and when taking ownership of a new RT.
	void release_rt();
	void use(individual_ray_tracer *new_rt);
	void print_memory_stats();
};

// std::vector<triangle> scene_triangles();
// scene load_scene(const std::string& filename);
