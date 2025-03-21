#include "pt.h"

#include "libgi/rt.h"
#include "libgi/context.h"
#include "libgi/intersect.h"
#include "libgi/util.h"
#include "libgi/color.h"
#include "libgi/sampling.h"

#include "libgi/timer.h"

#include "libgi/global-context.h"

using namespace glm;
using namespace std;

// #define WITH_RAY_EXPORT

#ifdef WITH_RAY_EXPORT
static std::string rayfile = "/tmp/raydata";
static std::vector<std::pair<int,ray>> all_rays;

static void record_ray(int bounce_and_kind, const ray &ray) {
	#pragma omp critical
	all_rays.push_back({bounce_and_kind, ray});
}
#else
static void record_ray(int bounce_and_kind, const ray &ray) {}
#endif

#ifndef RTGI_SKIP_SIMPLE_PT
// 
// ----------------------- simple pt -----------------------
//
// #define SIGNIFICANT_RAY_COUNT

vec3 simple_pt::sample_pixel(uint32_t x, uint32_t y) {
	vec3 r = path(cam_ray(rc->scene.camera, x, y, glm::vec2(rc->rng.uniform_float()-0.5f, rc->rng.uniform_float()-0.5f)), x, y);
#ifdef SIGNIFICANT_RAY_COUNT
	return r==vec3(0) ? vec3(0) : vec3(1);
#else
	return r;
#endif
}

#define check_range(X) {\
		assert(X.x >= 0);assert(std::isfinite(X.x));\
		assert(X.y >= 0);assert(std::isfinite(X.y));\
		assert(X.z >= 0);assert(std::isfinite(X.z)); }

#define check_rangef(X) \
		{assert(X >= 0);assert(std::isfinite(X));}

#define assert_valid_pdf(x) \
		{assert(x>0); assert(std::isfinite(x));}

vec3 simple_pt::path(ray ray, int x, int y) {
#ifdef RTGI_SKIP_SIMPLE_PT_IMPL
	// TODO implement the first variant of the PT algorithm (and later on, add RR)
	// Layout:
	// - radiance = 0,0,0, throughput = 1,1,1
	// - loop until max path length
	// - find closest hit starting with "current ray"
	// - check if intersected surface is emissive. if so, add correctly weighted radiance and break the loop
	// - if not, compute a properly sampled reflection ray
	// Notes
	// - for RR, compute the perceived brightness of the throughput via the luma function to obtain a scalar value
	return vec3(0);
#else
	time_this_block(pathtrace);
	vec3 radiance(0);
	vec3 throughput(1);
	for (int i = 0; i < max_path_len; ++i) {
		
		// find hitpoint with scene
		triangle_intersection closest = rc->scene.rt->closest_hit(ray);
		if (!closest.valid()) {
			if (rc->scene.sky)
				radiance = throughput * rc->scene.sky->Le(ray);
			break;
		}
		diff_geom hit(closest, rc->scene);
		if (rc->enable_denoising) {
			rc->framebuffer_normal.add(x, y, hit.ns);
			rc->framebuffer_albedo.add(x, y, hit.albedo());
		}
	
		// if it is a light, add the light's contribution
		if (hit.mat->emissive != vec3(0)) {
			radiance = throughput * hit.mat->emissive;
			break;
		}
		
		// bounce the ray
		auto [w_i, f, pdf, ia] = hit.mat->brdf->sample(hit, -ray.d, rc->rng.uniform_float2());
		::ray bounced(hit.x, w_i);
		if (pdf <= 0.0f)  break;
		throughput *= f * absdot(bounced.d, hit.ns) / pdf;
		check_range(throughput);
		if (reflection(ia))            ray = offset_ray(bounced, hit.ng);
		else if (transmission_out(ia)) ray = offset_ray(bounced, hit.ng);
		else                           ray = offset_ray(bounced, -hit.ng);

		// apply RR
		if (i > rr_start) {
			float xi = uniform_float();
			float p_term = 1.0f - luma(throughput);
			if (xi > p_term)
				throughput *= 1.0f/(1.0f-p_term);
			else
				break;
		}
		else if (luma(throughput) == 0)
			break;
	}
	check_range(radiance);
	return radiance;
#endif
}

bool simple_pt::interprete(const std::string &command, std::istringstream &in) {
	string sub, val;
	if (command == "path") {
		in >> sub;
		if (sub == "len") {
			int i = 0;
			in >> i;
			if (i <= 0)
				cerr << "error in path len: expected a positive integer, got " << i << endl;
			else
				max_path_len = i;
			return true;
		}
		else if (sub == "bounce") {
			in >> val;
			if (val == "uniform")     bounce = bounce::uniform;
			else if (val == "cosine") bounce = bounce::cosine;
			else if (val == "brdf")   bounce = bounce::brdf;
			else cerr << "error: invalid kind of path bounce: '" << val << "'" << endl;
			return true;
		}
		else if (sub == "rr-start") {
			int i = 0;
			in >> i;
			if (i <= 0)
				cerr << "error in russian roulette offset: expected a positive integer > 0, got " << i << endl;
			else
				rr_start = i;
			return true;
		}
		else {
			cerr << "unknown subcommand to path: '" << sub << "'" << endl;
			return true;
		}
	}
	return false;
}

#ifndef RTGI_SKIP_PT
// 
// ----------------------- pt with next event estimation -----------------------
//

vec3 pt_nee::path(ray ray, int x, int y) {
	vec3 radiance(0);
#ifdef RTGI_SKIP_PT_IMPL
	// Start by implementing PT with NEE
	// Layout
	// - find hitpoint with scene
	// - if it is a light AND we have not bounced yet, add the light's contribution
	// - for mis we take the next path vertex to be the brdf sample of the next-event path
	// - branch off direct lighting path that directly terminates
	// - bounce the ray  NOTE: bounce might bounce other than with the BRDF, but we strictly use the BRDF-pdf above
	// - apply RR
#else
	vec3 throughput(1);
	float brdf_pdf = 0;
	for (int i = 0; i < max_path_len; ++i) {
		record_ray(i, ray);
		// find hitpoint with scene
		triangle_intersection closest = rc->scene.rt->closest_hit(ray);
		if (!closest.valid()) {
			if (rc->scene.sky)
				if (!mis || i==0)
					radiance += throughput * rc->scene.sky->Le(ray);
				else {
#ifndef RTGI_SKIP_ASS
					// TODO misses light distrib
#endif
					float light_pdf = rc->scene.sky->pdf_Li(ray);
					radiance += throughput * rc->scene.sky->Le(ray) * brdf_pdf / (light_pdf+brdf_pdf);
				}
			break;
		}
		diff_geom hit(closest, rc->scene);
		if (rc->enable_denoising) {
			rc->framebuffer_normal.add(x, y, hit.ns);
			rc->framebuffer_albedo.add(x, y, hit.albedo());
		}
	
		// if it is a light AND we have not bounced yet, add the light's contribution
		if (i == 0 && hit.mat->emissive != vec3(0)) {
			radiance += throughput * hit.mat->emissive;
			break;
		}
		// for mis we take the next path vertex to be the brdf sample of the next-event path
		if (mis && hit.mat->emissive != vec3(0)) {
		    float a=0, b=0;
			trianglelight tl(rc->scene, closest.ref);
			float light_pdf = luma(tl.power()) / rc->scene.light_distribution->integral();
			light_pdf *= tl.pdf(ray, hit); // no problem if light_pdf is 0
			radiance += throughput * hit.mat->emissive * brdf_pdf / (light_pdf + brdf_pdf);
			break;
		}
		if (hit.mat->emissive != vec3(0)) {
			radiance += throughput * hit.mat->emissive;
			break;
		}

		// branch off direct lighting path that directly terminates
		auto [shadow_ray,light_col,light_pdf] = sample_light(hit);
		if (light_pdf != 0 && light_col != vec3(0)) {
			record_ray(i+100,shadow_ray);
			if (!rc->scene.rt->any_hit(shadow_ray)) {
				float divisor = light_pdf;
				assert_valid_pdf(light_pdf);
				if (mis)
					divisor += hit.mat->brdf->pdf(hit, -ray.d, shadow_ray.d);
				check_rangef(divisor);
				// note: light pdf cancels via balance heuristic, no need to divide by it (and is only part of divisor for non-mis)
				radiance += throughput
				            * light_col
							* hit.mat->brdf->f(hit, -ray.d, shadow_ray.d)
							* cdot(shadow_ray.d, hit.ns)
							/ divisor;
				check_range(radiance);
			}
		}

		// bounce the ray  TODO: bounce might bounce other than with the BRDF, but we strictly use the BRDF above
		auto [w_i, f, pdf, ia] = hit.mat->brdf->sample(hit, -ray.d, rc->rng.uniform_float2());
		::ray bounced(hit.x, w_i);
		brdf_pdf = pdf;	// for mis in next iteration
		throughput *= f * absdot(bounced.d, hit.ns) / pdf;
		if (pdf <= 0.0f || luma(throughput) <= 0.0f) break;
		if (reflection(ia))            ray = offset_ray(bounced, hit.ng);
		else if (transmission_out(ia)) ray = offset_ray(bounced, hit.ng);
		else                           ray = offset_ray(bounced, -hit.ng);

		// apply RR
		if (i > rr_start) {
			float xi = uniform_float();
			float p_term = 1.0f - luma(throughput);
			if (xi > p_term)
				throughput *= 1.0f/(1.0f-p_term);
			else
				break;
		}
	}
#endif
	check_range(radiance);
	return radiance;
}

std::tuple<ray,vec3,float> pt_nee::sample_light(const diff_geom &hit) {
	auto [l_id, l_pdf] = rc->scene.light_distribution->sample_index(rc->rng.uniform_float());
	light *l = rc->scene.lights[l_id];
	auto [shadow_ray,l_col,pdf] = l->sample_Li(hit, rc->rng.uniform_float2());
	return { shadow_ray, l_col, pdf * l_pdf };
}

bool pt_nee::interprete(const std::string &command, std::istringstream &in) {
	string sub, val;
	std::istringstream local_in(in.str());
	local_in >> sub; // remove command that has already been read
	if (command == "path") {
		local_in >> sub;
		if (sub == "mis") {
			local_in >> val;
			if (val == "on") mis = true;
			else if (val == "off") mis = false;
			else cerr << "usage: path mis [on|off]" << endl;
		}
#ifdef WITH_RAY_EXPORT
		if (sub == "rayfile") {
			local_in >> rayfile;
		}
#endif
		else
			simple_pt::interprete(command, in);
		return true;
	}

	return false;
}

void pt_nee::finalize_frame() {
#ifdef WITH_RAY_EXPORT
	ofstream out(rayfile);
	for (auto [i,r] : all_rays)
		out << i << " " << r.o.x << " " << r.o.y << " " << r.o.z << " " << r.d.x << " " << r.d.y << " " << r.d.z << " " << r.t_min << " " << r.t_max << endl;
#endif
}

#endif
#endif
