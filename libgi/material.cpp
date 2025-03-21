#include "material.h"
#include "scene.h"
#include "util.h"
#ifndef RTGI_SKIP_DIRECT_ILLUM
#include "sampling.h"
#endif

#ifndef RTGI_SKIP_BRDF
#ifndef RTGI_SKIP_LAYERED_BRDF
vec3 layered_brdf::f(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
#ifndef RTGI_SKIP_LAYERED_BRDF_IMPL
	const float F = fresnel_dielectric(absdot(geom.ns, w_o), 1.0f, geom.mat->ior);
	vec3 diff = base->f(geom, w_o, w_i);
	vec3 spec = coat->f(geom, w_o, w_i);
	return (1.0f-F)*diff + F*spec;
#else
	// todo proper fresnel reflection for layered material
	vec3 diff = base->f(geom, w_o, w_i);
	vec3 spec = coat->f(geom, w_o, w_i);
	return diff+spec;
#endif
}
#endif

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
float layered_brdf::pdf(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	const float F = fresnel_dielectric(absdot(geom.ns, w_o), 1.0f, geom.mat->ior);
	float pdf_diff = base->pdf(geom, w_o, w_i);
	float pdf_spec = coat->pdf(geom, w_o, w_i);
	return (1.0f-F)*pdf_diff + F*pdf_spec;
}

brdf::sampling_res layered_brdf::sample(const diff_geom &geom, const vec3 &w_o, const vec2 &xis) {
	const float F = fresnel_dielectric(absdot(geom.ns, w_o), 1.0f, geom.mat->ior);
	if (xis.x < F) {
		// specular sample
		vec2 new_xi((F-xis.x)/F, xis.y);
		auto [w_i, f_spec, pdf_spec, t] = coat->sample(geom, w_o, new_xi);
		vec3 f_diff = base->f(geom, w_o, w_i);
		float pdf_diff = base->pdf(geom, w_o, w_i);
		return { w_i, (1.0f-F)*f_diff + F*f_spec, (1.0f-F)*pdf_diff + F*pdf_spec, t };
	}
	else {
		vec2 new_xi((xis.x-F)/(1.0f-F), xis.y);
		auto [w_i, f_diff, pdf_diff, t] = base->sample(geom, w_o, new_xi);
		vec3 f_spec = coat->f(geom, w_o, w_i);
		float pdf_spec = coat->pdf(geom, w_o, w_i);
		return { w_i, (1.0f-F)*f_diff + F*f_spec, (1.0f-F)*pdf_diff + F*pdf_spec, t };
	}
}
#endif

// lambertian_reflection

vec3 lambertian_reflection::f(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	if (!same_hemisphere(w_i, geom.ns)) return vec3(0);
#ifndef RTGI_SKIP_BRDF_IMPL
	return one_over_pi * geom.albedo();
#else
	// todo
	// hint: make sure the incoming direction is in the upper hemisphere,
	//       see same_hemisphere (util.h)
	return vec3(0);
#endif
}

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
float lambertian_reflection::pdf(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
    return absdot(geom.ns, w_i) * one_over_pi;
}

brdf::sampling_res lambertian_reflection::sample(const diff_geom &geom, const vec3 &w_o, const vec2 &xis) {
	// uses malley's method, not what is asked on the assignment sheet
	vec3 w_i = align(cosine_sample_hemisphere(xis), geom.ns);
	if (!same_hemisphere(w_i, geom.ng)) return { w_i, vec3(0), 0, interaction_type::diffuse_reflection };
	float pdf_val = pdf(geom, w_o, w_i);
	assert(std::isfinite(pdf_val));
	return { w_i, f(geom, w_o, w_i), pdf_val, interaction_type::diffuse_reflection };
}
#endif


// specular_reflection

vec3 perfectly_specular_reflection::f(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	return vec3(0);
}

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
float perfectly_specular_reflection::pdf(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	return 0;
}

brdf::sampling_res perfectly_specular_reflection::sample(const diff_geom &geom, const vec3 &w_o, const vec2 &xis) {
	vec3 w_i = 2.0f*geom.ns*dot(geom.ns,w_o) - w_o;
	if (!same_hemisphere(w_i, geom.ng)) return { w_i, vec3(0), 0, interaction_type::specular_reflection };
	return { w_i, vec3(1), 1, interaction_type::specular_reflection };
}
#endif


#ifndef RTGI_SKIP_ASS
// dielectric specular reflection and transmission

vec3 dielectric_specular_bsdf::f(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	return vec3(0);
}

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
float dielectric_specular_bsdf::pdf(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	return 0;
}

brdf::sampling_res dielectric_specular_bsdf::sample_r(const diff_geom &geom, const vec3 &w_o, const vec2 &xis, float R) {
	vec2 new_xi((R-xis.x)/R, xis.y);
	vec3 w_i = 2.0f*geom.ns*dot(geom.ns,w_o) - w_o;
	if (!same_hemisphere(w_i, geom.ng)) return { w_i, vec3(0), 0, interaction_type::specular_reflection };
	vec3 col(R / absdot(w_i, geom.ns));
	return { w_i, col, R, interaction_type::specular_reflection };
}

// don't layer this!
brdf::sampling_res dielectric_specular_bsdf::sample(const diff_geom &geom, const vec3 &w_o, const vec2 &xis) {
	float cos_theta_o = dot(w_o, geom.ns);
	float R = fresnel_dielectric(cos_theta_o, 1.0f, geom.mat->ior);
	float T = 1.0f - R;
	if (xis.x < R)
		return sample_r(geom, w_o, xis, R);
	else {
		auto transmission_direction = (dot(w_o, geom.ns) > 0 ? interaction_type::transmission_in : interaction_type::transmission_out);
		vec2 new_xi((xis.x-R)/(1.0f-R), xis.y);
		auto [w_i, valid] = ::refract(w_o, geom.ns, geom.mat->ior);
		if (!valid) return { w_i, vec3(0), 0, interaction_type::specular | transmission_direction };
		vec3 col(T / absdot(w_i, geom.ns));
		// if "eye ray" (cf pbrt3, 566/571)
		float eta = (dot(w_o, geom.ns) > 0 ? 1.0f/geom.mat->ior : geom.mat->ior);
		col /= eta;
		return { w_i, col, T, interaction_type::specular | transmission_direction };
	}
}
#endif

vec3 thin_dielectric_specular_bsdf::f(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	return vec3(0);
}

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
float thin_dielectric_specular_bsdf::pdf(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	return 0;
}

// don't layer this!
// NOTE not tested extensively as I had no good scene at hand...
brdf::sampling_res thin_dielectric_specular_bsdf::sample(const diff_geom &geom, const vec3 &w_o, const vec2 &xis) {
	float cos_theta_o = dot(w_o, geom.ns);
	float R = fresnel_dielectric(cos_theta_o, 1.0f, geom.mat->ior);
	float T = 1.0f - R;
	if (R < 1.0f) {
		R += T*T*R / (1.0f - R*R);
		T = 1.0f - R;
	}
	if (xis.x < R) {
		// this is for two-sided surfaces
		if (same_hemisphere(w_o, geom.ng))
			return sample_r(geom, w_o, xis, R);
		else
			return sample_r(geom, -w_o, xis, R);
	}
	else {
		auto transmission_direction = (dot(w_o, geom.ns) > 0 ? interaction_type::transmission_in : interaction_type::transmission_out);
		vec3 w_i = -w_o;
		vec3 col(T / absdot(w_i, geom.ns));
		return { w_i, col, T, interaction_type::specular | transmission_direction };
	}
}
#endif
#endif

// specular phong brdf

vec3 phong_specular_reflection::f(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	if (!same_hemisphere(w_i, geom.ng)) return vec3(0);
#ifndef RTGI_SKIP_BRDF_IMPL
	float exponent = exponent_from_roughness(geom.mat->roughness);
	vec3 r = 2.0f*geom.ns*dot(w_i,geom.ns)-w_i;
	float cos_theta = cdot(w_o, r);
	const float norm_f = (exponent + 2.0f) * one_over_2pi;
	return (coat ? vec3(1) : geom.albedo()) * powf(cos_theta, exponent) * norm_f * cdot(w_i,geom.ns);
#else
	// todo
	// hint: make sure the incoming direction is in the upper hemisphere,
	return vec3(0);
#endif
}

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
float phong_specular_reflection::pdf(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	float exp = exponent_from_roughness(geom.mat->roughness);
	vec3 r = 2.0f*geom.ns*dot(geom.ns,w_o) - w_o;
	float z = cdot(r,w_i);
	return powf(z, exp) * (exp+1.0f) * one_over_2pi;
}

brdf::sampling_res phong_specular_reflection::sample(const diff_geom &geom, const vec3 &w_o, const vec2 &xis) {
	float exp = exponent_from_roughness(geom.mat->roughness);
#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING_IMPL
	float z = powf(xis.x, 1.0f/(exp+1));
	float phi = 2.0f*pi*xis.y;
	vec3 sample(sqrtf(1.0f-z*z) * cos(phi),
				sqrtf(1.0f-z*z) * sin(phi),
				z);
	vec3 r = 2.0f*geom.ns*dot(geom.ns,w_o) - w_o;
	vec3 w_i = align(sample, r);
	if (!same_hemisphere(w_i, geom.ng)) return { w_i, vec3(0), 0, interaction_type::specular_reflection };
	float pdf_val = pow(z,exp) * (exp+1.0f) * one_over_2pi;
	return { w_i, f(geom, w_o, w_i), pdf_val, interaction_type::specular_reflection };
#else
	// todo: derive the proper sampling formulas for phong and implement them here
	// note: make sure that the sampled direction is above the surface
	// note: for the pdf, do not forget the rotational-part for phi
	return { vec3(0), vec3(0), 0, interaction_type::specular_reflection };
#endif
}
#endif


#ifndef RTGI_SKIP_MF_BRDF

#ifndef RTGI_SKIP_MF_BRDF_IMPL
// Microfacet distribution helper functions
#define sqr(x) ((x)*(x))
#else
// In the following, make sure that corner cases are taken care of (positive
// dot products, >0 denominators, correctly aligned directions, etc)
#endif

inline float ggx_d(const float NdotH, float roughness) {
#ifndef RTGI_SKIP_MF_BRDF_IMPL
    if (NdotH <= 0) return 0.f;
    const float tan2 = tan2_theta(NdotH);
    if (!std::isfinite(tan2)) return 0.f;
    const float a2 = sqr(roughness);
    return a2 / (pi * sqr(sqr(NdotH)) * sqr(a2 + tan2));
#else
	// todo TR NDF
	return 0.0f;
#endif
}

inline float ggx_g1(const float NdotV, float roughness) {
#ifndef RTGI_SKIP_MF_BRDF_IMPL
    if (NdotV <= 0) return 0.f;
    const float tan2 = tan2_theta(NdotV);
    if (!std::isfinite(tan2)) return 0.f;
    return 2.f / (1.f + sqrtf(1.f + sqr(roughness) * tan2));
#else
	// todo TR Smith G_1
	return 0.0f;
#endif
}

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
vec3 ggx_sample(const vec2 &xi, float roughness) {
    const float theta = atanf((roughness * sqrtf(xi.x)) / sqrtf(1.f - xi.x));
    if (!std::isfinite(theta)) return vec3(0, 0, 1);
    const float phi = 2.f * pi * xi.y;
    const float sin_t = sinf(theta);
    return vec3(sin_t * cosf(phi), sin_t * sinf(phi), cosf(theta));
}

inline float ggx_pdf(float D, float NdotH, float HdotV) {
    return (D * (NdotH<0 ? -NdotH : NdotH)) / (4.f * (HdotV<0 ? -HdotV : HdotV));
}
#endif

#ifndef RTGI_SKIP_MF_BRDF_IMPL
#undef sqr
#endif

vec3 gtr2_reflection::f(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
#ifndef RTGI_SKIP_MF_BRDF_IMPL
    if (!same_hemisphere(geom.ng, w_i)) return vec3(0);
    const float NdotV = dot(geom.ns, w_o);
    const float NdotL = dot(geom.ns, w_i);
    if (NdotV <= 0.f || NdotL <= 0.f) return vec3(0);
    const vec3 H = normalize(w_o + w_i);
    const float NdotH = cdot(geom.ns, H);
    const float HdotL = cdot(H, w_i);
    const float roughness = geom.mat->roughness;
    const float F = fresnel_dielectric(HdotL, 1.f, geom.mat->ior);
    const float D = ggx_d(NdotH, roughness);
    const float G = ggx_g1(NdotV, roughness) * ggx_g1(NdotL, roughness);
    const float microfacet = (F * D * G) / (4 * glm::abs(NdotV) * glm::abs(NdotL));
    return coat ? vec3(microfacet) : geom.albedo() * microfacet;
#else
	// todo full TR Microfacet BRDF, use ggx_d and ggx_g1
	return vec3(0);
#endif
}

#ifndef RTGI_SKIP_IMPORTANCE_SAMPLING
float gtr2_reflection::pdf(const diff_geom &geom, const vec3 &w_o, const vec3 &w_i) {
	const vec3 H = normalize(w_o + w_i);
	const float NdotH = cdot(geom.ns, H);
	const float HdotV = dot(H, w_o);
	const float D = ggx_d(NdotH, geom.mat->roughness);
	const float pdf = ggx_pdf(D, NdotH, HdotV);
	assert(pdf >= 0);
	assert(std::isfinite(pdf));
	return pdf;
}

brdf::sampling_res gtr2_reflection::sample(const diff_geom &geom, const vec3 &w_o, const vec2 &xis) {
	// reflect around sampled macro normal w_h
	const vec3 w_h = align(ggx_sample(xis, geom.mat->roughness), geom.ns);
	vec3 w_i = 2.0f*w_h*dot(w_h, w_o) - w_o;
	if (!same_hemisphere(geom.ng, w_i)) return { w_i, vec3(0), 0, interaction_type::specular_reflection };
	if (!same_hemisphere(geom.ng, w_o)) return { w_i, vec3(0), 0, interaction_type::specular_reflection }; // this breaks two-sided faces (improper 3d geom)
	float sample_pdf = pdf(geom, w_o, w_i);
	return { w_i, f(geom, w_o, w_i), sample_pdf, interaction_type::specular_reflection };
}
#endif

#endif

brdf *new_brdf(const std::string name, scene &scene) {
	if (scene.brdfs.count(name) == 0) {
		brdf *f = nullptr;
		if (name == "lambert")
			f = new lambertian_reflection;
		else if (name == "mirror" || name == "specular")
			f = new perfectly_specular_reflection;
#ifndef RTGI_SKIP_ASS
		else if (name == "dielectric-specular")
			f = new dielectric_specular_bsdf;
		else if (name == "thin-dielectric-specular")
			f = new thin_dielectric_specular_bsdf;
#endif
		else if (name == "phong")
			f = new phong_specular_reflection;
		else if (name == "layered-phong") {
#ifndef RTGI_SKIP_LAYERED_BRDF
			brdf *base = new_brdf("lambert", scene);
			specular_brdf *coat = dynamic_cast<specular_brdf*>(new_brdf("phong", scene));
			assert(coat);
			f = new layered_brdf(coat, base);
#else
			throw std::logic_error("Not implemented, yet");
#endif
		}
#ifndef RTGI_SKIP_MF_BRDF
		else if (name == "gtr2")
			f = new gtr2_reflection;
		else if (name == "layered-gtr2") {
#ifndef RTGI_SKIP_LAYERED_BRDF_IMPL
			brdf *base = new_brdf("lambert", scene);
			specular_brdf *coat = dynamic_cast<specular_brdf*>(new_brdf("gtr2", scene));
			assert(coat);
			f = new layered_brdf(coat, base);
#else
			// todo intantiate proper layered brdf for lambert/gtr2
			throw std::logic_error(std::string("Not implemented yet: ") + name);
#endif
		}
#endif
		else
			throw std::runtime_error(std::string("No such brdf defined: ") + name);
		return f;
	}
	return scene.brdfs[name];
}
#endif

