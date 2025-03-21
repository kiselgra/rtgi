#include "bounce.h"

#include "libgi/gl/shader.h"

#include <iostream>
using std::cout, std::endl;

namespace wf::gl {

	extern compute_shader sample_uniform_shader;
	extern compute_shader sample_light_shader;
	extern compute_shader integrate_dir_sample_shader;
	extern compute_shader integrate_light_sample_shader;

	void sample_uniform_dir::run() {
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
		time_this_wf_step;
		auto res = rc->resolution();

		bind_texture_as_image bind_0(camdata->rays,          0, true, false);
		bind_texture_as_image bind_1(camdata->intersections, 1, true, false);
		bind_texture_as_image bind_2(camdata->framebuffer,   2, true, true);
		bind_texture_as_image bind_3(bouncedata->rays,       3, false, true);
		bind_texture_as_image bind_4(pdf->data,              4, false, true);
		compute_shader &cs = sample_uniform_shader;
		cs.bind();
		cs.uniform("w", res.x).uniform("h", res.y);
		cs.dispatch(res.x, res.y);
		cs.unbind();
	}

	void sample_cos_weighted_dir::run() {
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
		time_this_wf_step;
		auto res = rc->resolution();

		bind_texture_as_image bind_0(camdata->rays,          0, true, false);
		bind_texture_as_image bind_1(camdata->intersections, 1, true, false);
		bind_texture_as_image bind_2(camdata->framebuffer,   2, true, true);
		bind_texture_as_image bind_3(bouncedata->rays,       3, false, true);
		bind_texture_as_image bind_4(pdf->data,              4, false, true);
		compute_shader &cs = sample_uniform_shader;
		cs.bind();
		cs.uniform("w", res.x).uniform("h", res.y);
		cs.uniform("sample_cos_instead", true);
		cs.dispatch(res.x, res.y);
		cs.unbind();
	}

	void sample_light_dir::run() {
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
		time_this_wf_step;
		auto res = rc->resolution();
	
		bind_texture_as_image bind_0(camdata->rays,          0, true,  false);
		bind_texture_as_image bind_1(camdata->intersections, 1, true,  false);
		bind_texture_as_image bind_2(camdata->framebuffer,   2, true,  true);
		bind_texture_as_image bind_3(bouncedata->rays,       3, false, true);
		bind_texture_as_image bind_4(pdf->data,              4, false, true);
		bind_texture_as_image bind_5(light_dist->f,          5, true,  false);
		bind_texture_as_image bind_6(light_dist->cdf,        6, true,  false);
		bind_texture_as_image bind_7(light_dist->tri_lights, 7, true,  false);
		bind_texture_as_image bind_8(light_col->data,        8, false, true);
		compute_shader &cs = sample_light_shader;
		cs.bind();
		cs.uniform("w", res.x).uniform("h", res.y);
		cs.uniform("integral_1spaced", light_dist->integral_1spaced);
		cs.uniform("lights", light_dist->n);
		cs.dispatch(res.x, res.y);
		cs.unbind();
	}

	void integrate_dir_sample::run() {
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
		time_this_wf_step;
		auto res = rc->resolution();

		bind_texture_as_image bind_0(camrays->rays,             0, true, false);
		bind_texture_as_image bind_1(camrays->intersections,    1, true, false);
		bind_texture_as_image bind_2(camrays->framebuffer,      2, true, true);
		bind_texture_as_image bind_3(shadowrays->rays,          3, true, false);
		bind_texture_as_image bind_4(shadowrays->intersections, 4, true, false);
		bind_texture_as_image bind_5(pdf->data,                 5, true, false);
		compute_shader &cs = integrate_dir_sample_shader;
		cs.bind();
		cs.uniform("w", res.x).uniform("h", res.y);
		cs.dispatch(res.x, res.y);
		cs.unbind();
	}

	void integrate_light_sample::run() {
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
		time_this_wf_step;
		auto res = rc->resolution();

		bind_texture_as_image bind_0(camrays->rays,             0, true, false);
		bind_texture_as_image bind_1(camrays->intersections,    1, true, false);
		bind_texture_as_image bind_2(camrays->framebuffer,      2, true, true);
		bind_texture_as_image bind_3(shadowrays->rays,          3, true, false);
		bind_texture_as_image bind_4(shadowrays->intersections, 4, true, false);
		bind_texture_as_image bind_5(pdf->data,                 5, true, false);
		bind_texture_as_image bind_6(light_col->data,           6, true, false);
		compute_shader &cs = integrate_light_sample_shader;
		cs.bind();
		cs.uniform("w", res.x).uniform("h", res.y);
		cs.dispatch(res.x, res.y);
		cs.unbind();
	}
}
