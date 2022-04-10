#include "rni.h"

#include "libgi/timer.h"

#include <curand_kernel.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>

#include <iostream>

#include "kernels.h"
#include "cuda-helpers.h"

namespace wf {
	namespace cuda {
		// batch_cam_ray_setup::batch_cam_ray_setup() {}

		void batch_cam_ray_setup::run() {
			time_this_block(batch_cam_ray_setup);
			auto *rt = dynamic_cast<batch_rt*>(rc->scene.batch_rt);
			assert(rt != nullptr);

			int2 resolution{rc->resolution().x, rc->resolution().y};

			if (rt->use_incoherence) {
				// find max/min scene extent (scene size)
				float3 scene_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
				float3 scene_min = {FLT_MAX, FLT_MAX, FLT_MAX};
				for (auto vertex : rc->scene.vertices) {
					if (vertex.pos.x < scene_min.x) scene_min.x = vertex.pos.x;
					if (vertex.pos.y < scene_min.y) scene_min.y = vertex.pos.y;
					if (vertex.pos.z < scene_min.z) scene_min.z = vertex.pos.z;
					if (vertex.pos.x > scene_max.x) scene_max.x = vertex.pos.x;
					if (vertex.pos.y > scene_max.y) scene_max.y = vertex.pos.y;
					if (vertex.pos.z > scene_max.z) scene_max.z = vertex.pos.z;
				}
				float3 scene_size = { abs(scene_max.x - scene_min.x), abs(scene_max.y - scene_min.y), abs(scene_max.z - scene_min.z)};

				// find biggest scene dimension and set sphere centers
				float r_max;
				float3 sphere1, sphere2;
				if (scene_size.x >= scene_size.y && scene_size.x >= scene_size.z) {
					r_max = scene_size.x/4;
					// in the two smaller dimensions, center the sphere. in the largest dimension position one sphere in one quarter point, the other sphere in the other
					sphere1 = { scene_min.x + scene_size.x/4, scene_min.y + scene_size.y/2 , scene_min.z + scene_size.z/2 };
					sphere2 = { scene_max.x - scene_size.x/4, scene_min.y + scene_size.y/2 , scene_min.z + scene_size.z/2 };
				}
				else if (scene_size.y >= scene_size.x && scene_size.y >= scene_size.z) {
					r_max = scene_size.y/4;
					sphere1 = { scene_min.x + scene_size.x/2, scene_min.y + scene_size.y/4 , scene_min.z + scene_size.z/2 };
					sphere2 = { scene_min.x + scene_size.x/2, scene_max.y - scene_size.y/4 , scene_min.z + scene_size.z/2 };
				}
				else if (scene_size.z >= scene_size.x && scene_size.z >= scene_size.y) {
					r_max = scene_size.z/4;
					sphere1 = { scene_min.x + scene_size.x/2, scene_min.y + scene_size.y/2 , scene_min.z + scene_size.z/4 };
					sphere2 = { scene_min.x + scene_size.x/2, scene_min.y + scene_size.y/2 , scene_max.z - scene_size.z/4 };
				}

				// Setup RNG ()
				int blocks = 200; // max number for MTGP RNG
				int threads = 256; // max number for MTGP RNG
				global_memory_buffer<curandStateMtgp32> rand_state("rand_state", blocks);
				global_memory_buffer<mtgp32_kernel_params> rng_params("rng_params", 1);
				curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, rng_params.device_memory);
				curandMakeMTGP32KernelState(rand_state.device_memory, mtgp32dc_params_fast_11213, rng_params.device_memory, 200, 1234);

				setup_ray_incoherent<<<blocks, threads>>>(resolution,
														  rt->rd->rays.device_memory,
														  sphere1, sphere2,
														  rt->incoherence_r1, rt->incoherence_r2,
														  r_max,
														  rand_state.device_memory);
				CHECK_CUDA_ERROR(cudaGetLastError(), "");
				CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "");
			}
			else {
				camera &cam = rc->scene.camera;
				vec3 U = cross(cam.dir, cam.up);
				vec3 V = cross(U, cam.dir);
				setup_ray<<<NUM_BLOCKS_FOR_RESOLUTION(resolution), DESIRED_BLOCK_SIZE>>>(U, V, cam.near_w, cam.near_h, resolution,
																						 float3{cam.pos.x, cam.pos.y, cam.pos.z},
																						 float3{cam.dir.x, cam.dir.y, cam.dir.z},
																						 rt->rd->rays.device_memory);
				CHECK_CUDA_ERROR(cudaGetLastError(), "");
				CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "");
			}
			CHECK_CUDA_ERROR(cudaGetLastError(), "");
			CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "");
		}

		void store_hitpoint_albedo_cpu::run() {
			time_this_block(store_hitpoint_albedo_cpu);
			auto res = rc->resolution();

			auto *rt = dynamic_cast<batch_rt*>(rc->scene.batch_rt);
			rt->rd->intersections.download();
			::triangle_intersection *is = (::triangle_intersection*)rt->rd->intersections.host_data.data();

			#pragma omp parallel for
			for (int y = 0; y < res.y; ++y)
				for (int x = 0; x < res.x; ++x) {
					vec3 radiance(0);
					::triangle_intersection &hit = is[y*res.x+x];
					// std::cout << "is #" << (y*res.x+x) << " t" << hit.t << " b" << hit.beta << " g" << hit.gamma << " r" << hit.ref << std::endl;
					if (hit.valid()) {
						diff_geom dg(hit, rc->scene);
						radiance += dg.albedo();
					}
					// radiance *= one_over_samples;
					rc->framebuffer.color(x,y) = vec4(radiance, 1);
				}
		}
		
		void store_hitpoint_albedo::run() {
			time_this_block(store_hitpoint_albedo);
			auto res = int2{rc->resolution().x, rc->resolution().y};
			auto *rt = dynamic_cast<batch_rt*>(rc->scene.batch_rt);

			compute_hitpoint_albedo<<<NUM_BLOCKS_FOR_RESOLUTION(res), DESIRED_BLOCK_SIZE>>>(res,
																							rt->rd->intersections.device_memory,
																							rt->sd->triangles.device_memory,
																							rt->sd->materials.device_memory,
																							rt->rd->framebuffer.device_memory);
			CHECK_CUDA_ERROR(cudaGetLastError(), "");
			CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "");

			rt->rd->framebuffer.download();
			float4 *fb = rt->rd->framebuffer.host_data.data();

			#pragma omp parallel for
			for (int y = 0; y < res.y; ++y)
				for (int x = 0; x < res.x; ++x) {
					float4 c = fb[y*res.x+x];
					rc->framebuffer.color(x,y) = vec4(c.x, c.y, c.z, c.w);
				}
		}
	
	}
}

