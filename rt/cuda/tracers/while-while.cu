#include "while-while.h"

#include "libgi/timer.h"

#include <iostream>

#include "kernels.h"
#include "cuda-helpers.h"

namespace wf{
	namespace cuda{
		void whilewhile::compute_hit(bool anyhit) {
			auto *rt = dynamic_cast<wf::cuda::whilewhile*>(rc->scene.batch_rt);
			assert(rt != nullptr);
			int2 resolution{rc->resolution().x, rc->resolution().y};
			whilewhileTrace<<<NUM_BLOCKS_FOR_RESOLUTION(resolution), DESIRED_BLOCK_SIZE>>>(resolution,
															rd->rays.device_memory,
															rd->rays.tex,
															sd->vertex_pos.device_memory,
															sd->vertex_pos.tex,
															sd->triangles.device_memory,
															sd->triangles.tex,
															rt->bvh_index.device_memory,
															rt->bvh_index.tex,
															rt->bvh_nodes.device_memory,
															rt->bvh_nodes.tex,
															rd->intersections.device_memory,
															anyhit);
			CHECK_CUDA_ERROR(cudaGetLastError());
  			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		}
	}
}
