#pragma once

#include "libgi/algorithm.h"
#include "libgi/wavefront-rt.h"
#include "libgi/material.h"

/* \brief Display the color (albedo) of the surface closest to the given ray.
 *
 * - x, y are the pixel coordinates to sample a ray for.
 * - samples is the number of samples to take
 * - render_context holds contextual information for rendering (e.g. a random number generator)
 *
 */
class primary_hit_display : public recursive_algorithm {
public:
	gi_algorithm::sample_result sample_pixel(uint32_t x, uint32_t y, uint32_t samples) override;
};

#ifndef RTGI_SKIP_LOCAL_ILLUM
class local_illumination : public recursive_algorithm {
public:
	gi_algorithm::sample_result sample_pixel(uint32_t x, uint32_t y, uint32_t samples) override;
};
#endif

#ifndef RTGI_SKIP_WF
#include "primary-steps.h"
namespace wf {
	template<typename T>
	class primary_hit_display : public T  {
		raydata *rd = nullptr;
	public:
		primary_hit_display();
	};
}
#endif
