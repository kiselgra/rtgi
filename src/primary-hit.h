#pragma once

#include "algorithm.h"

class primary_hit_display : public gi_algorithm {
public:
	gi_algorithm::sample_result sample_pixel(uint32_t x, uint32_t y, uint32_t samples, const render_context &r) override;
};

class direct_light : public gi_algorithm {
public:
	gi_algorithm::sample_result sample_pixel(uint32_t x, uint32_t y, uint32_t samples, const render_context &r) override;
};
