#pragma once

#include "libgi/scene.h"
#include "libgi/intersect.h"

#include <vector>
#include <float.h>
#include <glm/glm.hpp>

// #define COUNT_HITS

#ifndef RTGI_SKIP_WF
/*! Here we are inconsistent and use the ::scene instead of wf::cpu::scene
 *  because this is code that is also run for the individual ray tracer.
 *
 *  TODO: will this cause problems?
 */
#endif
struct naive_bvh : public individual_ray_tracer {
	struct node {
		aabb box;
		uint32_t left, right;
		uint32_t triangle = (uint32_t)-1;
		//! is the node an inner node (as opposed to a leaf)
		bool inner() const { return triangle == (uint32_t)-1; }
	};

	std::vector<node> nodes;
	uint32_t root;
	void build(::scene *scene);
private:
	uint32_t subdivide(std::vector<triangle> &triangles, std::vector<vertex> &vertices, uint32_t start, uint32_t end);
	triangle_intersection closest_hit(const ray &ray) override;
	bool any_hit(const ray &ray) override;
};
	
/* Innere und Blattknoten werden durch trickserei unterschieden.
 * Für Blattknoten gilt:
 * - link_l = -tri_offset
 * - link_r = -tri_count
 */
struct bbvh_node {
	aabb box_l, box_r;
	int32_t link_l, link_r;
	
	bool inner() const { return link_r >= 0; }     //<! returns if the node is an inner node (as opposed to a leaf).
	int32_t tri_offset() const { return -link_l; } //<! gives the (leaf) node's offset into the scene's triangle data.
	int32_t tri_count()  const { return -link_r; } //<! a (leaf) node may hold more than one triangle.
	void tri_offset(int32_t offset) { link_l = -offset; }  // these two can be used to specify counts/offset
	void tri_count(int32_t count) { link_r = -count; }     // without having to take care of how they are represented.
};
static_assert(sizeof(bbvh_node) == 2*2*3*4+2*4);

#ifndef RTGI_SIMPLER_BBVH

constexpr const float ALPHA_THRESHOLD = 0.5f;

struct bvh {
	typedef bbvh_node node;
	uint32_t root;
	std::vector<node> nodes;
	std::vector<uint32_t> index;  // can be empty if we don't use indexing
};

enum class bbvh_triangle_layout { flat, indexed };
enum class bbvh_esc_mode { off, on };

template<bbvh_triangle_layout tr_layout, bbvh_esc_mode esc_mode, bool alpha_aware = false>
struct binary_bvh_tracer : public individual_ray_tracer {
	typedef bbvh_node node;

	template<bool cond, typename T>
    using variant = typename std::enable_if<cond, T>::type;

	::bvh bvh;
	enum binary_split_type {sm, om, sah};
	binary_split_type binary_split_type = om;

	// config options
	int max_triangles_per_node = 1;
	int number_of_planes = 1;
	bool should_export = false;

	binary_bvh_tracer();
	void build(::scene *scene) override;
	triangle_intersection closest_hit(const ray &ray) override;
	bool any_hit(const ray &ray) override;
	bool interprete(const std::string &command, std::istringstream &in) override;

private:
	void print_node_stats();
	void export_bvh(uint32_t node, uint32_t *id, uint32_t depth, const std::string &filename, int max_depth);

	// Get triangle index (if an index is used)
	template<bbvh_triangle_layout LO=tr_layout> 
	variant<LO==bbvh_triangle_layout::flat, int>
		triangle_index(int i) {
			return i;
		}
	
	template<bbvh_triangle_layout LO=tr_layout> 
	variant<LO!=bbvh_triangle_layout::flat, int>
		triangle_index(int i) {
			return bvh.index[i];
		}
};
#else
struct binary_bvh_tracer : public individual_ray_tracer {
	typedef bbvh_node node;

	std::vector<node> nodes;
	int32_t root = 0;

	enum binary_split_type {sm, om };
	binary_split_type split_type = om;

	// config options
	int max_triangles_per_node = 1;
	int number_of_planes = 1;
	
	// helper
	vec3 center(const triangle &t);
	
	binary_bvh_tracer(binary_split_type type);
	void build(::scene *scene) override;
	triangle_intersection closest_hit(const ray &ray) override;
	bool any_hit(const ray &ray) override;
	bool interprete(const std::string &command, std::istringstream &in) override;

private:
	/*! recursively builds the BVH (tree).
	 *  - start and end simply index into the scene's triangle list
	 *  - the box handed in is always the box (holding those trinalges) that should be subdivided
	 *  - before calling this recursively, compute the bounding boxes of the child nodes
	 *  - use this to determine the split axis
	 *  
	 *  note: which of those two is called can be configured via script, see binary_bvh_tracer::build
	 *  in your implementation, don't forget to call the function you are in (easy to miss by copy&paste)
	 */
	int32_t subdivide_om(int start, int end, aabb box); 
	int32_t subdivide_sm(int start, int end, aabb box); 
};
#endif

