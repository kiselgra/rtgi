#pragma once

#include <glm/glm.hpp>
#include <utility>
#include <string>
#include <ostream>

using glm::vec3;
using glm::vec2;
using glm::vec4;
using glm::ivec4;
using std::pair;
using std::tuple;

constexpr float pi = float(M_PI);
constexpr float one_over_pi = (1.0 / M_PI);
constexpr float one_over_2pi = (1.0 / (2*M_PI));
constexpr float one_over_4pi = (1.0 / (4*M_PI));


/*! \brief A ray holds its origin, direction and inverse direction (o, d, id) as well
 *         as its minimal and maximal parameter range.
 */
struct ray {
	static constexpr float eps = 1e-4f;
	vec3 o, d, id;
	float t_min = eps, t_max = FLT_MAX;
	ray(const vec3 &o, const vec3 &d) : o(o), d(d), id(1.0f/d.x, 1.0f/d.y, 1.0f/d.z) {}
	ray() {}
	void length_exclusive(float d) { t_max = d - eps; }
};

/*! \brief A vertex is a point in space with associated normal and texture coordinate.
 */
struct vertex {
	vec3 pos;   //!< vertex position in space
	vec3 norm;  //!< vertex normal (surface orientation)
	vec2 tc;    //!< texture coordinate
};

/*! \brief A trinalge references three vertices (from the scene's vertex data, cf scene::vertices)
 *         as well as its surface properties (via material_id, cf scene::materials).
 */
struct triangle {
	uint32_t a, b, c;
	uint32_t material_id;
};

/*! \brief A ray/triangle intersection holds the distance along the ray to where the
 *         triangle is intersected and two of the intersection's barycentric coordinates.
 *         Ref referes to the triangle (cf triangle abeove and scene::triangles).
 */
struct triangle_intersection {
	typedef unsigned int uint;
	float t, beta, gamma;
	uint ref;
	triangle_intersection() : t(FLT_MAX), ref(0) {
	}
	explicit triangle_intersection(uint t) : t(FLT_MAX), ref(t) {
	}
	bool valid() const {
		return t != FLT_MAX;
	}
	void reset() {
		t = FLT_MAX;
		ref = 0;
	}
	vec3 barycentric_coord() const {
		vec3 bc;
		bc.x = 1.0 - beta - gamma;
		bc.y = beta;
		bc.z = gamma;
		return bc;
	}
};

class scene;
class material;

/*  \brief Ray/Geometry intersections are represented as differential geometry patches
 *
 *  That is, a locally flat, infinitessimally small part of a surface.
 */
struct diff_geom {
	const vec3 x;           // position in space
	vec3 ng, ns;            // geometric normal, shading normal, TODO: currently set to be equal!
	const vec2 tc;          // texture coordinate
	const uint32_t tri;     // reference to triangle
	const material *mat;    // reference to triangle's material
	diff_geom(const triangle_intersection &is, const scene &scene);

	vec3 albedo() const;   // evaluates the surface albedo (including texture lookup)
	float opacity() const; // evaluates the surface opacity (including texture lookup)
private:
	diff_geom(const vertex &a, const vertex &b, const vertex &c, const material *m, const triangle_intersection &is, const scene &scene);
	diff_geom(const triangle &tri, const triangle_intersection &is, const scene &scene);
};

/*  \brief Ray tracing interface.
 *
 *  Do not confuse with \ref gi_algorithm, a ray tracer's job is to compute ray intersections with scene geometry,
 *  nothing more.
 *
 */
class ray_tracer {
protected:
	::scene *scene;
public:
	virtual bool interprete(const std::string &command, std::istringstream &in) { return false; }
	virtual ~ray_tracer() {}
};

class individual_ray_tracer : public ray_tracer {
protected:
	::scene *scene;
public:
	virtual void build(::scene *) = 0;
	virtual triangle_intersection closest_hit(const ray &) = 0;
	virtual bool any_hit(const ray &) = 0;
	virtual bool interprete(const std::string &command, std::istringstream &in) { return false; }
};


static inline std::ostream& operator<<(std::ostream &o, const vec2 &v) { o << v.x << " " << v.y; return o; }
static inline std::ostream& operator<<(std::ostream &o, const vec3 &v) { o << v.x << " " << v.y << " " << v.z; return o; }
static inline std::ostream& operator<<(std::ostream &o, const vec4 &v) { o << v.x << " " << v.y << " " << v.z << " " << v.w; return o; }

static inline std::ostream& operator<<(std::ostream &o, const glm::ivec2 &v) { o << v.x << " " << v.y; return o; }

