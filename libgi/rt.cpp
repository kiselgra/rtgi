#include "rt.h"
#include "scene.h"

using namespace glm;

	
diff_geom::diff_geom(const vertex &a, const vertex &b, const vertex &c,
					 const material *m, const triangle_intersection &is, const scene &scene)
 : x ((1.0f-is.beta-is.gamma)*a.pos  + is.beta*b.pos  + is.gamma*c.pos),
   ns(normalize((1.0f-is.beta-is.gamma)*a.norm + is.beta*b.norm + is.gamma*c.norm)),
   tc((1.0f-is.beta-is.gamma)*a.tc   + is.beta*b.tc   + is.gamma*c.tc),
   ng(normalize(cross(b.pos-a.pos, c.pos-a.pos))),
   tri(is.ref),
   mat(m) {
}

diff_geom::diff_geom(const triangle &t, const triangle_intersection &is, const scene &scene)
 : diff_geom(scene.vertices[t.a], scene.vertices[t.b], scene.vertices[t.c],
			 &scene.materials[t.material_id], is, scene) {
}

diff_geom::diff_geom(const triangle_intersection &is, const scene &scene)
 : diff_geom(scene.triangles[is.ref], is, scene) {
} 

vec3 diff_geom::albedo() const {
	if (mat->albedo_tex)
		return mat->albedo_tex->sample(tc);
	return mat->albedo;
}

float diff_geom::opacity() const {
	if (mat->albedo_tex)
		return mat->albedo_tex->sample(tc).w;
	return 1.0f;
};
