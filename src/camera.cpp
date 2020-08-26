#include "camera.h"

#include "cmdline.h"

#include <fstream>

using namespace glm;
using namespace std;

ray cam_ray(const camera &cam, int x, int y) {
	// find basis
	vec3 U = cross(cam.dir, cam.up);
	vec3 V = cross(cam.dir, U);
	// pixel position on near plane
	float u = ((float)x+0.5f)/(float)cam.w * 2.0f - 1.0f;	// \in (-1,1)
	float v = ((float)y+0.5f)/(float)cam.h * 2.0f - 1.0f;
	u = cam.near_w * u;	// \in (-near_w,near_w)
	v = cam.near_h * v;
	// near is implicitly at 1 (as per tanf above)
	return ray(cam.pos, normalize(cam.dir + U*u + V*v));
}

void test_camrays(const camera &camera) {
	ofstream out("test.obj");
	int i = 1;
	for (int y = 0; y < cmdline.vp_h; ++y)
		for (int x = 0; x < cmdline.vp_w; ++x) {
			ray ray = cam_ray(camera, x, y);
			out << "v " << ray.o.x << " " << ray.o.y << " " << ray.o.z << endl;
			out << "v " << ray.d.x << " " << ray.d.y << " " << ray.d.z << endl;
			out << "l " << i++ << " " << i++ << endl;
		}
}


