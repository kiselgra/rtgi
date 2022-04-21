#include "base.h"
#include "rni.h"
#include "find-hit.h"

#include "opengl.h"

#include <stdexcept>
#include <vector>

using std::vector;

namespace wf {
	namespace gl {

		texture_support_mode_t texture_support_mode = PROPER_BINDLESS;

		timer::timer() {
		}

		void timer::start(const std::string &name) {
			GLuint q0;
			if (queries.find(name) == queries.end()) {
				GLuint q[2];
				glGenQueries(2, q);
				queries[name] = {q[0], q[1]};
				q0 = q[0];
			}
			else
				q0 = queries[name].first;
			glQueryCounter(q0, GL_TIMESTAMP);
		}
		
		void timer::stop(const std::string &name) {
			auto [q0,q1] = queries[name];
			glQueryCounter(q1, GL_TIMESTAMP);
			
			GLint available = GL_FALSE;
			while (available == GL_FALSE)
				glGetQueryObjectiv(q1, GL_QUERY_RESULT_AVAILABLE, &available);

			GLuint64 start, stop;
			glGetQueryObjectui64v(q0, GL_QUERY_RESULT, &start);
			glGetQueryObjectui64v(q1, GL_QUERY_RESULT, &stop);

			// funnel to stats_timer
			stats_timer.timers[0].times[name] += (stop - start);
			stats_timer.timers[0].counts[name]++;
		}

		timer gpu_timer;

		void scenedata::upload(scene *scene) {
			triangles.resize(scene->triangles.size(), reinterpret_cast<ivec4*>(scene->triangles.data()));

			int N = scene->vertices.size();
			vector<vertex> v(N);

			for (int i = 0; i < N; ++i) {
				v[i].pos = vec4(scene->vertices[i].pos, 1);
				v[i].norm = vec4(scene->vertices[i].norm, 0);
				v[i].tc = scene->vertices[i].tc;
			}
			vertices.resize(v);

			vector<material> mtl(scene->materials.size());
			vector<vec4> tex_data_hacky;
			for (int i = 0; i < scene->materials.size(); ++i) {
				material &m = mtl[i];
				m.emissive = vec4(scene->materials[i].emissive, 1);
				m.albedo = vec4(scene->materials[i].albedo, 1);
				m.albedo_tex = 0;
				if (scene->materials[i].albedo_tex) {
					if (texture_support_mode == PROPER_BINDLESS) {
						m.has_tex = 1;
						GLuint tex;
						glGenTextures(1, &tex);
						glBindTexture(GL_TEXTURE_2D, tex);
						glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, scene->materials[i].albedo_tex->w, scene->materials[i].albedo_tex->h, 0, GL_RGB, GL_FLOAT, scene->materials[i].albedo_tex->texel);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
						glBindTexture(GL_TEXTURE_2D, 0);
						m.albedo_tex = glGetTextureHandleARB(tex);
						glMakeTextureHandleResidentARB(m.albedo_tex);
						textures.push_back(tex);
					}
					else if (texture_support_mode == HACKY) {
						m.has_tex = 1;
						m.albedo_tex = tex_data_hacky.size();
						texture2d *src = scene->materials[i].albedo_tex;
						tex_data_hacky.push_back(vec4(src->w, src->h, 0, 0));
						for (int y = 0; y < src->h; ++y)
							for (int x = 0; x < src->w; ++x)
								tex_data_hacky.push_back(vec4(src->value(x, y),1));
					}
				}
				materials.resize(mtl);
			}
			if (texture_support_mode == HACKY)
				texture_data_hacky.resize(tex_data_hacky);
			glFinish();
		}
		
		scenedata::~scenedata() {
			materials.download();
			for (auto &m : materials.org_data)
				glMakeTextureHandleNonResidentARB(m.albedo_tex);
			for (auto tex : textures)
				glDeleteTextures(1, &tex);
		}

	}
}
