#pragma once

#include <GL/glew.h>
#include <string>
#include <vector>
#include <glm/glm.hpp>

#include <iostream>
#include <algorithm>

class shader {

	std::vector<GLuint> shaders;

public:
	std::string name;
	GLuint program = 0;

	shader(const std::string &name) : name(name) {
	}
	~shader() {
		glDeleteProgram(program);
		program = 0;
	}

	virtual void compile() = 0;

	void bind() {
		if (!program) compile();
		glUseProgram(program);
	}
	void unbind() {
		glUseProgram(0);
	}

	void compile_shader( const std::string &shader_src, GLenum shader_type);
	void link_and_validate();
	void delete_and_detatch_shaders() {
		if(!program) return;
		for (auto const shader : shaders) {
			if(!shader) continue;
			// after the program has been linked and validated we don't longer need the shader objects
			glDetachShader(program, shader);
			glDeleteShader(shader);
		}
		shaders.clear();
	}

	int uniform_location(const std::string &name) {
		return glGetUniformLocation(program, name.c_str());
	}
	shader& uniform(const std::string &name, int x) {
		glUniform1i(uniform_location(name), x);
		return *this;
	}
	shader& uniform(const std::string &name, int x, int y) {
		glUniform2i(uniform_location(name), x, y);
		return *this;
	}
	shader& uniform(const std::string &name, float x, float y) {
		int l = uniform_location(name);
		glUniform2f(uniform_location(name), x, y);
		return *this;
	}
	shader& uniform(const std::string &name, const glm::vec3 &v) {
		glUniform3f(uniform_location(name), v.x, v.y, v.z);
		return *this;
	}
};

class render_shader : public shader {

	std::string vertex_src, fragment_src;

public:
	render_shader(const std::string &name, const std::string &vertex_src, const std::string &fragment_src) 
		: shader(name), vertex_src(vertex_src), fragment_src(fragment_src)
	{
	}

	void compile() override {
		compile_shader(vertex_src, GL_VERTEX_SHADER);
		compile_shader(fragment_src, GL_FRAGMENT_SHADER);
		link_and_validate();
	}
};

class compute_shader : public shader {

	std::string compute_src;

public:
	compute_shader(const std::string &name, const std::string &compute_src) 
		: shader(name), compute_src(compute_src)
	{
	}

	void compile() override {
		compile_shader(compute_src, GL_COMPUTE_SHADER);
		link_and_validate();
	}

	void dispatch(int size_x, int size_y = 1, int size_z = 1);
};
