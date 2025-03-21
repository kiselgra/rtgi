#pragma once

#include "libgi/wavefront-rt.h"
#include "rt/cpu/bvh.h"
#include "libgi/timer.h"

#include "cuda-helpers.h"

#include <cuda_runtime_api.h>
#define MULTIPROCESSOR_COUNT               82	// fixed (device dependent, 82 for RTX3090)
#define WARPSIZE                           32	// fixed (architecture-dependent)
#define MAX_WARPS_PER_MULTIPROCESSOR       48	// fixed (architecture-dependent)
#define DESIRED_WARPS_PER_BLOCK            6	// mostly arbitrary, doesn't really matter
#define DESIRED_THREADS_PER_BLOCK 		   DESIRED_WARPS_PER_BLOCK*32
#define DESIRED_BLOCKS_PER_MULTIPROCESSOR  MAX_WARPS_PER_MULTIPROCESSOR/DESIRED_WARPS_PER_BLOCK		// = 8
#define DESIRED_BLOCKS_COUNT               MULTIPROCESSOR_COUNT*DESIRED_BLOCKS_PER_MULTIPROCESSOR				// = 656
#define DESIRED_BLOCK_SIZE                 dim3(WARPSIZE, DESIRED_WARPS_PER_BLOCK, 1)
#define NUM_BLOCKS_FOR_RESOLUTION(resolution) dim3((resolution.x/DESIRED_BLOCK_SIZE.x) + 1, (resolution.y/DESIRED_BLOCK_SIZE.y) + 1, 1)

namespace wf {
	namespace cuda {
		
		//! \brief Take time of asynchronously running CUDA calls.
		struct timer : public wf::timer {
			std::map<std::string, std::pair<cudaEvent_t,cudaEvent_t>> events;
			void start(const std::string &name) override;
			void stop(const std::string &name) override;
			void synchronize() override;
		};


		struct __align__(16) tri_is {
			float t;
			float beta;
			float gamma;
			unsigned int ref;

			__device__ tri_is() : t(FLT_MAX), beta(-1), gamma(-1), ref(0) {};
			__device__ tri_is(float t, float beta, float gamma, unsigned int ref) : t(t), beta(beta), gamma(gamma), ref(ref) {};
			__device__ __inline__ bool valid() { return t != FLT_MAX; }
		};

		struct __align__(16) simple_bvh_node /*: public node*/ {
			float3 box_l_min;
			float3 box_l_max;
			int link_l;
			float3 box_r_min;
			float3 box_r_max;
			int link_r;
			__host__ __device__ simple_bvh_node() {};
			__host__ simple_bvh_node(::aabb box_l, ::aabb box_r, int link_l, int link_r) : link_l(link_l), link_r(link_r) {
				box_l_min = {box_l.min.x, box_l.min.y, box_l.min.z};
				box_l_max = {box_l.max.x, box_l.max.y, box_l.max.z};
				box_r_min = {box_r.min.x, box_r.min.y, box_r.min.z};
				box_r_max = {box_r.max.x, box_r.max.y, box_r.max.z};
			};
			__host__ simple_bvh_node(binary_bvh_tracer<bbvh_triangle_layout::flat, bbvh_esc_mode::off>::node n) : simple_bvh_node(n.box_l, n.box_r, n.link_l, n.link_r) {};
			// __host__ simple_bvh_node(binary_bvh_tracer<bbvh_triangle_layout::indexed, bbvh_esc_mode::off>::node n) : simple_bvh_node(n.box_l, n.box_r, n.link_l, n.link_r) {};
			// __host__ simple_bvh_node(binary_bvh_tracer<bbvh_triangle_layout::flat, bbvh_esc_mode::on>::node n) : simple_bvh_node(n.box_l, n.box_r, n.link_l, n.link_r) {};
			// __host__ simple_bvh_node(binary_bvh_tracer<bbvh_triangle_layout::indexed, bbvh_esc_mode::on>::node n) : simple_bvh_node(n.box_l, n.box_r, n.link_l, n.link_r) {};
			__host__ __device__ __inline__ bool inner() const { return link_r >= 0; }
			__host__ __device__ __inline__ int32_t tri_offset() const { return -link_l; }
			__host__ __device__ __inline__ int32_t tri_count()  const { return -link_r; }
		};

		struct __align__(16) compact_bvh_node {
			/* Für innere Knoten gilt:
			 *    child_index < 0 => Child ist ein Blattknoten (mit dem Index -child_index)
			 * Für Blattknoten gilt:
			 *   child_index0 = -tri_offset
			 *   child_index1 = -tri_count
			 */
			float4 data1;	// (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
			float4 data2;	// (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
			float4 data3;	// (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
			float4 data4;	// child_index0, child_index1
			__host__ __device__ compact_bvh_node() {};
		};

		struct compact_bvh_node_builder {
			static std::vector<compact_bvh_node> build(std::vector<binary_bvh_tracer<bbvh_triangle_layout::indexed, bbvh_esc_mode::on>::node> nodes);
		};

		struct buffer {
			std::string name;
			unsigned size = 0;	// number of elements! not bytes

			buffer(std::string name, unsigned size)
			: name(name), size(size) {
			}
			virtual ~buffer() {
			}
			virtual void print() {
			}
		};

		enum buffer_copy_mode_shallow { shallow_non_owning_copy };
		enum buffer_copy_mode_duplicate { mem_duplicating_copy_only }; // does not copy contents, but allocates new storage

		template<typename T> class global_memory_buffer : public buffer {
		protected:
			// those are protected/deleted to ensure buffers are explicitly duplicated or explicitly aliased
			global_memory_buffer(const global_memory_buffer &) = default;
			global_memory_buffer& operator=(const global_memory_buffer &) = delete;
		public:
			std::vector<T> host_data;
			T *device_memory = nullptr;
			bool owns_mem = true;

			global_memory_buffer(std::string name, unsigned size)
			: buffer(name, 0) {
				if (size > 0) resize(size);
			}

			global_memory_buffer(global_memory_buffer &org, buffer_copy_mode_shallow)
			: global_memory_buffer(org) {
				owns_mem = false;
			}

			global_memory_buffer(const global_memory_buffer &org, buffer_copy_mode_duplicate)
			: global_memory_buffer(org.name, org.size) {
			}

			global_memory_buffer(global_memory_buffer &&tmp) : buffer(std::move(tmp)), device_memory(tmp.device_memory), owns_mem(tmp.owns_mem), host_data(std::move(tmp.host_data)) {
				tmp.device_memory = nullptr;
			}
			global_memory_buffer& operator=(global_memory_buffer &&tmp) {
				name = std::move(tmp.name);
				size = tmp.size;
				device_memory = tmp.device_memory; tmp.device_memory = nullptr;
				owns_mem = tmp.owns_mem;
				host_data = std::move(tmp.host_data);
				return *this;
			}

			~global_memory_buffer() {
				if (device_memory && owns_mem) {
					CHECK_CUDA_ERROR(cudaFree(device_memory),name);
					CHECK_CUDA_ERROR(cudaDeviceSynchronize(),name);
				}
				device_memory = nullptr;
				size = 0;
			}
			
			explicit operator CUdeviceptr() {
				return (CUdeviceptr) device_memory;
			}

			int size_in_bytes() const {
				return size * sizeof(T);
			}

			void resize(int size) {
				assert(owns_mem);
				if (this->size == size) return;
				if (device_memory) {
					CHECK_CUDA_ERROR(cudaFree(device_memory), name);
					CHECK_CUDA_ERROR(cudaDeviceSynchronize(), name);
					device_memory = nullptr;
					this->size = 0;
				}

				T *new_device_memory = nullptr;
				CHECK_CUDA_ERROR(cudaMalloc((void**)&new_device_memory, size*sizeof(T)), name);
				CHECK_CUDA_ERROR(cudaDeviceSynchronize(), name);
				this->device_memory = new_device_memory;
				this->size = size;
			}

			void upload(const std::vector<T> &data) {
				upload(data.size(), data.data());
			}

			void upload(int size, const T *data) {
				resize(size);
				host_data.resize(size);

				std::copy(data, data + size, host_data.begin());

				CHECK_CUDA_ERROR(cudaMemcpy(device_memory, host_data.data(), size*sizeof(T), cudaMemcpyHostToDevice), name);
				CHECK_CUDA_ERROR(cudaGetLastError(), name);
				CHECK_CUDA_ERROR(cudaDeviceSynchronize(), name);
			}
			void download() {
				time_this_block(download_membuffer);
				if (host_data.size() != size)
					host_data.resize(size);

				CHECK_CUDA_ERROR(cudaMemcpy(host_data.data(), device_memory, size*sizeof(T), cudaMemcpyDeviceToHost), name);
				CHECK_CUDA_ERROR(cudaDeviceSynchronize(), name);
			}
			void free_host_data() {
				std::vector<T>{}.swap(host_data);
			}
		};

		template<typename T> class texture_buffer : public global_memory_buffer<T> {
		protected:
			texture_buffer(const texture_buffer &) = default;
			texture_buffer& operator=(const texture_buffer &) = delete;
		public:
			cudaTextureObject_t tex = 0;

			texture_buffer(std::string name, unsigned size)
			: global_memory_buffer<T>(name, size) {
				if (size > 0)
					update_texture();
			}
			
			texture_buffer(texture_buffer &org, buffer_copy_mode_shallow) : texture_buffer(org) {
				this->owns_mem = false;
			}
			
			texture_buffer(const texture_buffer &org, buffer_copy_mode_duplicate) : texture_buffer(org.name, org.size) {
			}

			texture_buffer(texture_buffer &&tmp) : global_memory_buffer<T>(tmp), tex(tmp.tex) {
				tmp.tex = 0;
			}

			texture_buffer& operator=(texture_buffer &&tmp) {
				global_memory_buffer<T>::operator=(std::move(tmp));
				tex = tmp.tex; tmp.tex = 0;
				return *this;
			}

			~texture_buffer() {
				if (tex != 0 && this->owns_mem)
					CHECK_CUDA_ERROR(cudaDestroyTextureObject(tex),this->name);
			}

			void update_texture() {
				assert(this->owns_mem);
				if (tex>0)
					CHECK_CUDA_ERROR(cudaDestroyTextureObject(tex),this->name);
				
				cudaResourceDesc res_desc = {};
				res_desc.resType = cudaResourceTypeLinear;
				res_desc.res.linear.devPtr = this->device_memory;
				res_desc.res.linear.sizeInBytes = this->size*sizeof(T);
				res_desc.res.linear.desc = cudaCreateChannelDesc<float4>();
				cudaTextureDesc tex_desc = {};
				memset(&tex_desc, 0, sizeof(tex_desc));
				tex_desc.addressMode[0] = cudaAddressModeClamp; // Wrap?
				tex_desc.addressMode[1] = cudaAddressModeClamp;
				tex_desc.addressMode[2] = cudaAddressModeClamp;
				tex_desc.filterMode = cudaFilterModePoint;
				tex_desc.readMode = cudaReadModeElementType;
				tex_desc.normalizedCoords = 0;

				CHECK_CUDA_ERROR(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr),this->name);
			}

			void resize(int size) {
				global_memory_buffer<T>::resize(size);
				update_texture();
			}

			void upload(const std::vector<T> &data) {
				upload(data.size(), data.data());
			}

			void upload(int size, const T *data) {
				global_memory_buffer<T>::upload(size, data);
				update_texture();
			}

			/*void download() {
				global_memory_buffer<T>::download();
			}*/
		};

		struct texture_image {
			std::string name;
			cudaArray *underlying = nullptr;
			cudaTextureObject_t tex = 0;
			int w, h;
			int pitch;

			texture_image(const texture2d<vec3> &base)
			: name(base.name + "-on-cuda"), w(base.w), h(base.h) {
				float4 *src = new float4[w*h];
				#pragma omp parallel for
				for (int y = 0; y < h; ++y)
					for (int x = 0; x < w; ++x)
						src[y*w+x] = float4{ base.texel[y*w+x].x, base.texel[y*w+x].y, base.texel[y*w+x].z, 0 };
				
				create_cuda_resource(src);
				delete [] src;
			}

			texture_image(const texture2d<vec4> &base)
			: name(base.name + "-on-cuda"), w(base.w), h(base.h) {
				float4 *src = new float4[w*h];
				#pragma omp parallel for
				for (int y = 0; y < h; ++y)
					for (int x = 0; x < w; ++x)
						src[y*w+x] = float4{ base.texel[y*w+x].x, base.texel[y*w+x].y, base.texel[y*w+x].z, base.texel[y*w+x].w };
				
				create_cuda_resource(src);
				delete [] src;
			}

			private:
				void create_cuda_resource(const float4 *src) {
					auto chan_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
					CHECK_CUDA_ERROR(cudaMallocArray(&underlying, &chan_desc, w, h), name);
					int spitch = w * sizeof(float4);
					CHECK_CUDA_ERROR(cudaMemcpy2DToArray(underlying, 0, 0, src, spitch, w*sizeof(float4), h, cudaMemcpyHostToDevice), name);
					cudaResourceDesc res_desc;
					memset(&res_desc, 0, sizeof(res_desc));
					res_desc.resType = cudaResourceTypeArray;
					res_desc.res.array.array = underlying;

					cudaTextureDesc tex_desc;
					memset(&tex_desc, 0, sizeof(tex_desc));
					tex_desc.addressMode[0] = cudaAddressModeWrap;
					tex_desc.addressMode[1] = cudaAddressModeWrap;
					tex_desc.filterMode = cudaFilterModeLinear;
					tex_desc.readMode = cudaReadModeElementType;
					tex_desc.normalizedCoords = 1;

					CHECK_CUDA_ERROR(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr), name);
				}
			// TODO cleanup missing
		};

		struct raydata : public wf::raydata {
			std::string name;
			int w, h;
			texture_buffer<float4> rays;
			global_memory_buffer<tri_is> intersections;
			global_memory_buffer<float4> framebuffer;

			raydata(glm::ivec2 dim) : raydata(dim.x, dim.y) {}
			raydata(int w, int h) : w(w), h(h),
									rays("rays", 2*w*h),
									intersections("intersections", w*h),
									framebuffer("framebuffer", w*h)	{
				  rc->call_at_resolution_change[this] = [this](int w, int h) {
					  this->w = w;
					  this->h = h;
					  this->rays.resize(2*w*h);
					  this->intersections.resize(w*h);
					  this->framebuffer.resize(w*h);
				  };
			}
			~raydata() {
				rc->call_at_resolution_change.erase(this);
			}
		};

		struct material {
			float4 albedo;
			float4 emissive;
			cudaTextureObject_t albedo_tex;
			float ior, roughness;
		};

		struct scenedata {
			int n_vertices = 0, n_triangles = 0;
			texture_buffer<float4> vertex_pos;
			texture_buffer<float4> vertex_norm;
			texture_buffer<float2> vertex_tc;
			texture_buffer<uint4> triangles;
			global_memory_buffer<material> materials;
			std::vector<texture_image> tex_images;
			scenedata() : vertex_pos("vertex_pos", 0),
						  vertex_norm("vertex_norm", 0),
						  vertex_tc("vertex_tc", 0),
						  triangles("triangles", 0),
						  materials("materials", 0)	{
			};
			scenedata(const scenedata &) = delete;
			scenedata(scenedata *org, buffer_copy_mode_shallow m) : vertex_pos(org->vertex_pos, m),
			                                                        vertex_norm(org->vertex_norm, m),
			                                                        vertex_tc(org->vertex_tc, m),
																	triangles(org->triangles, m),
																	materials(org->materials, m),
																	// tex_images not copied
																	n_vertices(org->n_vertices),
																	n_triangles(org->n_triangles) {
				this->org = org;
			}
			void upload(scene *scene);
			scenedata *org = nullptr;
		};

		struct cpu_bvh_builder_cuda_scene_traits {
			scenedata *s;
			typedef uint4 tri_t;
			int  triangles() { return s->triangles.host_data.size(); }
			tri_t triangle(int index) { return s->triangles.host_data[index]; }
			int triangle_a(int index) { return s->triangles.host_data[index].x; }
			int triangle_b(int index) { return s->triangles.host_data[index].y; }
			int triangle_c(int index) { return s->triangles.host_data[index].z; }
			glm::vec3 vertex_pos(int index) { float4 v = s->vertex_pos.host_data[index]; return glm::vec3(v.x,v.y,v.z); }
			void replace_triangles(std::vector<tri_t> &&new_tris) {
				s->triangles.host_data = new_tris;
			}
		};

		struct batch_rt : public batch_ray_tracer {
			raydata *rd = nullptr;
			bool use_incoherence = false;
			float incoherence_r1 = 0; // TODO
			float incoherence_r2 = 0;

			int bvh_max_tris_per_node = 4;
			std::string bvh_type = "sah";

			texture_buffer<wf::cuda::compact_bvh_node> bvh_nodes;
			texture_buffer<uint> bvh_index;

			batch_rt() : bvh_nodes("bvh_nodes", 0), bvh_index("index", 0) {
			}
			virtual void build(scenedata *scene);
			void use(wf::raydata *rays) override { 
			    rd = dynamic_cast<raydata*>(rays);
			}
			bool interprete(const std::string &command, std::istringstream &in) override;
			void compute_closest_hit() override {
				compute_hit(false);
			}
			void compute_any_hit() override {
				compute_hit(true);
			}
			virtual void compute_hit(bool anyhit) = 0;
		};

	}
}
