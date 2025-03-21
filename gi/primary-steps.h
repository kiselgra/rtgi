#pragma once
namespace wf
{

	class initialize_framebuffer : public step
	{
	public:
		static constexpr char id[] = "initialize framebuffer";

		virtual void use(raydata *rd) = 0;
	};
	namespace wire
	{

		template <typename RD>
		class initialize_framebuffer : public wf::initialize_framebuffer
		{
		public:
			using wf::initialize_framebuffer::initialize_framebuffer;
			RD *rd = nullptr;

			bool properly_wired()
			{
				return rd;
			}

			void use(raydata *rd)
			{
				this->rd = dynamic_cast<RD *>(rd);
			}
		};
	}
	//This should be combined with the initialize_framebuffer step

	class download_framebuffer : public step
	{
	public:
		static constexpr char id[] = "download framebuffer";

		virtual void use(raydata *rd) = 0;
	};
	namespace wire
	{

		template <typename RD>
		class download_framebuffer : public wf::download_framebuffer
		{
		public:
			using wf::download_framebuffer::download_framebuffer;
			RD *rd = nullptr;

			bool properly_wired()
			{
				return rd;
			}

			void use(raydata *rd)
			{
				this->rd = dynamic_cast<RD *>(rd);
			}
		};
	}

	class copy_to_preview : public step
	{
	public:
		static constexpr char id[] = "copy to preview";

		virtual void use(raydata *rd) = 0;
	};
	namespace wire
	{

		template <typename RD>
		class copy_to_preview : public wf::copy_to_preview
		{
		public:
			using wf::copy_to_preview::copy_to_preview;
			RD *rd = nullptr;

			bool properly_wired()
			{
				return rd;
			}

			void use(raydata *rd)
			{
				this->rd = dynamic_cast<RD *>(rd);
			}
		};
	}

	class sample_camera_rays : public step
	{
	public:
		static constexpr char id[] = "sample camera rays";

		virtual void use(raydata *rd) = 0;
	};
	namespace wire
	{

		template <typename RD>
		class sample_camera_rays : public wf::sample_camera_rays
		{
		public:
			using wf::sample_camera_rays::sample_camera_rays;
			RD *rd = nullptr;

			bool properly_wired()
			{
				return rd;
			}

			void use(raydata *rd)
			{
				this->rd = dynamic_cast<RD *>(rd);
			}
		};
	}

	class add_hitpoint_albedo : public step
	{
	public:
		static constexpr char id[] = "add hitpoint albedo";

		virtual void use(raydata *sample_rays) = 0;
	};
	namespace wire
	{

		template <typename RD>
		class add_hitpoint_albedo : public wf::add_hitpoint_albedo
		{
		public:
			using wf::add_hitpoint_albedo::add_hitpoint_albedo;
			RD *sample_rays = nullptr;

			bool properly_wired()
			{
				return sample_rays;
			}

			void use(raydata *sample_rays)
			{
				this->sample_rays = dynamic_cast<RD *>(sample_rays);
			}
		};
	}
	//This should be combined with the add_hitpoint_albedo step

	class add_hitpoint_albedo_to_framebuffer : public step
	{
	public:
		static constexpr char id[] = "add hitpoint albedo secondary";

		virtual void use(raydata *sample_rays) = 0;
	};
	namespace wire
	{

		template <typename RD>
		class add_hitpoint_albedo_to_framebuffer : public wf::add_hitpoint_albedo_to_framebuffer
		{
		public:
			using wf::add_hitpoint_albedo_to_framebuffer::add_hitpoint_albedo_to_framebuffer;
			RD *sample_rays = nullptr;

			bool properly_wired()
			{
				return sample_rays;
			}

			void use(raydata *sample_rays)
			{
				this->sample_rays = dynamic_cast<RD *>(sample_rays);
			}
		};
	}

	class add_hitpoint_normal_to_framebuffer : public step
	{
	public:
		static constexpr char id[] = "add hitpoint normal secondary";

		virtual void use(raydata *sample_rays) = 0;
	};
	namespace wire
	{

		template <typename RD>
		class add_hitpoint_normal_to_framebuffer : public wf::add_hitpoint_normal_to_framebuffer
		{
		public:
			using wf::add_hitpoint_normal_to_framebuffer::add_hitpoint_normal_to_framebuffer;
			RD *sample_rays = nullptr;

			bool properly_wired()
			{
				return sample_rays;
			}

			void use(raydata *sample_rays)
			{
				this->sample_rays = dynamic_cast<RD *>(sample_rays);
			}
		};
	}
	namespace wire
	{

		template <typename RD>
		class find_closest_hits : public wf::find_closest_hits
		{
		public:
			using wf::find_closest_hits::find_closest_hits;
			RD *rd = nullptr;

			bool properly_wired()
			{
				return rd;
			}

			void use(raydata *rd)
			{
				this->rd = dynamic_cast<RD *>(rd);
			}

			void run() override 
			{
				rt->use(rd);
				wf::find_closest_hits::run();
			}
		};
	}
	namespace wire
	{

		template <typename RD>
		class find_any_hits : public wf::find_any_hits
		{
		public:
			using wf::find_any_hits::find_any_hits;
			RD *rd = nullptr;

			bool properly_wired()
			{
				return rd;
			}

			void use(raydata *rd)
			{
				this->rd = dynamic_cast<RD *>(rd);
			}

			void run() override 
			{
				rt->use(rd);
				wf::find_any_hits::run();
			}
		};
	}
}