#pragma once
#include <string>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace gail{

	void init();

	//GLuint CCS(std::string);
	//GLuint CCP(std::vector<GLuint>);
	
	void set_spd(float);
	
	class layer{
	protected:
		std::string file;
		GLuint in, //doesnt exist within this context
			out, error, grad;
		void *ptr_in=0, *ptr_out=0, *ptr_error=0;
	public:
		layer(std::string);
		
		virtual GLuint operator>>(std::vector<float>); //forward provided with vector
		virtual GLuint operator<<(std::vector<float>); //backward provided with vector
		virtual GLuint operator>>(GLuint); //inlayer flow
		virtual GLuint operator<<(GLuint); //inlayer float
		virtual void update(); //update is based off speed
		virtual void save(); //save is just save
		virtual ~layer(); //lol
		
		void* get_out();
		
		//this is debug we dont really need it
		void* get_in();
		void* get_err();
	};
	
	class forward: public layer{
	protected:
		GLuint params, stats; //used
		void* ptr_params=0, *ptr_stats=0; //used
		unsigned int X,Y;
		forward(std::string, unsigned int, unsigned int, bool);
	public:
		forward(std::string, unsigned int x=1, unsigned int y=1);
		forward(std::string, std::vector<std::vector<float>>, std::vector<float>);
		
		GLuint operator>>(std::vector<float>) override;
		GLuint operator<<(std::vector<float>) override;
		GLuint operator>>(GLuint) override;
		GLuint operator<<(GLuint) override;
		void update() override;
		void save() override;
		
		~forward();
	};//VALID
	
	class activate: public layer{
	public:
		enum type{SIGMOID, TANH, RELU};
	private:
		type th;
		unsigned int X;
		void compute_forward(GLuint&);
		void compute_back(GLuint&);
	public:
		activate(type, unsigned int);
		GLuint operator>>(std::vector<float>) override;
		GLuint operator<<(std::vector<float>) override;
		GLuint operator>>(GLuint) override;
		GLuint operator<<(GLuint) override;
		~activate();
	};//VALID
	
	class convolute: public forward{
	protected:
		unsigned int I, O, img_in[2], img_out[2], S, P; //additional parameters
	public:
		convolute(std::string, unsigned int x=1, unsigned int y=1, unsigned int ii=1, unsigned int oo=1);
		convolute(std::string, std::vector<std::vector<std::vector<std::vector<float>>>>, std::vector<float>);
		void set_image_params(unsigned int, unsigned int, unsigned int, unsigned int);
		
		GLuint operator>>(std::vector<float>) override;
		GLuint operator<<(std::vector<float>) override;
		GLuint operator>>(GLuint) override;
		GLuint operator<<(GLuint) override;
		void update() override;
		void save() override;
		
		~convolute();
	}; //this is where we get both frustrated and excited
	
	class pooling:public layer{
		GLuint store, stats; void *ptr_store=0, *ptr_stats=0;
		unsigned int X,Y,Z;
	public:
		pooling(unsigned int, unsigned int, unsigned int);
		GLuint operator>>(std::vector<float>) override;
		GLuint operator<<(std::vector<float>) override;
		GLuint operator>>(GLuint) override;
		GLuint operator<<(GLuint) override;
		~pooling();
	};
	
	class merger{
		GLuint out;
		std::vector<GLuint> stats, err;
		std::vector<void*> ptr_stats, ptr_err;
		void* ptr_out;
		std::vector<unsigned int> sizes;
	public:
		merger(std::vector<unsigned int>, unsigned int);
		GLuint operator>>(std::vector<std::vector<float>>);
		GLuint operator>>(std::vector<GLuint>);
		std::vector<GLuint> operator<<(GLuint);
		std::vector<GLuint> operator<<(std::vector<float>);
		
		std::vector<void*> get_errors();
		void* get_out();
		~merger();
	};
	
	class buff{
		GLuint out;
		void* ptr_out;
	public:
		buff(unsigned int);
		GLuint operator>>(std::vector<float>);
		~buff();
	};
	
	class reccurent{
	
	};
	
	class lstm{
	
	};
}
