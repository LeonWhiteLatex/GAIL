#include "layers.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <cstring>
#define GL_SSB GL_SHADER_STORAGE_BUFFER
#define GL_UBO GL_UNIFORM_BUFFER
#define PRW GL_MAP_PERSISTENT_BIT|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT|GL_MAP_COHERENT_BIT

namespace gail{
	template<typename T>
	using SV=std::vector<T>;
	typedef std::string STR; //std::string is STR
	typedef unsigned int uint;
	
	float spd=0.5;
	
	class wrapper{
		GLuint fun;
	public:
		wrapper(){}
		wrapper(GLuint fun):fun(fun){}
		void operator()(GLuint &stats, GLuint &in, GLuint &param, GLuint &out,
		uint x=1, uint y=1, uint z=1){
			glUseProgram(fun);
			glBindBufferBase(GL_UBO, 0, stats);
			glBindBufferBase(GL_SSB, 0, in);
			glBindBufferBase(GL_SSB, 1, out);
			glBindBufferBase(GL_SSB, 2, param);
			//std::cout<<stats<<" "<<in<<" "<<out<<" "<<param<<"\n";
			glDispatchCompute(x,y,z);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT);
		}
		void non_layer(GLuint &in, GLuint &out,
		uint x=1, uint y=1, uint z=1){
			glUseProgram(fun);
			glBindBufferBase(GL_SSB,0,in);
			glBindBufferBase(GL_SSB,1,out);
			glDispatchCompute(x,y,z);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT);
		}
		//~wrapper(){ glDeleteProgram(fun); std::cout<<"dead\n"; }
	};
	
	GLuint render, default_buffers[4];
	wrapper ff_f, ff_b, ff_u, sigm_f, sigm_b, tanh_f, tanh_b, relu_f, relu_b,
			cnn_f, cnn_b, cnn_u, cnn_s, mp_f, mp_b, unite, split;
	
	std::random_device rd;
	float randomu(float l,float h){
		std::uniform_real_distribution<float> dist(l,h);
		std::mt19937 tm(rd());
		return dist(tm);
	}//dude
	
	STR RFF(STR textfile){
		STR con="", line;
		std::ifstream file(textfile);
		if(!file) return "";
		while(!file.eof()){
			std::getline(file,line); con.append(line).append("\n");
		}file.close(); return con;
	}//read text from file
	
	wrapper CCP(STR shader){
		STR CS=RFF("./"+shader+".comp"); //reads files
		const char *cs=CS.c_str(); //translates them to char*
		GLuint c_shader=glCreateShader(GL_COMPUTE_SHADER); //create vertex shader
		glShaderSource(c_shader, 1, &cs, NULL); //set its source code
		glCompileShader(c_shader); //compile
		std::cout<<shader<<" complete.\n";
		int ss, size=1000, ll; char* out=new char[1000];
		glGetShaderiv(c_shader, GL_COMPILE_STATUS, &ss);
		if(ss!=GL_TRUE){
			glGetShaderInfoLog(c_shader, size, &ll, out);
			std::cout<<out<<"\n\n";
		}
		GLuint fun=glCreateProgram();
		glAttachShader(fun, c_shader);
		glLinkProgram(fun);
		if(ss!=GL_TRUE){
			glGetProgramInfoLog(fun, size, &ll, out);
			std::cout<<out<<"\n";//*/
		}
		
		glUseProgram(fun);
		for(unsigned char i=0; i<3; i++){
			glBindBufferBase(GL_SSB, i, default_buffers[i]);
		}glBindBufferBase(GL_UBO, 0, default_buffers[3]);
		
		return wrapper(fun);
	}
	
	wrapper CCF(STR shader){
		STR CS=RFF("./"+shader+".comp"); //reads files
		const char *cs=CS.c_str(); //translates them to char*
		GLuint c_shader=glCreateShader(GL_COMPUTE_SHADER); //create vertex shader
		glShaderSource(c_shader, 1, &cs, NULL); //set its source code
		glCompileShader(c_shader); //compile
		
		int ss, size=1000, ll; char* out=new char[1000];
		glGetShaderiv(c_shader, GL_COMPILE_STATUS, &ss);
		if(ss!=GL_TRUE){
			glGetShaderInfoLog(c_shader, size, &ll, out);
			std::cout<<out<<"\n\n";
		}
		GLuint fun=glCreateProgram();
		glAttachShader(fun, c_shader);
		glLinkProgram(fun);
		if(ss!=GL_TRUE){
			glGetProgramInfoLog(fun, size, &ll, out);
			std::cout<<out<<"\n";//*/
		}
		
		glUseProgram(fun);
		for(unsigned char i=0; i<2; i++){
			glBindBufferBase(GL_SSB, i, default_buffers[i]);
		}
		
		return wrapper(fun);
	}
	
	void* make_buffer(GLuint &buffer, uint size, GLenum type){
		glGenBuffers(1,&buffer);
		glBindBuffer(GL_SSB, buffer);
		glBufferStorage(GL_SSB, size, 0, PRW);
		return glMapBufferRange(GL_SSB,0,size,PRW);
	}
	
	void init(){
		make_buffer(default_buffers[0], 16, GL_SSB);
		make_buffer(default_buffers[1], 16, GL_SSB);
		make_buffer(default_buffers[2], 16, GL_SSB);
		make_buffer(default_buffers[3], 32, GL_UBO);
		ff_f=CCP("ff_forward");
		ff_b=CCP("ff_backward");
		ff_u=CCP("ff_update");
		sigm_f=CCF("sigmoid");
		sigm_b=CCF("dsigmoid");
		tanh_f=CCF("tanh");
		tanh_b=CCF("dtanh");
		relu_f=CCF("relu");
		relu_b=CCF("drelu");
		cnn_f=CCP("cnn_forward");
		cnn_u=CCP("cnn_update");
		cnn_s=CCP("cnn_bias_update");
		cnn_b=CCP("cnn_backward");
		mp_f=CCF("pool");
		mp_b=CCF("dpool");
		unite=CCF("unite");
		split=CCF("split");
		std::cout<<"yay\n";
	}
	
	void set_spd(float s){ spd=s; }
	
	
	
	
	layer::layer(STR t):file(t){}
	
	GLuint layer::operator>>(SV<float> t){
		//memcpy((char*)ptr_in, t.data(), t.size()*4);
		return out;
	}
	GLuint layer::operator<<(SV<float> t){
		//memcpy((char*)ptr_error, t.data(), t.size()*4);
		return out;
	}
	GLuint layer::operator>>(GLuint t){
		return t;
	}
	GLuint layer::operator<<(GLuint t){
		return t;
	}
	void layer::update(){}
	void layer::save(){}
	void* layer::get_out(){return ptr_out; }
	void* layer::get_in(){return ptr_in; }
	void* layer::get_err(){return ptr_error; }
	layer::~layer(){}
	
	
	
	forward::forward(STR t, uint x, uint y, bool):layer(t),X(x),Y(y){
	
	}
	forward::forward(STR t, uint x, uint y):layer(t), X(x), Y(y){
		std::ifstream r(t, std::ifstream::binary);
		if(r){
			//read file
			std::cout<<"found file\n";
		}else{
			SV<float> p; uint e=X*Y+Y;
			for(uint i=0; i<e; i++) p.push_back(randomu(-0.1,0.1));
			//ptr_in=make_buffer(in, X*4, GL_SSB);
			ptr_out=make_buffer(out, Y*4, GL_SSB);
			ptr_error=make_buffer(error, X*4, GL_SSB);
			ptr_params=make_buffer(params, e*4, GL_SSB);
			memcpy((char*)ptr_params, p.data(), e*4);
			ptr_stats=make_buffer(stats,32,GL_UBO);
			memcpy((char*)ptr_stats, &X, 4);
			memcpy((char*)ptr_stats+4, &Y, 4);
			memcpy((char*)ptr_stats+16, &spd, 4);
		}
	} //init from file
	forward::forward(STR t, SV<SV<float>> w, SV<float> b):layer(t), 
		X(w[0].size()), Y(w.size()){
		uint e=X*Y+Y, size=X*4;
		//ptr_in=make_buffer(in, X*4, GL_SSB);
		ptr_out=make_buffer(out, Y*4, GL_SSB);
		ptr_error=make_buffer(error, X*4, GL_SSB);
		ptr_params=make_buffer(params, e*4, GL_SSB);
		for(uint i=0; i<w.size(); i++) 
			memcpy((char*)ptr_params+i*size, w[i].data(), size);
		memcpy((char*)ptr_params+size*Y, b.data(), Y*4);
		ptr_stats=make_buffer(stats,32,GL_UBO);
		memcpy((char*)ptr_stats, &X, 4);
		memcpy((char*)ptr_stats+4, &Y, 4);
		memcpy((char*)ptr_stats+16, &spd, 4);
	}
	GLuint forward::operator>>(SV<float> IN){
		if(!ptr_in) ptr_in=make_buffer(in,X*4,GL_SSB);
		memcpy((char*)ptr_in,IN.data(),X*4);
		ff_f(stats,in,params,out,Y);
		return out;
	}
	GLuint forward::operator<<(SV<float> OUT){
		memcpy((char*)ptr_out,OUT.data(),Y*4);
		grad=out;
		ff_b(stats,out,params,error,X);
		return error;
	}
	GLuint forward::operator>>(GLuint input){ in=input;
		ff_f(stats,in,params,out,Y);
		return out;
	}
	GLuint forward::operator<<(GLuint output){ grad=output;
		ff_b(stats,output,params,error,X);
		return error;
	}
	void forward::update(){
		ff_u(stats,in,params,grad,Y);
	}
	void forward::save(){
		std::vector<float> bebra(X*Y+Y,90);
		memcpy(bebra.data(), (char*)ptr_params, (X*Y+Y)*4);
		for(auto i:bebra) std::cout<<i<<" "; std::cout<<"\n";
	} //save to file
	forward::~forward(){
		
	}
	
	void activate::compute_forward(GLuint &i){
		//std::cout<<th<<" "<<SIGMOID<<"\n";
		switch(th){
			case SIGMOID: sigm_f.non_layer(i,out, X); break;
			case TANH: tanh_f.non_layer(i,out, X); break;
			case RELU: relu_f.non_layer(i,out, X); break;
			default:break;
		}
	}
	void activate::compute_back(GLuint &r){
		switch(th){
			case SIGMOID: sigm_b.non_layer(r,out, X); break;
			case TANH: tanh_b.non_layer(r,out, X); break;
			case RELU: relu_b.non_layer(r,out, X); break;
			default:break;
		}
	}
	activate::activate(type t, uint x):layer(""),th(t),X(x){
		ptr_out=make_buffer(out, X*4, GL_SSB);
		ptr_error=make_buffer(error, X*4, GL_SSB);
	}
	GLuint activate::operator>>(std::vector<float> IN){
		if(!ptr_in) ptr_in=make_buffer(in,X*4,GL_SSB);
		memcpy((char*)ptr_in,IN.data(),X*4);
		compute_forward(in);
		return out;
	}
	GLuint activate::operator<<(std::vector<float> OUT){
		if(!ptr_in) ptr_in=make_buffer(in,X*4,GL_SSB);
		memcpy((char*)ptr_in,OUT.data(),X*4);
		compute_back(in);
		return out;
	}
	GLuint activate::operator>>(GLuint _){
		compute_forward(_);
		return out;
	}
	GLuint activate::operator<<(GLuint _){
		compute_back(_);
		return out;
	}
	activate::~activate(){
		
	}
	
	
	
	
	convolute::convolute(std::string s, uint x, uint y, uint ii, uint oo):forward(s,x,y,0),I(ii),O(oo){
		std::vector<float> p;
		for(uint i=0; i<oo*ii*y*x; i++) p.push_back(randomu(-0.001,0.001));
		for(uint i=0; i<oo; i++) p.push_back(randomu(-0.001,0.001));
		ptr_params=make_buffer(params, p.size()*4, GL_SSB);
		memcpy((char*)ptr_params, p.data(), p.size()*4);
		ptr_stats=make_buffer(stats, 64, GL_UBO);
		memcpy((char*)ptr_stats, &X, 4);
		memcpy((char*)ptr_stats+4, &Y, 4);
		memcpy((char*)ptr_stats+24, &I, 4);
		memcpy((char*)ptr_stats+40, &O, 4);
		memcpy((char*)ptr_stats+48, &spd, 4);
	}
	convolute::convolute(std::string s, SV<SV<SV<SV<float>>>> w, SV<float> b):forward(s,w[0][0][0].size(),w[0][0].size(),0), I(w[0].size()), O(w.size()){
		//ptr_out=make_buffer(out, X*Y*O*4, GL_SSB);
		//ptr_error=make_buffer(error, X*Y*I*4, GL_SSB);
		ptr_params=make_buffer(params, (X*Y*I*O+O)*4, GL_SSB);
		unsigned int pos=0;
		for(auto _0:w) for(auto _1:_0) for(auto _2:_1) {
			memcpy((char*)ptr_params+pos, _2.data(), _2.size()*4);
			pos+=_2.size()*4;
		}//std::cout<<pos<<" "<<(X*Y*I*O+O)*4<<" "<<b.size()*4<<"\n";
		memcpy((char*)ptr_params+pos, b.data(), b.size()*4);
		ptr_stats=make_buffer(stats, 64, GL_UBO);
		memcpy((char*)ptr_stats, &X, 4);
		memcpy((char*)ptr_stats+4, &Y, 4);
		memcpy((char*)ptr_stats+24, &I, 4);
		memcpy((char*)ptr_stats+40, &O, 4);
		memcpy((char*)ptr_stats+48, &spd, 4);
	}
	void convolute::set_image_params(uint i_x, uint i_y, uint stride, uint padding){
		img_in[0]=i_x; img_in[1]=i_y;
		//img_out[0]=o_x; img_out[1]=o_y;
		S=stride; P=padding;
		if(!P){ i_x-=(X/2)*2; i_y-=(Y/2)*2; }
		img_out[0]=std::round(1.0*i_x/S); img_out[1]=std::round(1.0*i_y/S);
		memcpy((char*)ptr_stats+8, &S, 4);
		memcpy((char*)ptr_stats+12, &P, 4);
		memcpy((char*)ptr_stats+16, img_in, 8);
		memcpy((char*)ptr_stats+32, img_out, 8);
		ptr_out=make_buffer(out, img_out[0]*img_out[1]*O*4, GL_SSB);
		std::cout<<img_out[0]<<" "<<img_out[1]<<" "<<I<<" "<<O<<"\n";
		ptr_error=make_buffer(error, img_in[0]*img_in[1]*I*4, GL_SSB);
	}
	GLuint convolute::operator>>(std::vector<float> IN){ //X Y and I define it
		if(!ptr_in) ptr_in=make_buffer(in,IN.size()*4,GL_SSB);
		memcpy((char*)ptr_in,IN.data(),IN.size()*4);
		cnn_f(stats,in,params,out,O,img_out[0],img_out[1]);
		return out;
	}
	GLuint convolute::operator<<(std::vector<float> OUT){ //X Y and I define it
		memcpy((char*)ptr_out,OUT.data(),OUT.size()*4);
		grad=out;
		cnn_b(stats,out,params,error,I,img_in[0],img_in[1]);
		return error;
	}
	GLuint convolute::operator>>(GLuint IN){ in=IN;
		cnn_f(stats,IN,params,out,O,img_out[0],img_out[1]);
		return out;
	}
	GLuint convolute::operator<<(GLuint OUT){ grad=OUT;
		cnn_b(stats,grad,params,error,I,img_in[0],img_in[1]);
		return error;
	}
	void convolute::update(){
		cnn_u(stats,in,params,grad,X*Y,I,O);
		cnn_s(stats,in,params,grad,O);
	}
	void convolute::save(){
		std::vector<float> bebra(X*Y*I*O+O,90);
		memcpy(bebra.data(), (char*)ptr_params, (X*Y*I*O+O)*4);
		for(uint i=0; i<O; i++){ std::cout<<"out filter:\n";
			for(uint j=0; j<I; j++){std::cout<<"in filter:\n";
				for(uint k=0; k<Y; k++){
					for(uint l=0; l<X; l++){
						std::cout<<bebra[i*I*Y*X+j*Y*X+k*X+l]<<" ";
					}std::cout<<"\n";
				}
			}
		}std::cout<<"bias:\n";
		for(unsigned int i=X*Y*I*O; i<bebra.size(); i++) std::cout<<bebra[i]<<" ";
		std::cout<<"\n";
	}
	convolute::~convolute(){
	
	}
	
	
	
	pooling::pooling(uint x, uint y, uint z):layer(""),X(x),Y(y),Z(z){ uint U=X*Y*Z*4;
		ptr_stats=make_buffer(stats, 32, GL_UBO);
		ptr_store=make_buffer(store, U, GL_SSB);
		ptr_out=make_buffer(out, U, GL_SSB);
		ptr_error=make_buffer(error, U*4, GL_SSB);
		memcpy((char*)ptr_stats, &X, 4);
		memcpy((char*)ptr_stats+4, &Y, 4);
		memcpy((char*)ptr_stats+8, &Z, 4);
	}
	GLuint pooling::operator>>(SV<float> IN){
		if(!ptr_in) ptr_in=make_buffer(in,IN.size()*4,GL_SSB);
		memcpy((char*)ptr_in,IN.data(),IN.size()*4);
		mp_f(stats,in,store,out,Z,Y,X);
		return out;
	}
	GLuint pooling::operator<<(SV<float> IN){
		memcpy((char*)ptr_out,IN.data(),IN.size()*4);
		mp_b(stats,out,store,error,Z,Y,X);
		return error;
	}
	GLuint pooling::operator>>(GLuint IN){
		mp_f(stats,IN,store,out,Z,Y,X);
		return out;
	}
	GLuint pooling::operator<<(GLuint IN){ 
		mp_b(stats,IN,store,error,Z,Y,X);
		return error;
	}
	pooling::~pooling(){
	
	}
	
	merger::merger(SV<uint> s, uint S):sizes(s){ s.clear(); s.push_back(0);
		for(uint i=0; i<sizes.size(); i++){ s.push_back(sizes[i]); }
		for(uint i=0; i<sizes.size(); i++){ s[i]=s[i]+((i)?s[i-1]:0);
			stats.push_back(0); err.push_back(0);
			ptr_stats.push_back(make_buffer(stats[i],16,GL_UBO));
			memcpy((char*)ptr_stats[i], &s[i], 4);
			ptr_err.push_back(make_buffer(err[i],sizes[i]*4,GL_SSB));
		}ptr_out=make_buffer(out, S*4, GL_SSB);
	}
	
	GLuint merger::operator>>(SV<SV<float>> IN){ GLuint temp=0;
		for(uint i=0; i<IN.size(); i++){
			memcpy((char*)ptr_err[i]+sizes[i]*4, IN[i].data(), IN[i].size()*4);
			unite(stats[i],err[i],temp,out,sizes[i]);
		}return out;
	}
	
	GLuint merger::operator>>(SV<GLuint> IN){ GLuint temp=0;
		for(uint i=0; i<IN.size(); i++){
			unite(stats[i],IN[i],temp,out,sizes[i]);
		}return out;
	}
	
	SV<GLuint> merger::operator<<(GLuint OUT){ GLuint temp=0;
		for(uint i=0; i<err.size(); i++){
			split(stats[i],OUT,temp,err[i],sizes[i]);
		}return err;
	}
	
	SV<GLuint> merger::operator<<(SV<float> OUT){ GLuint temp=0;
		memcpy((char*)ptr_out,OUT.data(),OUT.size()*4);
		for(uint i=0; i<err.size(); i++){
			split(stats[i],out,temp,err[i],sizes[i]);
		}return err;
	}
	
	SV<void*> merger::get_errors(){ return ptr_err; }
	void* merger::get_out(){ return ptr_out; }
	
	merger::~merger(){
	
	}
	
	
	
	
	buff::buff(uint t){
		ptr_out=make_buffer(out, t*4, GL_SSB);
	}
	
	GLuint buff::operator>>(SV<float> in){
		memcpy((char*)ptr_out,in.data(),in.size()*4);
		return out;
	}
	
	buff::~buff(){
		
	}
}
