#include "detrov.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include "layers.hpp"
#include <cmath>
#include <random>
#include <cstring>

std::random_device rd;
float randomu(float l,float h){
	std::uniform_real_distribution<float> dist(l,h);
	std::mt19937 tm(rd()); return dist(tm);
}//функция случайного числа

std::vector<gail::layer*> init_neural_network(){
	gail::set_spd(10e-3); //установка скорости 0.001
	std::vector<gail::convolute*> cnns={
		new gail::convolute("",3,3,4,8), new gail::convolute("",3,3,8,12), new gail::convolute("",3,3,12,16)
	};//3 слоя сверточной нейронной сети
	std::vector<gail::layer*> network={
		cnns[0], new gail::activate(gail::activate::SIGMOID,16*16*8), new gail::pooling(16,16,8),
		cnns[1], new gail::activate(gail::activate::SIGMOID,8*8*12), new gail::pooling(8,8,12), 
		cnns[2], new gail::activate(gail::activate::SIGMOID,2*2*16), new gail::pooling(2,2,16), 
		new gail::forward("",64,16), new gail::activate(gail::activate::SIGMOID,16),
		new gail::forward("",16,4) //полная архитектура агента
	};cnns[0]->set_image_params(16,16,1,1);//установка параметров изображения
	cnns[1]->set_image_params(8,8,1,1);
	cnns[2]->set_image_params(4,4,1,0);
	return network;
}//функция создания агента для RL

std::vector<gail::layer*> init_receptor(){
	gail::set_spd(10e-6); std::vector<gail::convolute*> cnns={
		new gail::convolute("",3,3,4,8), new gail::convolute("",3,3,8,12), new gail::convolute("",3,3,12,16)
	};cnns[0]->set_image_params(16,16,1,1);
	cnns[1]->set_image_params(8,8,1,1);
	cnns[2]->set_image_params(4,4,1,0);
	std::vector<gail::layer*> network={
		cnns[0], new gail::activate(gail::activate::SIGMOID,16*16*8), new gail::pooling(16,16,8),
		cnns[1], new gail::activate(gail::activate::SIGMOID,8*8*12), new gail::pooling(8,8,12), 
		cnns[2], new gail::activate(gail::activate::SIGMOID,2*2*16), new gail::pooling(2,2,16)
	};return network;
}//функция создания рецептора

std::vector<gail::layer*> init_generator(){
	gail::set_spd(10e-5); std::vector<gail::layer*> network={
		new gail::forward("",68,100), new gail::activate(gail::activate::SIGMOID,100),
		new gail::forward("",100,4), new gail::activate(gail::activate::SIGMOID,4)
	};return network;
}//функция создания генератора для GAIL

std::vector<gail::layer*> init_discriminator(){
	gail::set_spd(10e-4);
	std::vector<gail::layer*> network={
		new gail::forward("",68,150), new gail::activate(gail::activate::SIGMOID, 150),
		new gail::forward("",150,75), new gail::activate(gail::activate::SIGMOID, 75),
		new gail::forward("",75,1), new gail::activate(gail::activate::SIGMOID, 1)
	};return network;
}//функция создания дискриминатора для GAIL

bool player_mode=0, read_records=0; //флаги состояний
unsigned int count_loss=0; //подсчет поражений

std::vector<float> softmax(std::vector<float> t){
	for(auto &i:t) i=std::exp(i); //экспонента
	float sum=0.0; for(auto i:t) sum+=i; //сумма всех значений
	for(auto &i:t) i/=sum; //среднее
	return t;
}//функция softmax

std::vector<float> run(std::vector<gail::layer*> lol, unsigned int out_size){
	std::vector<float> t; for(auto i:maps) for(auto j:i) t.push_back(j);
	GLuint T=(*lol[0])>>t; for(unsigned int i=1; i<lol.size(); i++) T=(*lol[i])>>T;
	glFinish(); std::vector<float> o(out_size,0);
	memcpy((char*)o.data(),(char*)lol[lol.size()-1]->get_out(),out_size*4);
	return o;
}//шаг нейронной сети 

GLuint run(std::vector<gail::layer*> lol){ std::vector<float> t;
	for(auto i:maps) for(auto j:i) t.push_back(j); GLuint T=(*lol[0])>>t;
	for(unsigned int i=1; i<lol.size(); i++) T=(*lol[i])>>T; return T;
}//шаг генератора
GLuint run(GLuint _, std::vector<gail::layer*> lol){
	for(unsigned int i=0; i<lol.size(); i++) _=(*lol[i])>>_; return _;
}//шаг 

std::vector<float> run(GLuint _, std::vector<gail::layer*> lol, unsigned int out_size){
	for(unsigned int i=0; i<lol.size(); i++) _=(*lol[i])>>_;
	glFinish(); std::vector<float> o(out_size,0);
	memcpy((char*)o.data(),(char*)lol[lol.size()-1]->get_out(),out_size*4);
	return o;
}//шаг

std::vector<float> run(std::vector<gail::layer*> lol,std::vector<gail::layer*> lmao, unsigned int out_size){
	return run(run(lol),lmao,out_size);
}//шаг


GLuint unrun(std::vector<float> t, std::vector<gail::layer*> lol){
	GLuint T=(*lol[lol.size()-1])<<t;
	for(int i=0, j=lol.size()-2; i<lol.size()-1; i++,j--) T=(*lol[i])<<T;
	return T;
}

GLuint unrun(GLuint T, std::vector<gail::layer*> lol){
	for(int i=0, j=lol.size()-1; i<lol.size(); i++,j--) T=(*lol[i])<<T; return T;
}

void train(std::vector<gail::layer*> lol){ for(auto i:lol) i->update(); }

float discriminate(std::vector<gail::layer*> in, std::vector<gail::layer*> dis, gail::merger &M, gail::buff &B, std::vector<float> correct){
	std::vector<GLuint> T{run(in), B>>correct}, U(2,0);
	std::vector<float> out=run(M>>T,dis,1); //needs to combine with correct answer
	glFinish(); float OUT=out[0]; out[0]=1.0-out[0];
	GLuint UT=unrun(out, dis); U=M<<UT; unrun(U[0],in);
	train(in); train(dis); return OUT;
}//обучение дискриминатора на основе актора

float unscriminate(std::vector<gail::layer*> in, std::vector<gail::layer*> gen, std::vector<gail::layer*> dis, gail::merger &M, gail::merger &H, gail::buff &B){
	std::vector<float> N{randomu(-1,1),randomu(-1,1),randomu(-1,1),randomu(-1,1)};
	GLuint S=run(in); std::vector<GLuint> IN{S,B>>N};
	GLuint SL=run(H>>IN,gen); std::vector<GLuint> T{S,SL}, U(2,0);
	std::vector<float> out=run(M>>T,dis,1); glFinish();
	float OUT=out[0]; out[0]=-out[0];
	GLuint UT=unrun(out,dis); U=M<<UT; unrun(U[0],in);
	train(dis); train(in); return OUT;
}//обучение дискриминатора на основе генератора

float degenerate(std::vector<gail::layer*> in, std::vector<gail::layer*> gen, std::vector<gail::layer*> dis, gail::merger &M, gail::merger &H, gail::buff &B){
	std::vector<float> N{randomu(-1,-1),randomu(-1,1),randomu(-1,1),randomu(-1,1)};
	GLuint S=run(in); std::vector<GLuint> IN{S,B>>N};
	GLuint SL=run(H>>IN,gen); std::vector<GLuint> T{S,SL}, U(2,0);
	std::vector<float> out=run(M>>T,dis,1); glFinish();
	float OUT=out[0]; out[0]=1.0-out[0]; GLuint UT=unrun(out,dis); U=M<<UT;
	UT=unrun(U[1],gen); train(gen); return OUT;
}//обучение генератора

unsigned int train_count=0; //подсчет эпох

void train(std::vector<gail::layer*> lol, std::vector<float> t){
	GLuint T=(*lol[lol.size()-1])<<t;
	for(int i=lol.size()-2; i>=0; i--){ T=(*lol[i])<<T; }
	for(auto i:lol) i->update(); train_count++;
}//

float AAAAAA=0.0;
std::vector<float> operator-(std::vector<float> a, std::vector<float> b){
	for(unsigned int i=0; i<a.size(); i++) a[i]-=b[i]; return a;
}//вычисление градиента

bool max(std::vector<float> a, std::vector<float> b){ unsigned int maximum1=0.0, maximum2=0.0; float T=0.0;
	for(unsigned int i=0; i<a.size(); i++) if(T<a[i]){T=a[i];maximum1=i;} T=0.0;
	for(unsigned int i=0; i<b.size(); i++) if(T<b[i]){T=b[i];maximum2=i;}
	return maximum1==maximum2;
}//лучший результат

void train(std::vector<gail::layer*> lol, std::vector<float> V, float t_v){
	AAAAAA+=t_v; if(t_v) {
		std::vector<float> F=V; float T=-1.0/0.0; unsigned int m=0;
		for(unsigned int i=0; i<F.size(); i++) if(T<F[i]) { T=F[i]; m=i; }
		F=std::vector<float>(V.size(),0.0);
		if(t_v<0) F[m]=(-V[m])*std::abs(t_v);
		else F[m]=(1.0-V[m])*std::abs(t_v);
		train(lol, F);
	}
}

bool game_over(std::vector<gail::layer*> lol, std::vector<float> t){
	if(steps>20) train(lol,t,-1.0);
	return collected>needed||steps>20;
}//условие окончания

bool generative=0, reinforcement=0, cloning=0;

int main(){
	char T; std::vector<gail::layer*> network;
	std::cout<<"Would you like to play? [Y/R/N]: "; std::cin>>T;
	switch(T){
		case 'y': case 'Y': player_mode=1; break;
		case 'r': case 'R': read_records=1; break;
		default: player_mode=0; break;
	}GLFWwindow* w; int current=0;
	if(player_mode){init_program(w,16,16);
		begin_record("record2.rec");
		read_map("./PETROV/lol.map");
		unsigned int sstp=0;
		while(!glfwWindowShouldClose(w)){ moveX=0; moveY=0; output(w,16,16);
			if(STEPS!=sstp) {record(std::vector<float>{(float)(moveX<0),(float)(moveX>0),(float)(moveY<0),(float)(moveY>0)}); sstp=STEPS; }
			if(game_over()) break;
		}KILLG(w);
	}else if(read_records){init_program(w,16,16,1);
		begin_read("record2.rec"); read_map("./PETROV/lol.map");
		while(read_record()){ moveX=0; moveY=0;
			if(glfwWindowShouldClose(w)) break;
			if(choice[0]) moveX=-1; else if(choice[1]) moveX=1;
			if(choice[2]) moveY=-1; else if(choice[3]) moveY=1;
			petrov_move(moveX,moveY); output(w, 16,16);
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}KILLG(w);
	}else{
		std::cout<<"Choose type of lerning? [B/R/G]: "; std::cin>>T;
		switch(T){
			case 'b': case 'B': cloning=1; break;
			case 'r': case 'R': reinforcement=1; break;
			case 'g': case 'G': generative=1; break;
			default: break;
		}if(reinforcement){
			init_program(w,16,16,1); gail::init();
			unsigned int count=0; network=init_neural_network();
			std::cout<<"\nL0\n";network[0]->save();
			std::cout<<"\nL1\n";network[3]->save();
			std::cout<<"\nL2\n";network[6]->save();
			uint mm[4]={0,0,0,0}, session=0;
			restart: read_map("./PETROV/map2.map"); for(uint i=0; i<4; i++) mm[i]=0; session=0;
			if(count>100000) goto OUTTA_HERE;
			while(!glfwWindowShouldClose(w)){
				moveX=0; moveY=0; std::vector<float> solution=softmax(run(network,4));
				unsigned int maximum; float T=-1.0/0.0;
				for(unsigned int i=0; i<solution.size(); i++) if(T<solution[i]){T=solution[i];maximum=i;} session++;
				switch(maximum){
					case 0: moveX=-1; mm[0]++; break;
					case 1: moveX=1; mm[1]++; break;
					case 2: moveY=-1; mm[2]++; break;
					case 3: moveY=1; mm[3]++; break;
					default: break;
				} float train_value;
				if((train_value=petrov_move(moveX,moveY))<-0.2) count_loss++;
				train(network, solution, train_value); output(w,16,16);
				if(game_over(network,solution)||count_loss>0) { overall+=collected; if(collected>maxochko) maxochko=collected; collected=0; needed=0; money=0; count_loss=0; steps=1; count++; system("clear"); std::cout<<"epoch: "<<count<<"\n";
				std::cout<<"average decision: "<<1.0*mm[0]/session<<" "<<1.0*mm[1]/session<<" "<<1.0*mm[2]/session<<" "<<1.0*mm[3]/session<<" "<<"\n"; goto restart; }
			}OUTTA_HERE:KILLG(w); //delete
			std::cout<<"\nL0\n";network[0]->save();
			std::cout<<"\nL1\n";network[3]->save();
			std::cout<<"\nL2\n";network[6]->save();
		}if(cloning){ std::cout<<"BC\n";
			init_program(w,16,16,1); gail::init();
			network=init_neural_network();
			for(unsigned int _=0; _<100000; _++){
				begin_read("record.rec"); if(glfwWindowShouldClose(w)) goto OUT_OF_CLON;
				read_map("./PETROV/lol.map");
				while(read_record()){ moveX=0; moveY=0;
					if(glfwWindowShouldClose(w)) goto OUT_OF_CLON;
					unsigned int maximum; float T=-1.0/0.0;
					std::vector<float> solution=softmax(run(network,4)), SOL=solution;
					for(auto i:solution) std::cout<<i<<" "; std::cout<<"\n";
					solution=choice-solution; output(w,16,16);
					if(!max(SOL,choice)){
						train(network, solution);
						overall+=collected; if(collected>maxochko) maxochko=collected; collected=0; needed=0; money=0; count_loss=0; steps=1;
						break; }
					for(unsigned int i=0; i<SOL.size(); i++) if(T<SOL[i]){T=SOL[i];maximum=i;}
					moveX=0; moveY=0;
					switch(maximum){
						case 0: moveX=-1; break; case 1: moveX=1; break;
						case 2: moveY=-1; break; case 3: moveY=1; break;
						default: break;
					}petrov_move(moveX,moveY);
					std::cout<<moveX<<" "<<moveY<<"\n";
					if(game_over(network,solution)||count_loss>0) { overall+=collected; if(collected>maxochko) maxochko=collected; collected=0; money=0; count_loss=0; steps=1; break; }
				}
			}OUT_OF_CLON:;/*read_map("./PETROV/lol.map");
			steps=0;
			while(!glfwWindowShouldClose(w)){
				moveX=0; moveY=0;
				std::vector<float> solution=softmax(run(network,4));
				unsigned int maximum; float T=-1.0/0.0;
				for(unsigned int i=0; i<solution.size(); i++) if(T<solution[i]){T=solution[i];maximum=i;}
				switch(maximum){
					case 0: moveX=-1; break;
					case 1: moveX=1; break;
					case 2: moveY=-1; break;
					case 3: moveY=1; break;
					default: break;
				} float train_value;
				petrov_move(moveX,moveY);
				output(w,16,16);
				steps++;
				//std::this_thread::sleep_for(std::chrono::milliseconds(500));
				//glfwSwapBuffers(window); glfwPollEvents();
				if(game_over()||count_loss>0) { overall+=collected; if(collected>maxochko) maxochko=collected; collected=0; needed=0; money=0; count_loss=0; steps=1; break; }
			}*/
		}if(generative){
			init_program(w,16,16,1); gail::init();
			std::vector<gail::layer*> receptor=init_receptor(), imitator=init_generator(), discr=init_discriminator();
			gail::merger Mg(std::vector<uint>{64,4},68), Hg(std::vector<uint>{64,4},68);
			gail::buff B(4), N(4);
			for(unsigned int epoch=1; epoch<100000; epoch++){ train_count++;
				if(glfwWindowShouldClose(w)) break;
				if(epoch%3){ begin_read("record"+std::to_string(epoch%3)+".rec"); steps=1;
					read_map("./PETROV/lol.map");
					while(read_record()){
						if(glfwWindowShouldClose(w)) break;
						unsigned int maximum; float T=-1.0/0.0;
						glFinish();
						discriminate(receptor,discr,Mg,B,choice);
						unscriminate(receptor,imitator,discr, Mg, Hg, N);
						for(unsigned int i=0; i<choice.size(); i++) if(T<choice[i]){T=choice[i];maximum=i;}
						moveX=0; moveY=0;
						switch(maximum){
							case 0: moveX=-1; break;case 1: moveX=1; break;
							case 2: moveY=-1; break;case 3: moveY=1; break;
							default: break;
						}petrov_move(moveX,moveY);
						if(game_over()||count_loss>0) { overall+=collected; if(collected>maxochko) maxochko=collected; collected=0; money=0; count_loss=0; steps=1; break; }
					}collected=0;
				}else{ steps=1;
					read_map("./PETROV/lol.map");
					while(1){
						unsigned int maximum; float T=-1.0/0.0; std::vector<float> ran{randomu(-1,1),randomu(-1,1),randomu(-1,1),randomu(-1,1)};
						GLuint COND=run(receptor), IN=N>>ran, MGD=Hg>>(std::vector<GLuint>{COND,IN});
						std::vector<float> sos, solution=softmax(sos=run(MGD,imitator,4));
						glFinish(); system("clear");
						std::cout<<"epoch: "<<train_count<<"\n";
						std::cout<<"quality: "<<degenerate(receptor,imitator,discr, Mg, Hg, N)<<"\n";
						for(unsigned int i=0; i<solution.size(); i++) if(T<solution[i]){T=solution[i];maximum=i;}
						moveX=0; moveY=0;
						switch(maximum){
							case 0: moveX=-1; break;
							case 1: moveX=1; break;
							case 2: moveY=-1; break;
							case 3: moveY=1; break;
							default: break;
						}if(petrov_move(moveX,moveY)<-0.01) count_loss++; output(w,16,16);
						for(auto i:sos) std::cout<<i<<" "; std::cout<<"\n";
						if(game_over()||count_loss>0) { overall+=collected; if(collected>maxochko) maxochko=collected; collected=0; money=0; count_loss=0; steps=1; break; }
					}
				}
			}std::cout<<"\nL0\n"; receptor[0]->save();
			std::cout<<"\nL1\n"; receptor[3]->save();
			std::cout<<"\nL2\n"; receptor[6]->save();
		}
	}std::cout<<"score: "<<money<<" | "<<"collected: "<<collected<<"/"<<needed<<" | train count: "<<train_count<<" | best score: "<<maxochko<<" | overall: "<<overall<<"\n"; //вывод о сессии
	glfwTerminate(); 
}
