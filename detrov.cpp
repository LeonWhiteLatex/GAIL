#include "detrov.hpp"
#include <map>
#include <SOIL/SOIL.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <iomanip>
#define GL_SSB GL_SHADER_STORAGE_BUFFER
#define GL_UBO GL_UNIFORM_BUFFER

template <typename _> using  SV=std::vector<_>;
template <typename A, typename B> using SM=std::map<A,B>;

SV<SV<int>> maps={};
SM<int, SV<SV<int>>> doors, keys;
SV<SV<int>> just_doors;
SV<SV<SV<SV<int>>>> pairs;
SV<int> pos;
SV<float> choice={};
std::string record_path="";

GLuint RP, vao[1], texture_array, ubo[2];
void *const_ptr=0, *dynamic_ptr=0;
GLbitfield PRW=GL_MAP_PERSISTENT_BIT|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT;
uint boot=0;

std::string RFF(std::string textfile){
	std::string con="", line;
	std::ifstream file(textfile);
	if(!file) return "";
	while(!file.eof()){
		std::getline(file,line); con.append(line).append("\n");
	}file.close(); return con;
}//read text from file

void KILLG(GLFWwindow*& _){
	glfwDestroyWindow(_);
	glUnmapNamedBuffer(ubo[0]);
	glUnmapNamedBuffer(ubo[1]);
	glDeleteTextures(32, &texture_array);
}//destoy program

void begin_record(std::string t){ record_path=t;
	std::ofstream writer(record_path);
	writer.close();
}//создание файла поведения

void record(std::vector<float> t){
	std::ofstream writer(record_path, std::ios::app|std::ios::binary);
	std::cout<<moveX<<" "<<moveY<<"\n";
	for(auto i:t) std::cout<<i<<" "; std::cout<<"\n";
	writer.write(reinterpret_cast<char*>(t.data()),t.size()*4);
	writer.close();
}//запись действия
std::ifstream reader;

void begin_read(std::string t){ //std::cout<<"begin\n";
	reader.close();
	reader=std::ifstream(t, std::ios::binary);
}//открытие файла поведения

bool read_record(){//std::cout<<"start\n";
	choice=std::vector<float>(4,0);
	if(!reader) { reader.close(); return (bool)reader; }
	reader.read(reinterpret_cast<char*>(choice.data()),choice.size()*4);
	//for(auto i:choice) std::cout<<i<<" "; std::cout<<"\n";
	return (bool)reader;
}//воспроизведение действий

float money=0.0;
unsigned int steps=1, STEPS=0, collected=0, needed=0, overall=0, maxochko=0;
int moveX, moveY;
//измерительные параметры

void read_line(std::string line, int c){ int C=c; c*=16; int P;
	std::vector<int> nums; std::string num=""; //инициализация массива
	for(unsigned int i=0; i<line.size(); i++){
		switch(line[i]){
			case ' ': nums.push_back(std::stoi(num)); num=""; break; //извлечение данных
			default: num+=line[i]; break;
		}
	}nums.push_back(std::stoi(num));//последний символ
	for(int i=0; i<nums.size(); i++)
		switch(nums[i]){
			case 0: for(auto _map:maps) _map[c+i]=0; break;
			case 1: maps[0][c+i]=1; break;
			case -1: maps[1][c+i]=1; pos=std::vector<int>{i,C}; break;
			case 2: maps[2][c+i]=1; needed++; break;
			default: 
				if(nums[i]>0){
					keys[nums[i]].push_back(std::vector<int>{i,C});
					maps[3][c+i]=1;
				}else{
					doors[-nums[i]].push_back(std::vector<int>{i,C});
					maps[0][c+i]=1;
				}
			break;
		}//условие записи по слоям
}//функция перевода строки в логическое представление для игры

void read_map(std::string f){needed=0;
	maps.clear(); pairs.clear(); doors.clear(); keys.clear(); just_doors.clear();
	maps=std::vector<std::vector<int>>(4, std::vector<int>(16*16,0));
	just_doors=std::vector<std::vector<int>>(16,std::vector<int>(16,0));
	pairs=std::vector<std::vector<std::vector<std::vector<int>>>>(16,std::vector<std::vector<std::vector<int>>>(16));
	std::ifstream file(f);
	if(!file) return; int count=0;
	for(std::string line; getline(file,line); read_line(line, count++)); //чтение строк
	for(auto [i,j]:keys){ //внесение информации о ключах
		for(auto k:j){
			for(auto I:doors[i]) { pairs[k[0]][k[1]].push_back(I); just_doors[I[1]][I[0]]=1; }
		}
	}//std::cout<<needed<<"\n";
}//чтение карты для генерации игры

void unlock(unsigned int x, unsigned int y){
	maps[3][y*16+x]=0; //удаление ключа
	for(auto i:pairs[x][y]){
		maps[0][i[1]*16+i[0]]=0; //удаление препятствий
	}
}//открытие двери

float petrov_move(int x, int y){ moveX=x; moveY=y; //std::cout<<"now\n";
	std::vector<int> predict{pos[0]+x, pos[1]+y};
	//std::cout<<"borders\n";
	if(predict[0]<0||predict[1]<0||predict[0]>15||predict[1]>15) return -1.0;
	//std::cout<<"next: "<<maps[0][predict[1]*16+predict[0]]<<"\n";
	if(maps[0][predict[1]*16+predict[0]]) return -1.0;
	//std::cout<<"move\n";
	maps[1][pos[1]*16+pos[0]]=0;
	pos=predict;
	int position=pos[1]*16+pos[0];
	float rew=0.0;//-0.1/(maxochko+1);//-1.0*steps/40.0;
	maps[1][position]=1;
	if(maps[2][position]){ maps[2][position]=0; money+=1.0/steps; steps=1; collected++;
		rew=1.0; }
	if(maps[3][position]) unlock(pos[0],pos[1]);
	steps++; STEPS++;
	return rew;
}

float *materix=nullptr;

//corner left down - 0x

void output(GLFWwindow* w, uint X, uint Y){
	glClearColor(1.0,1.0,1.0,1.0); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glUseProgram(RP);
	glBindBufferBase(GL_UBO, 0, ubo[0]);
	glBindBufferBase(GL_SSB, 1, ubo[1]);
	//std::vector<std::vector<std::string>> display(16,std::vector<std::string>(16,"  "));
	for(unsigned int i=0, t=0; i<Y; i++){
		for(unsigned int j=0; j<X; j++, t++){
			materix[t]=0;
			if(maps[1][t]) materix[t]=1;//player
			if(maps[2][t]) materix[t]=2;//doller
			if(maps[3][t]) materix[t]=3;//key
		}
	}//наложение объектов
	
	for(int i=0, t=0; i<Y; i++){
		for(int j=0; j<X; j++,t++){
			if(maps[0][t]){
				materix[t]=4;//door
				if(just_doors[i][j]) continue;
				if(i+1<16&&maps[0][t+1]||i-1>=0&&maps[0][t-1]) materix[t]=5;//wall horl
				if(j+1<16&&maps[0][t+16]||j-1>=0&&maps[0][t-16]){
					if(materix[t]==5) materix[t]=7;//later add more
					else materix[t]=6;//wall vert
				}
			}
		}
	}//генерация стен
	for(int i=1; i<Y-1; i++){
		for(int j=1; j<X-1; j++){
			bool up=1, down=1, left=1, right=1;
			if(materix[i*X+j]==7){
				if(materix[i*X+j-1]>3) left=0;
				if(materix[i*X+j+1]>3) right=0;
				if(materix[(i-1)*X+j]>3) up=0;
				if(materix[(i+1)*X+j]>3) down=0;
				
				materix[i*X+j]+=left*8+right*4+up*2+down;
			}//std::cout<<std::setw(4)<<materix[i*X+j];
		}//std::cout<<"\n";
	}
	//шлифовка стен
	memcpy((char*)dynamic_ptr,materix, X*Y*4);
	glDrawArrays(GL_TRIANGLES, 0, boot);
	glfwSwapBuffers(w); glfwPollEvents();
}//ЗАМЕНИТЬ НА OPENGL

bool game_over(){
	//std::cout<<collected<<"/"<<needed<<"\n";
	return collected==needed||steps>20;
}//условие окончания игры

void get_key(GLFWwindow* w, int key, int sc, int act, int mods){
	if(act==GLFW_RELEASE) switch(key){
		case GLFW_KEY_W: petrov_move(0,-1); break;
		case GLFW_KEY_A: petrov_move(-1,0); break;
		case GLFW_KEY_S: petrov_move(0,1); break;
		case GLFW_KEY_D: petrov_move(1,0); break;
		default: break;
	}
}

void init_program(GLFWwindow*& window, uint X, uint Y, bool control){
	if(materix) delete[] materix; materix=new float[X*Y];
	if(!glfwInit()) exit(1);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); //no idea but dont touch
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //same here
	window=glfwCreateWindow(768, 768, "DETROV SIMULATOR", NULL, NULL); //creates window
	glfwMakeContextCurrent(window); if(glewInit()!=GLEW_OK) exit(2);
	glfwSwapInterval(0);
	
	std::string VS=RFF("./color_class.vert"), FS=RFF("./color_class.frag");
	const char *vs=VS.c_str(), *fs=FS.c_str();
	GLuint v_s=glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(v_s, 1, &vs, NULL); glCompileShader(v_s);
	GLuint f_s=glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(f_s, 1, &fs, NULL); glCompileShader(f_s);
	RP=glCreateProgram(); //create shader program
	glAttachShader(RP, v_s); //attach shaders
	glAttachShader(RP, f_s);
	glLinkProgram(RP); //link everything
	
	int ss; bool killer=true;
	glGetShaderiv(v_s, GL_COMPILE_STATUS, &ss);
	std::cout<<((killer&=(ss==GL_TRUE))? 
		"vertex shader ready.":
		"vertex shader did not compile.")<<"\n";
	
	glGetShaderiv(f_s, GL_COMPILE_STATUS, &ss);
	std::cout<<((killer&=(ss==GL_TRUE))? 
		"fragment shader ready.":
		"fragment shader did not compile.")<<"\n";
	
	glGetProgramiv(RP, GL_LINK_STATUS, &ss);
	std::cout<<((killer&=(ss==GL_TRUE))? 
		"linking complete.":
		"linking failed.")<<"\n";
		
	if(!killer) {
		std::cout<<"FATAL ERROR: graphics failed, aborting.\n";
		int size=1000, ll; char* out=new char[1000];
		glGetShaderInfoLog(v_s, size, &ll, out);
		std::cout<<out<<"\n\n";
		glGetShaderInfoLog(f_s, size, &ll, out);
		std::cout<<out<<"\n\n";
		glGetProgramInfoLog(RP, size, &ll, out);
		std::cout<<out<<"\n";//*/
		delete [] out;
		exit(1);
	}
	
	glGenVertexArrays(1,vao); glBindVertexArray(vao[0]);
	
	glGenBuffers(2,ubo); 
	glBindBuffer(GL_UBO, ubo[0]);
	glBindBufferBase(GL_UBO, 0, ubo[0]);
	glBufferStorage(GL_UBO, 112, 0, PRW);
	const_ptr=glMapBufferRange(GL_UBO,0,112,PRW);
	float pos[24]={
	1,-1,0,1,
	-1,1,0,1,
	1,1,0,1,
	1,-1,0,1,
	-1,1,0,1,
	-1,-1,0,1
	};memcpy((char*)const_ptr, pos, 96);
	memcpy((char*)const_ptr+96, &X, 4);
	memcpy((char*)const_ptr+100, &Y, 4);
	
	glBindBuffer(GL_SSB, ubo[1]);
	glBindBufferBase(GL_SSB, 1, ubo[1]);
	boot=X*Y*6; glBufferStorage(GL_SSB, X*Y*4, 0, PRW);
	dynamic_ptr=glMapBufferRange(GL_SSB,0,X*Y*4,PRW);
	
	glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthFunc(GL_LESS); glDepthMask(GL_TRUE);
	
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &texture_array);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texture_array);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //texture array
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //settings
	glTexStorage3D(GL_TEXTURE_2D_ARRAY,1, GL_RGBA8, 50,50,32);
	int wid, hgt;
	for(unsigned int i=0; i<32; i++){std::string p="./STUFF/image_"+std::to_string(i)+".png";
		unsigned char* data=SOIL_load_image(p.c_str(),&wid,&hgt,0,SOIL_LOAD_RGBA);
		glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,i,wid,hgt,1,GL_RGBA,GL_UNSIGNED_BYTE,data);
		SOIL_free_image_data(data);
	}//waits for its hour*/
	
	if(!control) glfwSetKeyCallback(window, get_key);
}

/*int main(){ GLFWwindow* w;
	init_program(w, 16,16);
	//std::cout<<boot<<"\n";
	read_map("./PETROV/lol.map");
	while(!glfwWindowShouldClose(w)){
		glClearColor(1.0,1.0,1.0,1.0); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		glUseProgram(RP);
		output(16,16);
		//glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D_ARRAY, texture_array);
		glDrawArrays(GL_TRIANGLES, 0, boot);
		glfwSwapBuffers(w); glfwPollEvents();
	}//for(unsigned int i=0; i<16; i++){
		//for(unsigned int j=0, t=0; j<16; j++, t++) std::cout<<materix[t]<<" ";
		//std::cout<<"\n";
	//}
	KILLG(w);
}*/
