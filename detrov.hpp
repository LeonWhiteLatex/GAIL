#pragma once
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

extern std::vector<std::vector<int>> maps; //игровая карта
//std::map<int, std::vector<std::vector<int>>> doors, keys; //support
//std::vector<std::vector<int>> just_doors; //support
//std::vector<std::vector<std::vector<std::vector<int>>>> pairs; //support
//std::vector<int> pos; //support
extern std::vector<float> choice; //выбор нейронной сети
//std::string record_path=""; //support

void init_program(GLFWwindow*&, unsigned int, unsigned int, bool control=0);
void begin_record(std::string);
void record(std::vector<float>);
void begin_read(std::string);
bool read_record();

extern float money;
extern unsigned int steps, STEPS, collected, needed, overall, maxochko;
extern int moveX, moveY;
//void read_line(std::string, int);//support
void read_map(std::string);
//void unlock(unsigned int, unsigned int);//support
float petrov_move(int,int);
void output(GLFWwindow*, unsigned int, unsigned int);
bool game_over();
void KILLG(GLFWwindow*&);
