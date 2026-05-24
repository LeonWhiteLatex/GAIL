#version 430

layout(std140, binding=0) uniform constanta{
	vec4 real_points[6];
	uvec2 size;
};

layout(std430, binding=1) buffer dynamic{
	float matrix[];
};

layout(binding=0) uniform sampler2DArray texs;

in vec4 point;
out vec4 color;
flat in uint pos;

void main(void){
	//vec3 pp=vec3();
	color=texture(texs,point.xyz);
}//нет переделать
