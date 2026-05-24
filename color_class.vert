#version 430

out vec4 point;
flat out uint pos;

layout(std140, binding=0) uniform constanta{
	vec4 real_points[6];
	uvec2 size;
};

layout(std430, binding=1) buffer dynamic{
	float matrix[];
};

void main(void){
	uint id=gl_VertexID, lid=id%6, gid=id/6,
	y=gid/size.x, x=gid%size.x; pos=gid;
	gl_Position=vec4(2.0*real_points[lid].x/size.x+2.0*x/size.x-1,
					 -(2.0*real_points[lid].y/size.y+2.0*y/size.y-1),
					 real_points[lid].z,
					 1.0);
	point=vec4(real_points[lid].xy,matrix[gid],1);
}//нет переделать
