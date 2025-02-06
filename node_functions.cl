float maprange(float val, float oldmin, float oldmax, float newmin, float newmax){
    float fac = (val - oldmin) / (oldmax - oldmin);
    return newmin + fac*(newmax - newmin);
}

float lerp(float a, float b, float fac){
    return a*(1.0f-fac) + b*fac;
}

float4 lerp4(float4 a, float4 b, float fac){
    return a*(1.0f-fac) + b*fac;
}

float4 simple_sample(__global const float* tex, float uvx, float uvy, int resx, int resy){
    int pixx = uvx * resx;
    int pixy = (int)(uvy * resy) * resx;

    int off = (pixx + pixy) * 4;

    return (float4)(tex[off+0], tex[off+1], tex[off+2], tex[off+3]);
}

// fmod that properly repeats <0
float repeatmod(float val, float div){
    if(val < 0.0f){
        return div+fmod(val, div);
    }
    return fmod(val, div);
}

// mod that repeats like /\/\/\/\/\/
float mirror(float val, float step){
    return fabs(repeatmod(val+step, step*2.0f)-step);
}

float4 sample(__global const float* tex, float uvx, float uvy, int resx, int resy, 
                int interp, int extend){

    float uvxmax = (float)(resx-1)/(float)(resx);
    float uvymax = (float)(resy-1)/(float)(resy);
    switch(extend){
        case 0: // clip
            if(uvx < 0.0f || uvx >= 1.0f || uvy < 0.0f || uvy >= 1.0f){
                return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            }
            break;
        case 1: // repeat
            uvx = repeatmod(uvx, 1.0f);
            uvy = repeatmod(uvy, 1.0f);
            break;
        case 2: // extend
            uvx = clamp(uvx, 0.0f, uvxmax);
            uvy = clamp(uvy, 0.0f, uvymax);
            break;
        case 3: // mirror
            uvx = mirror(uvx, 1.0f);
            uvy = mirror(uvy, 1.0f);
    }

    if(interp == 0){ // closest
        return simple_sample(tex, round(uvx*resx)/resx, round(uvy*resy)/resy, resx, resy);
    }
    if(interp == 1){ // linear
        float uvx1 = clamp(uvx+1.0f/(float)resx, 0.0f, uvxmax);
        float uvy1 = clamp(uvy+1.0f/(float)resy, 0.0f, uvymax);

        float4 uvx0y0 = simple_sample(tex, uvx, uvy, resx, resy);
        float4 uvx1y0 = simple_sample(tex, uvx1, uvy, resx, resy);
        float4 uvx0y1 = simple_sample(tex, uvx, uvy1, resx, resy);
        float4 uvx1y1 = simple_sample(tex, uvx1, uvy1, resx, resy);

        float xfac = maprange(uvx*resx, floor(uvx*resx), floor(1.0f+uvx*resx), 0.0f, 1.0f);
        float yfac = maprange(uvy*resy, floor(uvy*resy), floor(1.0f+uvy*resy), 0.0f, 1.0f);

        float4 uvxxy0 = lerp4(uvx0y0, uvx1y0, xfac);
        float4 uvxxy1 = lerp4(uvx0y1, uvx1y1, xfac);

        return lerp4(uvxxy0, uvxxy1, yfac);
    }
    // cubic... later
    float4 samples[4][4];
    float newuvx, newuvy;
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            newuvx = clamp(uvx+(float)i/(float)resx, 0.0f, uvxmax);
            newuvy = clamp(uvx+(float)i/(float)resx, 0.0f, uvymax);
            samples[i][j] = simple_sample(tex, newuvx, newuvy, resx, resy);
        }
    }
    return simple_sample(tex, round(uvx*resx)/resx, round(uvy*resy)/resy, resx, resy);
}

__kernel void transform(__global const float *in_img, const int inx, const int iny, const int outx, const int outy,
    __global const float *locrotscale, const int interp, const int extend, __global float *res_floats){

    int gid = get_global_id(0);

    float tx = locrotscale[gid*5+0];
    float ty = locrotscale[gid*5+1];
    float rot = locrotscale[gid*5+2];
    float sx = locrotscale[gid*5+3];
    float sy = locrotscale[gid*5+4];

    float ar = (float)(outy) / (float)(outx);
    float outuvx = (float)(gid % outx) / (float)(outx);
    float outuvy = (float)(gid / outx) / (float)(outy);

    float inuvx = outuvx;
    float inuvy = outuvy;
    inuvx = (inuvx-0.5f) * sx + 0.5f;
    inuvy = (inuvy-0.5f) * sy + 0.5f;

    inuvx -= tx;
    inuvy += ty;

    inuvx /= ar;

    inuvx -= (0.5f/ar);
    inuvy -= 0.5f;
    float r = sqrt((inuvx)*(inuvx) + (inuvy)*(inuvy));
    float phi = atan2(inuvy, inuvx);

    phi += rot;
    inuvx = r*cos(phi)*ar + 0.5f;
    inuvy = r*sin(phi) + 0.5f;

    int off = gid*4;
    float4 pix = sample(in_img, inuvx, inuvy, inx, iny, interp, extend);

    res_floats[off+0] = pix.x;
    res_floats[off+1] = pix.y;
    res_floats[off+2] = pix.z;
    res_floats[off+3] = pix.w;
}