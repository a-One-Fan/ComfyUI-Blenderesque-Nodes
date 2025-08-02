#define EPS 0.0000001f
#define PI 3.14159f

float maprange(float val, float oldmin, float oldmax, float newmin, float newmax) {
    float fac = (val - oldmin) / (oldmax - oldmin);
    return newmin + fac*(newmax - newmin);
}

float4 maprange4s(float4 val, float oldmin, float oldmax, float newmin, float newmax) {
    float4 fac = (val - oldmin) / (oldmax - oldmin);
    return newmin + fac*(newmax - newmin);
}

float lerp(float a, float b, float fac) {
    return a*(1.0f-fac) + b*fac;
}

float4 lerp4(float4 a, float4 b, float fac) {
    return a*(1.0f-fac) + b*fac;
}

float4 getf4(__global const float* arr, int i){
    return (float4)(arr[i*4+0], arr[i*4+1], arr[i*4+2], arr[i*4+3]);
}

float3 getf3(__global const float* arr, int i){
    return (float3)(arr[i*3+0], arr[i*3+1], arr[i*3+2]);
}

float2 getf2(__global const float* arr, int i){
    return (float2)(arr[i*2+0], arr[i*2+1]);
}

// UV:  (0.0,   0.0) ----- (0.999,   0.0)
//             |                  |
//      (0.999, 0.0) ----- (0.999, 0.999)

// 1D: (0.0, 0.0) ----- (0.999, 0.0) - (0.0, 0.01) ---- (0.999, 0.01) ---- (0.0, 0.999) --- (0.999, 0.999)

float4 simple_sample(__global const float* tex, float uvx, float uvy, int resx, int resy) {
    int pixx = uvx * resx;
    int pixy = (int)(uvy * resy) * resx;

    int off = (pixx + pixy) * 4;

    return (float4)(tex[off+0], tex[off+1], tex[off+2], tex[off+3]);
}

// fmod that properly repeats <0
float repeatmod(float val, float div) {
    if(val < 0.0f) {
        return div+fmod(val, div);
    }
    return fmod(val, div);
}

// mod that repeats like /\/\/\/\/\/
float mirror(float val, float step) {
    return fabs(repeatmod(val+step, step*2.0f)-step);
}

float4 sample(__global const float* tex, float uvx, float uvy, int resx, int resy, 
                int interp, int extend) {

    float uvxmax = (float)(resx-1)/(float)(resx);
    float uvymax = (float)(resy-1)/(float)(resy);
    switch(extend) {
        case 0: // clip
            if(uvx < 0.0f || uvx >= 1.0f || uvy < 0.0f || uvy >= 1.0f) {
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

    if(interp == 0) { // closest
        return simple_sample(tex, round(uvx*resx)/resx, round(uvy*resy)/resy, resx, resy);
    }
    if(interp == 1) { // linear
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
    for(int i=0; i<4; i++) {
        for(int j=0; j<4; j++) {
            newuvx = clamp(uvx+(float)i/(float)resx, 0.0f, uvxmax);
            newuvy = clamp(uvx+(float)i/(float)resx, 0.0f, uvymax);
            samples[i][j] = simple_sample(tex, newuvx, newuvy, resx, resy);
        }
    }
    return simple_sample(tex, round(uvx*resx)/resx, round(uvy*resy)/resy, resx, resy);
}

float2 getuv(int gid, int sizex, int sizey) {
    return (float2)(
        (float)(gid % sizex) / (float)(sizex),
        (float)(gid / sizex) / (float)(sizey)
    );
}

__kernel void transform(__global const float *in_img, const int inx, const int iny, const int outx, const int outy,
    __global const float *locrotscale, const int interp, const int extend, __global float *res_floats) {

    int gid = get_global_id(0);

    float tx = locrotscale[gid*5+0];
    float ty = locrotscale[gid*5+1];
    float rot = locrotscale[gid*5+2];
    float sx = locrotscale[gid*5+3];
    float sy = locrotscale[gid*5+4];

    float ar = (float)(outy) / (float)(outx);
    float2 outuv = getuv(gid, outx, outy);

    float inuvx = outuv.x;
    float inuvy = outuv.y;
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
    //pix = (float4)(outuv, 0.0f, 1.0f);

    res_floats[off+0] = pix.x;
    res_floats[off+1] = pix.y;
    res_floats[off+2] = pix.z;
    res_floats[off+3] = pix.w;
}

// TODO: Might be good but not sure, stare
float hash_to_float(int x, int y, int seed) {
    x ^= seed;
    x += x << 10;
    y -= x;
    x ^= x >> 6;
    x += x << 3;
    y += x;
    x ^= x >> 11;
    x += x << 15;
    y -= x;

    y += seed;
    y ^= y << 8;
    y += y >> 4;
	y ^= y << 13;
	y += y >> 7;
	y ^= y << 17;
    x ^= y;
    const float mapmax = pown(2.0f, 24);
    return fmod(maprange(x / 128, -mapmax, mapmax, 0.0f, 1.0f), 1.0f);
}

// blender/source/blender/nodes/composite/nodes/node_composite_lensdist.cc

#define MINIMUM_DISTORTION -0.999f
#define PROJECTOR_DISPERSION_SCALE 5.0f
#define SCREEN_DISPERSION_SCALE 4.0f
#define DISTORTION_SCALE 4.0f

float compute_distortion_scale(const float distortion, const float distance_squared) {
    return 1.0f / (1.0f + sqrt(max(0.0f, 1.0f - distortion * distance_squared)));
}

float3 compute_chromatic_distortion_scale(const float3 chromatic_distortion,  const float distance_squared) {
    return 1.0f / (1.0f + sqrt(max((float3)(0.0f), 1.0f - chromatic_distortion * distance_squared)));
}

float2 compute_distorted_uv(const float2 uv, const float uv_scale, const int2 size) {
    return (uv * uv_scale + 0.5f);
}

static int compute_number_of_integration_steps_heuristic(const float distortion, const bool use_jitter) {
    if (use_jitter) {
        return distortion < 4.0f ? 2 : (int)(sqrt(distortion + 1.0f));
    }
    return (int)(distortion + 1.0f);
}

int3 compute_number_of_integration_steps(const float3 chromatic_distortion,
                                                const int2 size,
                                                const float2 uv,
                                                const float distance_squared,
                                                const bool use_jitter) {

    float3 distortion_scale = compute_chromatic_distortion_scale(chromatic_distortion,
                                                                distance_squared);
    float2 distorted_uv_red = compute_distorted_uv(uv, distortion_scale.x, size);
    float2 distorted_uv_green = compute_distorted_uv(uv, distortion_scale.y, size);
    float2 distorted_uv_blue = compute_distorted_uv(uv, distortion_scale.z, size);

    float distortion_red = distance(distorted_uv_red * convert_float2(size), distorted_uv_green * convert_float2(size));
    int steps_red = compute_number_of_integration_steps_heuristic(distortion_red, use_jitter);

    float distortion_blue = distance(distorted_uv_green * convert_float2(size), distorted_uv_blue * convert_float2(size));
    int steps_blue = compute_number_of_integration_steps_heuristic(distortion_blue, use_jitter);

    return (int3)(steps_red, steps_red + steps_blue, steps_blue);
}

static float get_jitter(const int gid, const int seed, const bool use_jitter) {
    if (use_jitter) {
        return hash_to_float(gid, 0, seed);
    }
    return 0.5f;
}

float3 integrate_distortion(const int gid,
                                   __global const float* input,
                                   const int2 size,
                                   const float3 chromatic_distortion,
                                   const int start,
                                   const int end,
                                   const float distance_squared,
                                   const float2 uv,
                                   const int steps,
                                   const bool use_jitter) {

    float3 accumulated_color = (float3)(0.0f);
    float distortion_amount = chromatic_distortion[end] - chromatic_distortion[start];
    for (int i = 0; i < steps; i++) {
        float increment = (i + get_jitter(gid, start * steps + i, use_jitter)) / steps;
        float distortion = chromatic_distortion[start] + increment * distortion_amount;
        float distortion_scale = compute_distortion_scale(distortion, distance_squared);

        float2 distorted_uv = compute_distorted_uv(uv, distortion_scale, size);
        float4 color = sample(input, distorted_uv.x, distorted_uv.y, size.x, size.y, 1, 1);
        accumulated_color[start] += (1.0f - increment) * color[start];
        accumulated_color[end] += increment * color[end];
    }
    return accumulated_color;
}

float4 screen_lens_distortion(const int gid,
                                   __global const float* input,
                                   const int2 size,
                                   const float3 chromatic_distortion,
                                   const float scale,
                                   const bool use_jitter){
    float2 uv = scale * (getuv(gid, size.x, size.y) - 0.5f) * 2.0f;
    float distance_squared = dot(uv, uv);

    float3 distortion_bounds = chromatic_distortion * distance_squared;
    if (distortion_bounds.x > 1.0f || distortion_bounds.y > 1.0f || distortion_bounds.z > 1.0f) {
        return (float4)(0.0f);
    }

    int3 number_of_steps = compute_number_of_integration_steps(
        chromatic_distortion, size, uv, distance_squared, use_jitter);

    float3 color = (float3)(0.0f);
    color += integrate_distortion(gid,
                                input,
                                size,
                                chromatic_distortion,
                                0,
                                1,
                                distance_squared,
                                uv,
                                number_of_steps.x,
                                use_jitter);
    color += integrate_distortion(gid,
                                input,
                                size,
                                chromatic_distortion,
                                1,
                                2,
                                distance_squared,
                                uv,
                                number_of_steps.z,
                                use_jitter);

    color *= 2.0f / convert_float3(number_of_steps);

    return (float4)(color, 1.0f);
}

__kernel void execute_projector_distortion(  __global const float* tex, const int sizex, const int sizey, 
                                    __global const float* distortion_buf, __global const float* dispersion_buf,
                                    const int jitterfit, __global float* out) {
    int gid = get_global_id(0);

    bool jitter = jitterfit & 1;
    bool fit = jitterfit & 2;
    float distortion = distortion_buf[gid];
    float dispersion = dispersion_buf[gid];

    dispersion = (dispersion * PROJECTOR_DISPERSION_SCALE) / sizex;

    float2 uv = getuv(gid, sizex, sizey);

    const float red = sample(tex, uv.x + dispersion, uv.y, sizex, sizey, 1, 0).x;
    const float green = tex[gid*4+1];
    const float blue = sample(tex, uv.x - dispersion, uv.y, sizex, sizey, 1, 0).z;

    out[gid*4+0] = red;
    out[gid*4+1] = green;
    out[gid*4+2] = blue;
    out[gid*4+3] = 1.0f;
}

float3 compute_chromatic_distortion(float distortion, float dispersion) {
    const float green_distortion = distortion;
    const float dispersion_scaled = dispersion / SCREEN_DISPERSION_SCALE;
    const float red_distortion = clamp(green_distortion + dispersion_scaled, MINIMUM_DISTORTION, 1.0f);
    const float blue_distortion = clamp(green_distortion - dispersion_scaled, MINIMUM_DISTORTION, 1.0f);
    return (float3)(red_distortion, green_distortion, blue_distortion) * DISTORTION_SCALE;
}

float compute_scale(float distortion, float dispersion, bool fit) {
    const float3 chromatic_distortion = compute_chromatic_distortion(distortion, dispersion) / DISTORTION_SCALE;
    const float maximum_distortion = max(chromatic_distortion[0], max(chromatic_distortion[1], chromatic_distortion[2]));

    if (fit && (maximum_distortion > 0.0f)) {
        return 1.0f / (1.0f + 2.0f * maximum_distortion);
    }
    return 1.0f / (1.0f + maximum_distortion);
}

__kernel void execute_screen_distortion( __global const float* tex, const int sizex, const int sizey, 
                                __global const float* distortion_buf, __global const float* dispersion_buf,
                                const int jitterfit, __global float* out) {
    int gid = get_global_id(0);

    bool jitter = jitterfit & 1;
    bool fit = jitterfit & 2;
    float distortion = distortion_buf[gid];
    float dispersion = dispersion_buf[gid];

    const float scale = compute_scale(distortion, dispersion, fit);
    const float3 chromatic_distortion = compute_chromatic_distortion(distortion, dispersion);

    const int2 size = (int2)(sizex, sizey);
    float4 pix = screen_lens_distortion(gid, tex, size, chromatic_distortion, scale, jitter);

    out[gid*4+0] = pix.x;
    out[gid*4+1] = pix.y;
    out[gid*4+2] = pix.z;
    out[gid*4+3] = pix.w;
}

float4 convert_uv4(float4 uv4){
    uv4.y = 1.0f - uv4.y - EPS;
    return uv4;
}

float3 convert_uvw(float3 uvw){
    uvw.y = 1.0f - uvw.y - EPS;
    return uvw;
}
float2 convert_uv(float2 uv){
    uv.y = 1.0f - uv.y - EPS;
    return uv;
}

__kernel void map_uv(__global const float* tex, const int texx, const int texy,
                     __global const float* uvw,  const int uvx,  const int uvy,
                     const int interp, const int extend, __global float* out) {
    int gid = get_global_id(0);

    float3 uvw_converted = (float3)(uvw[gid*3], uvw[gid*3+1], uvw[gid*3+2]);
    uvw_converted = convert_uvw(uvw_converted); // Blender UV is flipped on Y compared to this

    float4 col = sample(tex, uvw_converted.x, uvw_converted.y, texx, texy, interp, extend);

    out[gid*4+0] = col.x;
    out[gid*4+1] = col.y;
    out[gid*4+2] = col.z;
    out[gid*4+3] = col.w * uvw_converted.z;
}

__kernel void brick_texture(__global const float* uv_in, const int uvwx, const int uvwy,
                            __global const float* color1_in, __global const float* color2_in,
                            __global const float* mortar_in, __global const float* sssbwh,
                            const float offset, const int frequency_offset, const float squash, const int frequency_squash,
                            __global float* out) {
    int gid = get_global_id(0);

    float2 uv_converted = getf2(uv_in, gid);
    uv_converted = convert_uv(uv_converted);

    float4 color1 = getf4(color1_in, gid);
    float4 color2 = getf4(color2_in, gid);
    float4 mortar = getf4(mortar_in, gid);
    float scale =           sssbwh[gid*6+0];
    float mortar_size =     sssbwh[gid*6+1];
    float mortar_smooth =   sssbwh[gid*6+2];
    float bias =            sssbwh[gid*6+3];
    float width =           sssbwh[gid*6+4];
    float height =          sssbwh[gid*6+5];


    float2 uv = (float2)(uv_converted.x, uv_converted.y);
    uv *= scale;

    float topd = fmod(uv.y, height);
    int topi = trunc(uv.y / height);

    float squash_factor = (topi % frequency_squash) != 0 ? 1.0f : squash;
    width *= squash_factor;

    float offset_factor = (topi % frequency_offset) != 0 ? 0.0f : offset;
    offset_factor *= width;
    uv.x += offset_factor;
    float sided = fmod(uv.x, width);
    int sidei = trunc(uv.x / width);

    float vert_dist = fabs(height - topd);
    float hor_dist = fabs(width - sided);
    float dist1 = min(vert_dist, hor_dist);
    float dist2 = min(topd, sided);
    float dist = min(dist1, dist2);
    //dist = clamp(dist, 0.0f, mortar_size);

    float mortar_fac = maprange(dist, (1.0f-mortar_smooth) * mortar_size, mortar_size, 1.0f, 0.0f);
    mortar_fac = clamp(mortar_fac, 0.0f, 1.0f);
    float color_fac = hash_to_float(topi, sidei, 42);
    //color_fac = maprange(color_fac+bias, -1.0, bias, 0.0f, 1.0f);
    color_fac = clamp(color_fac+bias, 0.0f, 1.0f);

    float4 res_color;
    res_color = lerp4(color1, color2, color_fac);
    res_color = lerp4(res_color, mortar, mortar_fac);

    out[gid*5+0] = res_color.x;
    out[gid*5+1] = res_color.y;
    out[gid*5+2] = res_color.z;
    out[gid*5+3] = res_color.w;
    out[gid*5+4] = mortar_fac;
}

__kernel void checker_texture(__global const float* uvw_in, const int uvwx, const int uvwy,
                            __global const float* color1_in, __global const float* color2_in, __global const float* scale_in,
                            __global float* out) {
    int gid = get_global_id(0);

    float3 uvw_converted = getf3(uvw_in, gid);
    uvw_converted = convert_uvw(uvw_converted);

    float4 color1 = getf4(color1_in, gid);
    float4 color2 = getf4(color2_in, gid);
    float scale = scale_in[gid];

    uvw_converted *= scale;

    int color_fac;
    color_fac += floor(repeatmod(uvw_converted.x, 2.0f));
    color_fac += floor(repeatmod(uvw_converted.y, 2.0f));
    color_fac += floor(repeatmod(uvw_converted.z, 2.0f));
    color_fac = color_fac % 2;
    color_fac = 1 - color_fac; // Mimic Blender behavior, where the factor is swapped

    float4 res_color;
    res_color = color_fac ? color1 : color2;

    out[gid*5+0] = res_color.x;
    out[gid*5+1] = res_color.y;
    out[gid*5+2] = res_color.z;
    out[gid*5+3] = res_color.w;
    out[gid*5+4] = color_fac;
}

// Permutation is between 0-15 (0000, 0001, 0010, ... 1111)
float4 get_corner(const float4 uv4, int permutation){
    float x = floor(uv4.x) + (float)((permutation >> 0) % 2);
    float y = floor(uv4.y) + (float)((permutation >> 1) % 2);
    float z = floor(uv4.z) + (float)((permutation >> 2) % 2);
    float w = floor(uv4.w) + (float)((permutation >> 3) % 2);

    return (float4)(x, y, z, w);
}

float4 fract4(const float4 v){
    return v - floor(v);
}

float4 hash4_2(const float4 uv4, int seed){
    float3 hs;
    float4 res;
    hs.x = hash_to_float((int)(uv4.x), (int)(uv4.y), seed);
    hs.y = hash_to_float((int)(uv4.y)+8151, (int)(uv4.z)-88, seed+81923);
    hs.z = hash_to_float((int)(uv4.z)+75182, (int)(uv4.w)+5924, seed-79341);
    hs.x = maprange(hs.x, 0.0f, 1.0f, 0.0f, PI * 2.0f);
    hs.y = maprange(hs.y, 0.0f, 1.0f, 0.0f, PI * 2.0f);
    hs.z = maprange(hs.z, 0.0f, 1.0f, 0.0f, PI * 2.0f);
        
    res.x = cos(hs.x);
    res.y = sin(hs.x) * cos(hs.y);
    res.z = sin(hs.x) * sin(hs.y) * cos(hs.z);
    res.w = sin(hs.x) * sin(hs.y) * sin(hs.z);
    return res;
}

float4 get_gradient(int i){
    float4 gradients[32] = {
        (float4)(0.0f, 1.0f, 1.0f, 1.0f),
        (float4)(0.0f, 1.0f, 1.0f, -1.0f),
        (float4)(0.0f, 1.0f, -1.0f, 1.0f),
        (float4)(0.0f, -1.0f, 1.0f, 1.0f),
        (float4)(0.0f, 1.0f, -1.0f, -1.0f),
        (float4)(0.0f, -1.0f, 1.0f, -1.0f),
        (float4)(0.0f, -1.0f, -1.0f, 1.0f),
        (float4)(0.0f, -1.0f, -1.0f, -1.0f),

        (float4)(1.0f, 0.0f, 1.0f, 1.0f),
        (float4)(1.0f, 0.0f, 1.0f, -1.0f),
        (float4)(1.0f, 0.0f, -1.0f, 1.0f),
        (float4)(-1.0f, 0.0f, 1.0f, 1.0f),
        (float4)(1.0f, 0.0f, -1.0f, -1.0f),
        (float4)(-1.0f, 0.0f, 1.0f, -1.0f),
        (float4)(-1.0f, 0.0f, -1.0f, 1.0f),
        (float4)(-1.0f, 0.0f, -1.0f, -1.0f),

        (float4)(1.0f, 1.0f, 0.0f, 1.0f),
        (float4)(1.0f, 1.0f, 0.0f, -1.0f),
        (float4)(-1.0f, 1.0f, 0.0f, 1.0f),
        (float4)(1.0f, -1.0f, 0.0f, 1.0f),
        (float4)(-1.0f, 1.0f, 0.0f, -1.0f),
        (float4)(1.0f, -1.0f, 0.0f, -1.0f),
        (float4)(-1.0f, -1.0f, 0.0f, 1.0f),
        (float4)(-1.0f, -1.0f, 0.0f, -1.0f),

        (float4)(1.0f, 1.0f, 1.0f, 0.0f),
        (float4)(-1.0f, 1.0f, 1.0f, 0.0f),
        (float4)(1.0f, 1.0f, -1.0f, 0.0f),
        (float4)(1.0f, -1.0f, 1.0f, 0.0f),
        (float4)(-1.0f, 1.0f, -1.0f, 0.0f),
        (float4)(-1.0f, -1.0f, 1.0f, 0.0f),
        (float4)(1.0f, -1.0f, -1.0f, 0.0f),
        (float4)(-1.0f, -1.0f, -1.0f, 0.0f),
    };
    return gradients[i%32];
}

float4 fade(float4 t) {
    // 6t^5 - 15t^4 + 10t^3
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float perlin(float4 uv4, int seed){
    float4 uv4_fract = fract4(uv4);
    float res=0.0f, dots[16];
    float lerp1[8], lerp2[4], lerp3[2];

    for(int i=0; i<16; i++){
        float4 corner = get_corner(uv4, i);
        float4 offset = uv4 - corner;

        // Not sure which gradient implementation is better.
        //float4 cornervec = get_gradient((int)(floor(maprange(hash_to_float((int)(corner[0]-4096.0f), (int)(corner[1]-4096.0f), seed) + hash_to_float((int)(corner[2]+4096.0f), (int)(corner[3]+4096.0f), seed), 0.0f, 2.0f, 0.0f, 31.999f))));
        float4 cornervec=hash4_2(corner, seed);

        cornervec = normalize(cornervec);
        
        dots[i] = dot(offset, cornervec);
    }

    uv4_fract = fade(uv4_fract);

    for(int i=0; i<16; i+=2){
        lerp1[i/2] = lerp(dots[i], dots[i+1], uv4_fract.x);
    }
    for(int i=0;i<16; i+=4){
        lerp2[i/4] = lerp(lerp1[i/2], lerp1[i/2+1], uv4_fract.y);
    }
    lerp3[0] = lerp(lerp2[0], lerp2[1], uv4_fract.z);
    lerp3[1] = lerp(lerp2[2], lerp2[3], uv4_fract.z);
    res = lerp(lerp3[0], lerp3[1], uv4_fract.w);
    
    //res = 1.0f + res / 2.0f;

    return res;
}

float4 perlin4(const float4 uv4, int seed){
    return (float4)(perlin(uv4, seed*32+0), perlin(uv4, seed*32+1), perlin(uv4, seed*32+2), perlin(uv4, seed*32+3));
}

// Types:
// 0 - Multifractal
// 1 - Ridged multifractal
// 2 - Hybrid multifractal
// 3 - Fractal Brownian Motion (fBM)
// 4 - Hetero Terrain
__kernel void noise_texture(__global const float* uv4_in, const int uvx, const int uvy,
                            __global const float* sdrlogd, const int type, const int normalize, __global float* out) {
    int gid = get_global_id(0);

    float4 uv4_converted = getf4(uv4_in, gid);
    uv4_converted = convert_uv4(uv4_converted);

    float scale =       sdrlogd[gid*7+0];
    float detail =      sdrlogd[gid*7+1];
    float roughness =   sdrlogd[gid*7+2];
    float lacunarity =  sdrlogd[gid*7+3];
    float offset =      sdrlogd[gid*7+4];
    float gain =        sdrlogd[gid*7+5];
    float distortion =  sdrlogd[gid*7+6];

    float4 res = (float4)(0.0f);

    if(type == 3){
        float4 detail_current=perlin4(scale*uv4_converted, 0), detail_lower=detail_current;
        for(int i=0; i<ceil(detail); i++){
            detail_lower=detail_current;
            scale *= lacunarity;
            detail_current += roughness * perlin4(scale*uv4_converted, i+1);
            roughness *= roughness;
        }
        res = lerp4(detail_lower, detail_current, detail-floor(detail));
        if (normalize){
            res = maprange4s(res, -1.0f-detail, 1.0f+detail, 0.0f, 1.0f);
        }
    }

    out[gid*3+0] = res.x;
    out[gid*3+1] = res.y;
    out[gid*3+2] = res.z;
}