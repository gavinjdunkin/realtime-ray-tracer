#define PI 			3.1415926535
#define MAXFLOAT	99999.99

// change these parameters for better quality (and lower framerate :-) )
#define MAXDEPTH 	5
#define NUMSAMPLES 	4
#define ROTATION	true

//
// Hash functions by Nimitz:
// https://www.shadertoy.com/view/Xt3cDn
//

uint base_hash(uvec2 p) {
    p = 1103515245U*((p >> 1U)^(p.yx));
    uint h32 = 1103515245U*((p.x)^(p.y>>3U));
    return h32^(h32 >> 16);
}

float g_seed = 0.;

float hash1(inout float seed) {
    uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    return float(n)/float(0xffffffffU);
}

vec2 hash2(inout float seed) {
    uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    uvec2 rz = uvec2(n, n*48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU))/float(0x7fffffff);
}

vec3 hash3(inout float seed) {
    uint n = base_hash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    uvec3 rz = uvec3(n, n*16807U, n*48271U);
    return vec3(rz & uvec3(0x7fffffffU))/float(0x7fffffff);
}


// random number generator
vec2 randState;



float rand2D()
{
    randState.x = fract(sin(dot(randState.xy, vec2(12.9898, 78.233))) * 43758.5453);
    randState.y = fract(sin(dot(randState.xy, vec2(12.9898, 78.233))) * 43758.5453);;
    
    return randState.x;
}
struct Ray
{
    vec3 origin;
    vec3 direction;
};


struct material
{
    int type;
    vec3 albedo;
    float v;};
struct hit_record
{
    // surface properties
    float t;
    vec3  p;
    vec3  normal;
	material mat;
};
    
    
struct Sphere
{
    // sphere properties
    vec3 center;
    float radius;
};

    
bool sphere_hit(Sphere sphere, Ray r, float ray_tmin, float ray_tmax, out hit_record rec)
{
    vec3 oc = r.origin - sphere.center;
        float a = dot(r.direction, r.direction);
        float half_b = dot(oc, r.direction);
        float c = dot(oc, oc) - sphere.radius*sphere.radius;

        float discriminant = half_b*half_b - a*c;
        if (discriminant < 0.0) return false;
        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (-half_b - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (-half_b + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.origin + rec.t * r.direction;
        rec.normal = (rec.p - sphere.center) / sphere.radius;

        return true;
}





float schlick(float cosine, float ior) {
    float r0 = (1.-ior)/(1.+ior);
    r0 = r0*r0;
    return r0 + (1.-r0)*pow((1.-cosine),5.);
}


bool refract_vec(vec3 v, vec3 n, float ni_over_nt, out vec3 refracted)
{
    vec3 uv = normalize(v);

    float dt = dot(uv, n);

    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0f - dt * dt);

    if (discriminant > 0.0f)
    {
        refracted = ni_over_nt*(uv - n * dt) - n * sqrt(discriminant);

        return true;
    }
    else
        return false;
}


vec3 reflectVec(vec3 v, vec3 n)
{
     return v - 2.0f * dot(v, n) * n;
}




// random direction in unit sphere (for lambert brdf)
vec3 random_in_unit_sphere()
{
    float phi = 2.0 * PI * rand2D();
    float cosTheta = 2.0 * rand2D() - 1.0;
    float u = rand2D();

    float theta = acos(cosTheta);
    float r = pow(u, 1.0 / 3.0);

    float x = r * sin(theta) * cos(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(theta);

    return vec3(x, y, z);
}


// random point on unit disk (for depth of field camera)
vec3 random_in_unit_disk()
{
    float spx = 2.0 * rand2D() - 1.0;
    float spy = 2.0 * rand2D()- 1.0;

    float r, phi;


    if(spx > -spy)
    {
        if(spx > spy)
        {
            r = spx;
            phi = spy / spx;
        }
        else
        {
            r = spy;
            phi = 2.0 - spx / spy;
        }
    }
    else
    {
        if(spx < spy)
        {
            r = -spx;
            phi = 4.0f + spy / spx;
        }
        else
        {
            r = -spy;

            if(spy != 0.0)
                phi = 6.0 - spx / spy;
            else
                phi = 0.0;
        }
    }

    phi *= PI / 4.0;


    return vec3(r * cos(phi), r * sin(phi), 0.0f);
}


struct Camera {
    vec3 lookfrom;
    vec3 pixel_delta_u, pixel_delta_v;
    vec3 lowerLeftCorner;
    float focus_dist;
    float defocus_angle;
    vec3 u, v, w;
    float lens_radius;
    vec3   defocus_disk_u;  // Defocus disk horizontal radius
    vec3   defocus_disk_v;  // Defocus disk vertical radius
};
void Camera_init(out Camera cam, float vfov, vec3 lookfrom, vec3 lookat, vec3 vup, float focus_dist, float aspect, float aperature, float defocus_angle) {
    cam.lookfrom = lookfrom;
    cam.focus_dist = focus_dist;
    cam.lens_radius = aperature / 2.0;
    cam.w = normalize(lookfrom - lookat);
    cam.u = normalize(cross(vup, cam.w));
    cam.v = cross(cam.w, cam.u);

    float theta = vfov * PI / 180.0;
    float h = tan(theta/2.0);
    float viewport_height = 2.0 * h * focus_dist;
    float viewport_width = viewport_height * aspect;
    
    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    cam.pixel_delta_u = viewport_width * cam.u;    // Vector across viewport horizontal edge
    cam.pixel_delta_v = viewport_height * cam.v;  // Vector down viewport vertical edge

    
    cam.lowerLeftCorner = cam.lookfrom  - aspect * h  * focus_dist * cam.u
                                            - h * focus_dist * cam.v
                                            -              focus_dist * cam.w;


    // Calculate the camera defocus disk basis vectors.
    float defocus_radius = focus_dist * tan(PI *  (defocus_angle / 2.0) / 180.0);
    cam.defocus_disk_u = cam.u * defocus_radius;
    cam.defocus_disk_v = cam.v * defocus_radius;
    cam.defocus_angle = defocus_angle;
}


Ray Camera_getRay(Camera camera, float s, float t)
{
    vec3 rd = camera.lens_radius * random_in_unit_disk();
    vec3 offset = camera.u * rd.x + camera.v * rd.y;

    Ray ray;

    ray.origin = camera.lookfrom + offset;
    ray.direction = camera.lowerLeftCorner + s * camera.pixel_delta_u + t * camera.pixel_delta_v - camera.lookfrom - offset;

    return ray;
}
#define LAMBERTIAN    0
#define METAL      1
#define DIELECTRIC 2

    bool material_scatter(const in Ray r_in, const in hit_record rec, out vec3 attenuation, 
                      out Ray scattered) {
    if(rec.mat.type == LAMBERTIAN) {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = Ray(rec.p, target - rec.p);
        attenuation = rec.mat.albedo;
        return true;
    } else if(rec.mat.type == METAL) {
        vec3 rd = reflect(r_in.direction, rec.normal);
        scattered = Ray(rec.p, normalize(rd + rec.mat.v*random_in_unit_sphere()));
        attenuation = rec.mat.albedo;
        return true;
    } else if(rec.mat.type == DIELECTRIC) {
    
    //random number usage adapted from reinder
    // https://www.shadertoy.com/view/XlycWh
        vec3 outward_normal, refracted, 
             reflected = reflect(r_in.direction, rec.normal);
        float ni_over_nt, reflect_prob, cosine;
        
        attenuation = vec3(1);
        if (dot(r_in.direction, rec.normal) > 0.) {
            outward_normal = -rec.normal;
            ni_over_nt = rec.mat.v;
            cosine = dot(r_in.direction, rec.normal);
            cosine = sqrt(1. - rec.mat.v*rec.mat.v*(1.-cosine*cosine));
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1. / rec.mat.v;
            cosine = -dot(r_in.direction, rec.normal);
        }
        
        if (refract_vec(r_in.direction, outward_normal, ni_over_nt, refracted)) {
	        reflect_prob = schlick(cosine, rec.mat.v);
        } else {
            reflect_prob = 1.;
        }
        
        if (rand2D() < reflect_prob) {
            scattered = Ray(rec.p, reflected);
        } else {
            scattered = Ray(rec.p, refracted);
        }
        return true;
    }
    return false;
    }


// uses the procedural scene generation 
bool world_hit(Ray r, float t_min, float t_max, out hit_record rec)
{
        rec.t = t_max;
    bool hit = false;
  	if (sphere_hit(Sphere(vec3(0,-1000,-1),1000.),r,t_min,rec.t,rec)) hit=true,rec.mat=material(LAMBERTIAN,vec3(.5),0.);

  	if (sphere_hit(Sphere(vec3( 0,1,0),1.),r,t_min,rec.t,rec))        hit=true,rec.mat=material(DIELECTRIC,vec3(0),1.5);
    if (sphere_hit(Sphere(vec3(-4,1,0),1.),r,t_min,rec.t,rec))        hit=true,rec.mat=material(LAMBERTIAN,vec3(.4,.2,.1),0.);
	if (sphere_hit(Sphere(vec3( 4,1,0),1.),r,t_min,rec.t,rec))        hit=true,rec.mat=material(METAL     ,vec3(.7,.6,.5),0.);
    
    //random number usage adapted from reinder
    // https://www.shadertoy.com/view/XlycWh
    int NO_UNROLL = min(0,iFrame);
    for (int a = -11; a < 11+NO_UNROLL; a++) {
        for (int b = -11; b < 11+NO_UNROLL; b++) {
            float m_seed = float(a) + float(b)/1000.;
            vec3 rand1 = hash3(m_seed);            
            vec3 center = vec3(float(a)+.9*rand1.x,.2,float(b)+.9*rand1.y); 
            float choose_mat = rand1.z;
            
            if (distance(center,vec3(4,.2,0)) > .9) {
                if (choose_mat < .8) { // diffuse
                    if (sphere_hit(Sphere(center,.2),r,t_min,rec.t,rec)) {
                        hit=true, rec.mat=material(LAMBERTIAN, hash3(m_seed)* hash3(m_seed),0.);
                    }
                } else if (choose_mat < 0.95) { // metal
                    if (sphere_hit(Sphere(center,.2),r,t_min,rec.t,rec)) {
                        hit=true, rec.mat=material(METAL,.5*(hash3(m_seed)+1.),.5*hash1(m_seed));
                    }
                } else { // glass
                    if (sphere_hit(Sphere(center,.2),r,t_min,rec.t,rec)) {
                        hit=true, rec.mat=material(DIELECTRIC,vec3(0),1.5);
                    }
                }
            }
        }
    }
    return hit;
}



vec3 ray_color(Ray r) {
        vec3 col = vec3(1);
        hit_record rec;

        for(int i = 0; i < MAXDEPTH; i++) {

        if (world_hit(r, 0.001, 99999.9, rec)) {
            Ray scattered;
            vec3 attenuation;
            if (material_scatter(r, rec, attenuation, scattered)) {
                col *= attenuation;
                r = scattered;
                }
            else {
                return vec3(0);
                }
        }
        else {
        vec3 unit_direction = normalize(r.direction);
        float a = 0.5*(unit_direction.y + 1.0);
        return col * ((1.0-a)*vec3(1.0, 1.0, 1.0) + a*vec3(0.5, 0.7, 1.0));
        }
    }
    return vec3(0);
    }

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{   
    // fetch the stored offset from the keyboard
    vec2 offsetkey = texelFetch( iChannel0, ivec2(0,0), 0 ).xy * 10.0;
    vec3 lookfrom = vec3(13.0, 2.0, 3.0);
    vec3 lookat = vec3(offsetkey.x, 1.0, offsetkey.y);
    float distToFocus = 10.0;
    float aperture = 0.1;
    
    //fetch the stored rotational matrix from the mouse
    vec2 offset = texelFetch( iChannel1, ivec2(0,0), 0 ).xy;
    	mat4 rotationMatrix = mat4(cos(offset.x), cos(offset.y), sin(offset.x),sin(offset.y),
                                          0.0, 1.0,        0.0, 0.0,
                                 -sin(offset.x),  -sin(offset.y), cos(offset.x), cos(offset.y),
                                         0.0,  0.0,        0.0, 1.0);
    
    lookfrom = vec3(rotationMatrix * vec4(lookfrom, 1.0));
    lookfrom.x += offsetkey.x;
    lookfrom.z += offsetkey.y;
    if(lookfrom.y < 0.) lookfrom.y = 0.;
    Camera camera;
    Camera_init(camera, 20.0f, lookfrom, lookat, vec3(0.0f, 1.0f, 0.0f),distToFocus, float(iResolution.x) / float(iResolution.y), aperture, 10.0f);


    randState = fragCoord.xy / iResolution.xy;
    
    vec3 col = vec3(0.0, 0.0, 0.0);

    for (int s = 0; s < NUMSAMPLES; s++)
    {
        float u = float(fragCoord.x + rand2D()) / float(iResolution.x);
        float v = float(fragCoord.y + rand2D()) / float(iResolution.y);

        Ray r = Camera_getRay(camera, u, v);
        col += ray_color(r);
    }

    col /= float(NUMSAMPLES);
    col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

    fragColor = vec4(col, 1.0);
}
