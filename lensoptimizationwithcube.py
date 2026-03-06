import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'path',
    },
    'sensor': {
        'type': 'thinlens',
        'id': 'sensor', # ID added for easier traversal
        'focus_distance': 1.0, 
        'aperture_radius': 0.1, 
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0],
            target=[0, 0, 1], # Camera now looks toward the cube at z=1
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
            'pixel_format': 'rgb',
        }
    },
    # The Cube
    'my_cube': {
        'type': 'cube',
        'to_world': mi.ScalarTransform4f.translate([0, 0, 1.2])@mi.ScalarTransform4f.scale([0.2, 0.2, 0.2]),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [0.1, 0.5, 0.8]}
        }
    },
    # Moving the light to illuminate the cube
    'light': {
        'type': 'sphere',
        'center': [1, 1, 0.5], # Positioned to the side to create shadows/shading
        'radius': 0.1,
        'emitter': {
            'type': 'area',
            'radiance': {
                'type': 'rgb',
                'value': 1000.0
            }
        }
    }
}

scene = mi.load_dict(scene_dict)
img_ref = mi.render(scene, spp=512)
mi.util.write_bitmap('reference.png', img_ref)

params = mi.traverse(scene)
# print(params.keys())
key = 'sensor.focus_distance'

param_ref = params[key]
print(f"Reference focus distance: {param_ref[0]:.3f}")
params[key] = 1.5
print(f"Initial focus distance: {params[key][0]:.3f}")
params.update()
img = mi.render(scene, spp=512)
mi.util.write_bitmap('initial.png', img)

opt= mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt)

def mse(image):
    return dr.mean(dr.square(image - img_ref))

iterations = 20
tv_weight = 0.01  # Adjust this weight to balance between MSE and TV
for i in range(iterations):
    img = mi.render(scene, params, seed=i, spp=64)
    loss = mse(img)

    dr.backward(loss)
    opt.step()
    params.update(opt)
    print(f"Iteration {i+1}/{iterations}, Loss: {loss.array[0]:.6f}")

print(f"Final focus distance: {param_ref[0]:.3f}")
img = mi.render(scene, spp=512)
mi.util.write_bitmap('optimized.png', img)

