import mitsuba as mi
import drjit as dr

# a mitsuba sensor with a differentible camera matrix and distortion parameters (using the Brown-Conrady distortion model)
class Camera(mi.Sensor):
    def __init__(self, props):
        """
        Props:
            'fx', 'fy', 'cx', 'cy' _Float_: camera matrix coefficents from opencv
            'k1', 'k2', 'p1', 'p2', 'k3' _Float_: distortion coefficents from opencv
            'aperture_radius' _Float_
            'focus_distance' _Float_
            & inherited sensor props
        """
        super().__init__(props)
        
        film_size = self.film().size()
        w, h = float(film_size.x), float(film_size.y)
        
        self.fx = mi.Float(props.get('fx', 0.0))
        self.fy = mi.Float(props.get('fy', 0.0))
        self.cx = mi.Float(props.get('cx', 0.0))
        self.cy = mi.Float(props.get('cy', 0.0))
        
        self.aperture_radius = mi.Float(props.get('aperture_radius', 0.0))
        self.focus_distance = mi.Float(props.get('focus_distance', 1.0))
        
        self.k1 = props.get('k1', 0.0)
        self.k2 = props.get('k2', 0.0)
        self.p1 = props.get('p1', 0.0)
        self.p2 = props.get('p2', 0.0)
        self.k3 = props.get('k3', 0.0)

    def sample_ray(self, time, wavelength_sample, sample_pos, aperture_sample, active=True):
        
        dummy_si = dr.zeros(mi.SurfaceInteraction3f)
        wav, spec = self.sample_wavelengths(dummy_si, wavelength_sample, active)
        
        # convert to normalized coords
        x = -(sample_pos.x - self.cx) / self.fx
        y = -(sample_pos.y - self.cy) / self.fy
        
        # apply distortion
        
        x2 = x**2
        y2 = y**2
        xy = x*y
        r2 = x2 + y2
        r4 = r2*r2
        # radial
        distr = (1.0 + self.k1*r2 + self.k2*r4 + self.k3*r4*r2)
        # tangential
        distx = 2.0*self.p1*xy + self.p2*(3.0*x2 + y2)
        disty = 2.0*self.p2*xy + self.p1*(3.0*y2 + x2)

        xd = x*distr + distx
        yd = y*distr + disty
        
        # central ray
        crd = dr.normalize(mi.Vector3f(xd, yd, 1.0))
        #cro = mi.Point3f(0.0, 0.0, 0.0)
        
        # point on lens
        lens_pos = self.aperture_radius * mi.warp.square_to_uniform_disk(aperture_sample)
        
        # point on focal plane
        p_focus = crd*self.focus_distance/crd.z
        
        # ray
        ro = mi.Point3f(lens_pos.x, lens_pos.y, 0.0)
        rd = dr.normalize(p_focus - ro)

        CtoW = self.world_transform()
        ray =  mi.Ray3f(
            o = CtoW @ ro,
            d = CtoW @ rd,
            time = time,
            wavelengths = wav
        )
        
        return ray, spec

    def traverse(self, callback):
        callback.put('fx', self.fx, mi.ParamFlags.Differentiable)
        callback.put('fy', self.fy, mi.ParamFlags.Differentiable)
        callback.put('cx', self.cx, mi.ParamFlags.Differentiable)
        callback.put('cy', self.cy, mi.ParamFlags.Differentiable)
        
        callback.put('aperture_radius', self.aperture_radius, mi.ParamFlags.Differentiable)
        callback.put('focus_distance', self.focus_distance, mi.ParamFlags.Differentiable)
        
        callback.put('k1', self.k1, mi.ParamFlags.Differentiable)
        callback.put('k2', self.k2, mi.ParamFlags.Differentiable)
        callback.put('k3', self.k3, mi.ParamFlags.Differentiable)
        callback.put('p1', self.p1, mi.ParamFlags.Differentiable)
        callback.put('p2', self.p2, mi.ParamFlags.Differentiable)
        
        super().traverse(callback)

mi.register_sensor("distorted_camera", lambda props: Camera(props))