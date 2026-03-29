import mitsuba as mi
import drjit as dr

# todo
# a mitsuba sensor with a differentible camera matrix and distortion parameters (using the Brown-Conrady distortion model)
class Camera(mi.Sensor):
    # note: parameters are in normalized coords
    def __init__(self, props):
        """
        Props:
            'camera_matrix' _ScalarMatrix3f_: camera matrix from opencv
            'distortion_coefficents' _List<Float>_: coefficents from opencv
            'aperture_radius' _Float_
            'focus_distance' _Float_
            & inherited sensor props
        """
        super().__init__(props)
        
        film_size = self.film().size()
        w, h = float(film_size.x), float(film_size.y)
        
        cm = props.get('camera_matrix', mi.ScalarMatrix3f(1000, 0, 4000, 0, 1000, 2700, 0, 0, 1))
        
        self.fx = mi.Float(cm[0, 0]/w)
        self.fy = mi.Float(cm[1, 1]/h)
        self.cx = mi.Float(cm[0, 2]/w)
        self.cy = mi.Float(cm[1, 2]/h)
        
        self.aperture_radius = mi.Float(props.get('aperture_radius', 0.0))
        self.focus_distance = mi.Float(props.get('focus_distance', 1.0))
        
        dist = props.get('distortion_coefficients', [0.0] * 5)
        self.k1 = mi.Float(dist[0])
        self.k2 = mi.Float(dist[1])
        self.p1 = mi.Float(dist[2])
        self.p2 = mi.Float(dist[3])
        self.k3 = mi.Float(dist[4])

    def sample_ray(self, time, sample_pos, aperture_sample, active=True):
        # convert to normalized coords
        x = (sample_pos.x - self.cx) / self.fx
        y = (sample_pos.y - self.cy) / self.fy
        
        # apply distortion
        x2 = x**2
        y2 = y**2
        xy = x*y
        r2 = x2 + y2
        r4 = r2*r2
        # radial
        dr = (1.0 + self.k1*r2 + self.k2*r4 + self.k3*r4*r2)
        # tangential
        dx = 2.0*self.p1*xy + self.p2*(3.0*x2 + y2)
        dy = 2.0*self.p2*xy + self.p1*(3.0*y2 + x2)

        xd = x*dr + dx
        yd = y*dr + dy
        
        # central ray
        crd = dr.normalize(mi.Vector3f(xd, yd, 1.0))
        #cro = mi.Point3f(0.0, 0.0, 0.0)
        
        # point on lens
        lens_pos = self.aperture_radius * mi.warp.to_uniform_disk(aperture_sample)
        
        # point on focal plane
        p_focus = crd*self.focus_distance/crd.z
        
        # ray
        ro = mi.Point3f(lens_pos.x, lens_pos.y, 0.0)
        rd = dr.normalize(p_focus - ro)

        CtoW = self.world_transform().eval(time, active)
        return mi.Ray3f(
            o = CtoW @ ro,
            d = CtoW @ rd,
            time = time,
            wavelengths = []
        )

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