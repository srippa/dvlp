import numpy as np
import pyransac3d 
from easydict import EasyDict as edict

class Plane3d:
    def __init__(self, eq):
        """Initialize a plane
          eq[0]*x + eq[1]*y + eq[2]*z + eq[3] = 0

        Args:
            eq (_type_): _description_
        """
        self.eq = eq
        
    @property
    def normal(self):
        return np.array([self.eq[0], self.eq[1], self.eq[2]])

    def z_val(self, x, y):
        return ((self.eq[0] * x + self.eq[1] * y + self.eq[3]) / (-self.eq[2]))
    
    def x_val(self, y, z):
        return ((self.eq[2]*z + self.eq[1]*y + self.eq[3]) / (-self.eq[0]))

    def transform_plane(self, transformation):
        p1 = [0, 1, self.z_val(0, 1)]
        p2 = [1, 0, self.z_val(1, 0)]
        p3 = [1, 1, self.z_val(1, 1)]

        transform_points = transformation(np.array([p1, p2, p3]))
        return Plane3d.find_eq(np.array(transform_points))
    
    def get_span_vectors(self):
        p0 = np.array([self.x_val(0, 0),0,0])     # point on the plane with coordinates x, y=0, z=0
        p1 = np.array([self.x_val(0, 1),0,1])     # point on the plane with coordinates x, y=0, z=1

        v1 = p1 - p0                             # vector p0 --> p1
        print(self.normal)
        v2 = np.cross(v1, self.normal)      # vector on the plane perpendicular to v1
        v1 = v1/np.linalg.norm(v1)               
        v2= v2/np.linalg.norm(v2)                
        
        return edict(origin=p0,v1=v1, v2=v2)
        
    def project_3D_points(self,p):
        p_center = p-self.project_2d['origin']
        return np.matmul(p_center,self.project_2d['projection'].T)
    
    def inject_2D_points(self,p):
        return np.matmul(p,self.project_2d['projection']) + self.project_2d['origin']
        

    @staticmethod
    def normal_to_3_points(p0: np.array, p1: np.array, p2: np.array):
        v1 = p1 - p0         # line p1 -- p0
        v2 = p2 - p0         # line p2 -- p0 
        n = np.cross(v1, v2)
        return n

    @staticmethod
    def from_normal_and_point(n: np.array, p0: np.array):
        eq = np.array([n[0], n[1], n[2], -np.dot(n,p0)])
        return Plane3d(eq)

    @staticmethod
    def from_3_points(p0: np.array, p1: np.array, p2: np.array):
        n = Plane3d.normal_to_3_points(p0,p1,p2)
        return Plane3d.from_normal_and_point(n, p0)

    @staticmethod
    def fit_plane(points, thresh=0.1,minPoints=30):
        plane1 = pyransac3d.Plane()
        best_eq, best_inliers = plane1.fit(points, thresh=thresh,minPoints=minPoints)
        return Plane3d(best_eq), best_inliers

