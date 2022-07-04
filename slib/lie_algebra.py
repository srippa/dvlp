import numpy as np
from scipy.spatial.transform import Rotation as srot

def hat(v: np.ndarray) -> np.ndarray:
    """ The hat operator

    Args:
        v (np.ndarray): 3x1 vector

    Returns:
        np.ndarray: 3x3 skew symmetric matrix
    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def vee(skew_symm: np.ndarray) -> np.ndarray:
    """get 3 elements of a skew-symmetric matrix

    Args:
        skew_symm (np.ndarray): 3x3 skew symmetric matrix

    Returns:
        np.ndarray: 3x1 vector
    """
    return np.array([skew_symm[2, 2], skew_symm[0, 2], skew_symm[1, 0]])

def so3_mul(r1, r2 ):
        """Multiply by other: self \dot other

        Args:
            r1 (SO3): left side of the multiplication
            r2 (SO3): right side of the multiplication

        Returns:
            SO3: the product r1 \cdot r2
        """
        r_mat = np.matmul(r1.as_matrix(), r2.as_matrix())
        return SO3().from_matrix(r_mat)

def random_quaternion3():
    """Return uniform random unit quaternion.
    """
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([
        np.cos(t2)*r2,
        np.sin(t1)*r1,
        np.cos(t1)*r1, 
        np.sin(t2)*r2])

def random_rot3():
    q = random_quaternion3()
    return SO3().from_quat(q).as_matrix()

def random_trans3():
    return np.random.rand(3,1)

def random_trans33():
    return np.random.rand(3,1)

class SO3:
    def __init__(self):
        self._rot_mat = None

    @property
    def R(self):
        return self._rot_mat

    def from_rotvec(self, rotation_vector: np.ndarray):
        """define rotation from [rotation vector](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector)

        Args:
            rotation_vector (np.ndarray): rotation vector defining rotation in angle theta = norm(rotation_vector) around the rotation vector
        Returns:
            SO3: element of SO3
        """
        self._rot_mat = srot.from_rotvec(rotation_vector).as_matrix()

        self._assert()
        return self

    def from_matrix(self, rot_mat: np.ndarray):
        """define from rotation matrix

        Args:
            rot_mat (np.ndarray): 3x3 rotation matrix

        Returns:
            SO3: element of SE3
        """
        self._rot_mat = rot_mat

        self._assert()
        return self
        
    def from_quat(self, q: np.ndarray):
        """Define SO3 from [unit quaternions](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)

        Args:
            q (np.ndarray): 4x1 unit quaternion in the [x,y,z,w] ordering (as scipy.transformation)
        """
        self._rot_mat = srot.from_quat(q).as_matrix()
        
        self._assert()
        return self

    def as_matrix(self) -> np.ndarray:
        """Return matrix representatio 

        Returns:
            np.ndarray: 3x3 transformation matrix
        """
        return self._rot_mat

    def as_rotvec(self):
        return srot.from_matrix(self._rot_mat).as_rotvec()

    def as_quat(self):
        """Return 4x1 unit quaternion in the [x,y,z,w] ordering (as scipy.transformation)
        """
        q = srot.from_matrix(self._rot_mat).as_quat()
        return q

    def inverse(self):
        """return the inverse of the rotation matrix as SO3 lement

        Returns:
            np.ndarray: [description]
        """
        inverse_mat = self._rot_mat.transpose()
        return SO3().from_matrix(inverse_mat)

    def relative(self, other):
        """ get the relative SO3 transformation to other

        Args:
            other SO3: [The other SO3 element
        """

        return SO3().from_matrix(np.dot(self.inverse().as_matrix(), other.as_matrix()))


    def exp(self) -> np.ndarray:
        """Return the exponential map, namely a rotation matrix: so(3) --> SO(3)

        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        return self.as_matrix()

    def log(self) -> np.ndarray:
        """Return the logarithm map: SO(3) --> so(3), namely the rotation vector whose components are angular velocities

        Returns:
            np.ndarray: 3x1 vector as element of so(3)
        """
        return self.as_rotvec()

    
    def transform(self, p: np.array) -> np.array:
        """Transform a 3D point or array of 3D points

        Args:
            p (np.array): 3xn array of 3D points

        Returns:
            np.array: 3xn array of transformed 3D points
        """
        return np.matmul(self.R, p)

    def _assert(self):
        # Check the determinant.
        det_valid = np.allclose(np.linalg.det(self._rot_mat), [1.0], atol=1e-6)
        assert det_valid, "Determinanr of rotation matrix should be 1"
       
        # Check if the transpose is the inverse.
        inv_valid = np.allclose(self._rot_mat.transpose().dot(self._rot_mat), np.eye(3), atol=1e-6)
        assert inv_valid, "The inverse of a rotation matrix should be it's transpose"



class SE3:
    def __init__(self):
        self._so3 = None
        self._t = None
        self._se3_mat = None
        self._twist = None

    @property
    def R(self):
        return self._se3_mat

    @property 
    def t(self):
        return self._t

    @property
    def linear_velocity(self):
        return self._twist[:3]

    @property
    def angular_velocity(self):
        return self._twist[3:]

    def from_matrix(self, rigid_mat: np.ndarray):
        """define from rigid body matrix

        Args:
            rigid_mat (np.ndarray): 4x4 ridig body matrix

        Returns:
            SO3: element of SE3
        """
        r = rigid_mat[:3,:3]
        t = rigid_mat[:3,3]
        return self.from_rot_trans(r, t)


    def from_twist(self, twist: np.ndarray):
        """define rigin body motion from a twist vector 

        Args:
            twist vector (np.ndarray): a 6x1 vector [v,w] composed of a 3D linear velocity vector v and 3D angular velocity vector w
        Returns:
            SE3: element of SE2
        """
        v = twist[:3]
        w = twist[3:]

        rot = SO3().from_rotvec(w).as_matrix()
        return self.from_rot_trans(rot, v)


    def from_rot_trans(self, r: np.ndarray, t: np.ndarray):
        """Define from a rotation matrix and a translation vector

        Args:
            r (np.ndarray): 3x3 rotation matrix
            t (np.ndarray): 3x1 translation vector 
        """
        self._se3_mat = np.eye(4)
        self._se3_mat[:3, :3] = r
        self._se3_mat[:3, 3] = t

        self._so3 = SO3().from_matrix(r)
        self._t = t

        w = self._so3.as_rotvec()
        self._twist = [t[0], t[1], t[2], w[0], w[1], w[2]]

        return self

    def as_matrix(self):
        """return as 4x4 matrix
        """
        return self._se3_mat

    def as_twist(self):
        return self._twist


    def as_so3(self):
        return self._so3

    def inverse(self):
        r_inv = self._so3.inverse().as_matrix()
        t_inv = -r_inv.dot(self._t)
        return SE3().from_rot_trans(r_inv, t_inv)

    def relative(self, other):
        """ get the relative SE3 pose to other

        Args:
            other SE3: The other SE3 element
        """
        return SE3().from_matrix(np.dot(self.inverse().as_matrix(), other.as_matrix()))


    def exp(self) -> np.ndarray:
        """Return the exponential map, namely a rigid body transformation matrix: se(3) --> SE(3)

        Returns:
            np.ndarray: 4x4 rigid body transformation  matrix
        """
        return self.as_matrix()

    def log(self) -> np.ndarray:
        """Return the logarithm map: SE(3) --> se(3), namely the wist vector whose components are linear and angular velocities

        Returns:
            np.ndarray: 6x1 vector as element of se(3)
        """
        return self.as_twist()

    def __call__(self, p: np.array) -> np.array:
        """Transform a 3D point or array of 3D points

        Args:
            p (np.array): 3xn array of 3D points

        Returns:
            np.array: 3xn array of transformed 3D points
        """
        return self.R.dot(p) + self.t


    def _assert(self):
        lower_valid = np.equal(self._se3_mat[3, :], np.array([0.0, 0.0, 0.0, 1.0])).all()
        assert lower_valid, 'last row should be equal to identity'


class SIM3:
    def __init__(self, R, t, c):
        self._t = t.reshape(3,1)
        self._R= R
        self._c = c

    def __str__(self):
        s =  f'R = {self.R}\n'
        s += f't = {self.t}\n'
        s += f's = {self.c}'
        return s

    def as_colmap_similarity_transform_matrix(self) -> np.array:
        """Transform a 3D point or array of 3D points

        Returns:
            np.array: 3x4 similrity transformatrion in a format expected by colmap 
        """
        s = np.zeros((3,4), dtype=np.float64)
        s[:3,:3] = self.c*self.R
        s[:3,3] = self.t.reshape(3)
        return s


    @staticmethod
    def random_sim3(seed=None):
        if seed is not None:
            np.random.seed(seed)
        R = random_rot3()
        t = random_trans3()
        c = np.random.random_sample()
        return SIM3(R,t,c)

    @staticmethod
    def from_dict(d: dict):
        R = np.array(d['R'])
        t = np.array(d['t'])
        c = np.array(d['c'])
        return SIM3(R,t,c)

    def to_dict(self):
        d = dict(
            R=self.R,
            t=self.t,
            c=self.c
        )
        return d

    @property
    def R(self):
        return self._R

    @property 
    def t(self):
        return self._t

    @property 
    def c(self):
        return self._c

    def __call__(self, p: np.array) -> np.array:
        """Transform a 3D point or array of 3D points

        Args:
            p (np.array): 3xn array of 3D points

        Returns:
            np.array: 3xn array of transformed 3D points
        """
        d = self.c*self.R.dot(p) 
        return self.c * self.R.dot(p) + self.t

