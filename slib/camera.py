import copy
import numpy as np
import cv2
from easydict import EasyDict as edict

from slib.path_utils import read_json_file, write_json_file
from slib.lie_algebra import SE3, SO3

#-------------------------------------------------------------------------------------
# see https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
# camera_model_id   name             params
# 0                 SIMPLE_PINHOLE   [f, cx, cy]
# 1                 PINHOLE          [fx, fy, cx, cy]
# 2                 SIMPLE_RADIAL    [f, cx, cy, k]
# 3                 RADIAL           [f, cx, cy, k1, k2]
# 4                 OPENCV           [fx, fy, cx, cy, k1, k2, p1, p2]
# 5                 OPENCV_FISHEYE   [fx, fy, cx, cy, k1, k2, k3, k4]
# 6                 FULL_OPENCV      [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
# 7                 FOV              [fx, fy, cx, cy, omega]
# NA                OPENCV5          [fx, fy, cx, cy, k1, k2, p1, p2, k3]
#--------------------------------------------------------------------------------------
COLMAP_CAMERA_MODELS = dict(
    SIMPLE_PINHOLE = dict(id=0, n_params=3),
    PINHOLE        = dict(id=1, n_params=4),
    SIMPLE_RADIAL  = dict(id=2, n_params=4),
    RADIAL         = dict(id=3, n_params=5),
    OPENCV         = dict(id=4, n_params=8),
    OPENCV_FISHEYE = dict(id=5, n_params=8),
    FULL_OPENCV    = dict(id=6, n_params=12),
    FOV            = dict(id=7, n_params=5),
    UNKNOWN        = dict(id=-1, n_params=0),
    OPENCV5        = dict(id=-1, n_params=9)
)


class CameraIntrinsicts:
    def __init__(self, 
                 camera_model_name: str, 
                 width, 
                 height, 
                 params: list):
        # prior_focal_length : 1 if we have confidence in the modelparameters and 0 if we do not trust the model parameters

        if camera_model_name not in COLMAP_CAMERA_MODELS:
            raise ValueError(f'Camera model ["{camera_model_name}"] not recognized as colmap camera model')
        
        if len(params) != COLMAP_CAMERA_MODELS[camera_model_name]['n_params']:
            n_params = COLMAP_CAMERA_MODELS[camera_model_name]['n_params']
            raise ValueError(f'Expected {n_params} parameters for camera ["{camera_model_name}"] but got {len(params)}') 

        self._w = width
        self._h = height

        self.camera_model_name = camera_model_name
        self._params = params

        if camera_model_name == 'SIMPLE_PINHOLE':
            self.fx = self.fy = self.f = params[0]
            self.cx = params[1]
            self.cy = params[2]
            self._D = []
        elif camera_model_name == 'PINHOLE':
            self.fx = params[0]
            self.fy = params[1]
            self.cx = params[2]
            self.cy = params[3]
            self._D = []
        elif camera_model_name == 'SIMPLE_RADIAL':
            self.fx = self.fy = self.f = params[0]
            self.cx = params[1]
            self.cy = params[2]
            self.k  = params[3]
            self._D = [self.k]
        elif camera_model_name == 'RADIAL':
            self.fx = self.fy = self.f = params[0]
            self.cx = params[1]
            self.cy = params[2]
            self.k1 = params[3]
            self.k2 = params[4]
            self._D = params[3:]
        elif camera_model_name == 'OPENCV':
            self.fx = params[0]
            self.fy = params[1]
            self.cx = params[2]
            self.cy = params[3]
            self.k1 = params[4]
            self.k2 = params[5]
            self.p1 = params[6]
            self.p2 = params[7]
            self._D = params[4:]
        elif camera_model_name == 'OPENCV5':
            self.fx = params[0]
            self.fy = params[1]
            self.cx = params[2]
            self.cy = params[3]
            self.k1 = params[4]
            self.k2 = params[5]
            self.p1 = params[6]
            self.p2 = params[7]
            self._D = params[4:]
        elif camera_model_name == 'OPENCV_FISHEYE':
            self.fx = params[0]
            self.fy = params[1]
            self.cx = params[2]
            self.cy = params[3]
            self.k1 = params[4]
            self.k2 = params[5]
            self.k3 = params[6]
            self.k4 = params[7]
            self._D = params[4:]
        elif camera_model_name == 'FULL_OPENCV':
            self.fx = params[0]
            self.fy = params[1]
            self.cx = params[2]
            self.cy = params[3]
            self.k1 = params[4]
            self.k2 = params[5]
            self.p1 = params[6]
            self.p2 = params[7] 
            self.k3 = params[8]
            self.k4 = params[9]
            self.k5 = params[10]
            self.k6 = params[11]
            self._D = params[2:]
        elif camera_model_name == 'FOV':
            self.fx = params[0]
            self.fy = params[1]
            self.cx = params[2]
            self.cy = params[3]
            self.omega = params[4]
            self._D = params[2:]
        else:
            raise(f'This should not happen')

    @property 
    def w(self):
        return self._w

    @property 
    def width(self):
        return self._w

    @property 
    def h(self):
        return self._h

    @property 
    def height(self):
        return self._h


    def K(self, use_homogenous_coordinates=True):
        K_mat = np.array(
            [
                [self.fx, 0.0,     self.cx, 0.0],
                [0.0,     self.fy, self.cy, 0.0],
                [0.0,     0.0,     1.0,     0.0 ],
                [0.0,     0.0,     0.0,     1.0 ]
            ]
        )

        if not use_homogenous_coordinates:
            return K_mat[:3,:3]
        return K_mat

    @property
    def distortions(self):
        return np.array(self._D)

    @staticmethod
    def from_pinhole_model(fx: float, fy: float, cx:float, cy: float, width: int, height: int):
        if fx == fy:
            camera_model_name = 'SIMPLE_PINHOLE'
            params = [fx, cx, cy]
        else:
            camera_model_name = 'PINHOLE'
            params = [fx, fy, cx, cy]

        return CameraIntrinsicts(camera_model_name,width, height, params)

    @staticmethod
    def from_opencv_model(fx: float, fy: float, cx:float, cy: float, distortions: np.array, width: int, height: int):
        if not isinstance(distortions, list):
            if len(distortions.shape) == 2:
                distortions = distortions.squeeze()
            distortions= distortions.tolist()
 
        params = [fx, fy, cx, cy]
        if len(distortions) == 4:
            camera_model_name = 'OPENCV'
            params += distortions
        elif len(distortions) == 5:
            camera_model_name = 'OPENCV5'
            params += distortions
        else:
            raise ValueError(f'Do not support opencv model with {len(distortions)} parameters')

        return CameraIntrinsicts(camera_model_name,width, height, params)

    def get_undistort_matrix(self, alpha=1.0):
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.distortions, (self.w,self.h), alpha, (self.w,self.h))
        return newcameramtx

    def _get_params_to_new_cx_cy_fx_fy(self, new_cx, new_cy, new_fx=None, new_fy=None):
        new_fx = self.fx if new_fx is None else new_fx
        new_fy = self.fy if new_fy is None else new_fy

        params = copy.deepcopy(self._params)
        camera_model_name = self.camera_model_name
        if camera_model_name == 'SIMPLE_PINHOLE':
            params[0] = new_fx
            params[1] = new_cx
            params[2] = new_cy
        elif camera_model_name == 'PINHOLE':
            params[0] = new_fx
            params[1] = new_fy
            params[2] = new_cx
            params[3] = new_cy
        elif camera_model_name == 'SIMPLE_RADIAL':
            params[0] = new_fx
            params[1] = new_cx
            params[2] = new_cy
        elif camera_model_name == 'RADIAL':
            params[0] = new_fx
            params[1] = new_cx
            params[2] = new_cy
        elif camera_model_name == 'OPENCV':
            params[0] = new_fx
            params[1] = new_fy
            params[2] = new_cx
            params[3] = new_cy
        elif camera_model_name == 'OPENCV_FISHEYE':
            params[0] = new_fx
            params[1] = new_fy
            params[2] = new_cx
            params[3] = new_cy
        elif camera_model_name == 'FULL_OPENCV':
            params[0] = new_fx
            params[1] = new_fy
            params[2] = new_cx
            params[3] = new_cy
        elif camera_model_name == 'FOV':
            params[0] = new_fx
            params[1] = new_fy
            params[2] = new_cx
            params[3] = new_cy
        else:
            raise(f'This should not happen')

        return params

    def crop_bbox(self, in_bbox, c_bbox, new_width, new_height):
        if in_bbox is None: return None

        if in_bbox.minx >= c_bbox.maxx or in_bbox.maxx <= c_bbox.minx: return None
        if in_bbox.miny >= c_bbox.maxy or in_bbox.maxy <= c_bbox.miny: return None

        b_xmin = max(in_bbox.minx - c_bbox.minx,0)
        b_xmax = min(max(in_bbox.maxx - c_bbox.minx,0), new_width)
        b_ymin = max(in_bbox.miny - c_bbox.miny,0)
        b_ymax = min(max(in_bbox.maxy - c_bbox.miny,0), new_height)
        if b_xmin == b_xmax or b_ymin == b_ymax: return None

        return edict(
            minx = b_xmin,
            maxx = b_xmax,
            miny = b_ymin,
            maxy = b_ymax
        )

    def __str__(self):
        s  = f'Camera: {self.camera_model_name}\n'
        s += f'  w,h={self.width,self.height}\n'
        s += f'  params: {self.params}\n'
        s += f'  cx,cy= ({self.cx},{self.cy})\n'
        s += f'  fx,fy= ({self.fx},{self.fy})\n'

        return s

    @property
    def params(self):
        return self._params


    def resize(self, new_size):
        """Change camera intrinsicts due to image resize --> scale of focal lenghts
        Args:
            new_size (tuple): (destination_width, destination_height)
        """
        new_width = new_size[0]
        new_height = new_size[1]
        scale_w = new_width / self.width
        scale_h = new_height / self.height

        fx  = self.fx * scale_w    # fx
        cx  = self.cx * scale_w    # cx
        fy  = self.fy * scale_h    # fy
        cy  = self.cy * scale_h    # cy

        new_params = self._get_params_to_new_cx_cy_fx_fy(cx, cy, fx, fy)

        return CameraIntrinsicts(
            camera_model_name=self.camera_model_name, 
            width=new_width, 
            height=new_height, 
            params=new_params
        )

    def crop(self, bbox, new_name=None):
        """Change camera intrinsicts due to clipping to a rectangular window --> shifting the proincipal point
        Args:
            min_crop_x (float): Minimal coordinate of clipping rectangle in x directior, in pixels
            min_crop_y ([float]): Minimal coordinate of clipping rectangle in y directior, in pixels
        """
        new_cx = self.cx -  int(round(bbox.minx))   # cx
        new_cy = self.cy - int(round(bbox.miny))   # cy

        new_width = bbox.maxx - bbox.minx
        new_height = bbox.maxy - bbox.miny

        new_params = self._get_params_to_new_cx_cy_fx_fy(new_cx, new_cy)

        return CameraIntrinsicts(
            camera_model_name=self.camera_model_name, 
            width=new_width, 
            height=new_height, 
            params=new_params
        )

    def get_optimal_new_camera_matrix(self, alpha):
        """_summary_

        Args:
            alpha (float): A number between 0 and 1, If the value is 
               alpha = 0 --> when all the pixels in the undistorted image are valid
               alpha = 1 --> when all the source image pixels are retained in the undistorted image but with many black pixels 
            new_image_size (_type_, optional): _description_. Defaults to None.
        """
        # See cvGetOptimalNewCameraMatrix in line 2714 of https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration.cpp
        # See https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html

        outer, inner = self.icv_get_rectangles()

        new_image_width = self.width
        new_image_height = self.height
   
        # Projection mapping inner rectangle to viewport
        fx0 = (new_image_width-1)/ inner.width
        fy0 = (new_image_height-1)/ inner.height
        cx0 = -fx0 * inner.x
        cy0 = -fy0 * inner.y

        # Projection mapping outer rectangle to viewport
        fx1 = (new_image_width-1)/ outer.width
        fy1 = (new_image_height-1)/ outer.height
        cx1 = -fx1 * outer.x
        cy1 = -fy1 * outer.y

        # Interpolate between the two optimal projections
        fx = fx0*(1 - alpha) + fx1*alpha
        fy = fy0*(1 - alpha) + fy1*alpha
        cx = cx0*(1 - alpha) + cx1*alpha
        cy = cy0*(1 - alpha) + cy1*alpha

        new_params = [fx,fy,cx,cy]
        return CameraIntrinsicts(
            camera_model_name='PINHOLE', 
            width=new_image_width, 
            height=new_image_height, 
            params=new_params
        )



    def project_camera_plane_to_image_plane(self, pc: np.array):
        K = self.K(use_homogenous_coordinates=False)
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        x_pix = fx*pc[0] + cx
        y_pix = fy*pc[1] + cy
        return np.array([x_pix, y_pix])

    def project_image_plane_to_camera_plane(self, pu: np.array):
        K = self.K(use_homogenous_coordinates=False)
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        
        xc = (pu[0] - cx)/fx
        yc = (pu[1] - cy)/fy
        return np.array([xc, yc])

    def undistort(self, pc_distorted: np.array):
        # see line 565 in https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
        eps =np.finfo(np.float64).eps

        kNumIterations = 100
        kMaxStepNorm = np.float32(1e-10)
        kRelStepSize = np.float32(1e-6)

        J = np.eye(2)
        x0 = pc_distorted.copy()
        x = pc_distorted.copy()
        for i in range(kNumIterations):
            step0 = np.max([eps, kRelStepSize * x[0]])
            step1 = np.max([eps, kRelStepSize * x[1]])

            dx = self.distort(x)

            dx_0b = self.distort(np.array([x[0] - step0, x[1]]))
            dx_0f = self.distort(np.array([x[0] + step0, x[1]]))
            dx_1b = self.distort(np.array([x[0]        , x[1] - step1]))
            dx_1f = self.distort(np.array([x[0]        , x[1] + step1]))
            J[0, 0] = 1 + (dx_0f[0] - dx_0b[0]) / (2 * step0)
            J[0, 1] = (dx_1f[0] - dx_1b[0]) / (2 * step1)
            J[1, 0] = (dx_0f[1] - dx_0b[1]) / (2 * step0)
            J[1, 1] = 1 + (dx_1f[1] - dx_1b[1]) / (2 * step1)
    
            step_x = np.linalg.inv(J) @ (dx - x0)
            x -= step_x

            squaren_norm = step_x[0]*step_x[0] + step_x[1]*step_x[1]
            if squaren_norm < kMaxStepNorm:
                break

        return  x   # undistorted


    def distort(self, p_cam_distorted: np.array):
        # see line 888 in https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
        camera_model_name = self.camera_model_name
        distortions = self.distortions

        if camera_model_name == 'SIMPLE_PINHOLE' or camera_model_name == 'PINHOLE':
            p_cam_undistorted =  p_cam_distorted.copy()

        if camera_model_name == 'OPENCV5':
            # See https://learnopencv.com/understanding-lens-distortion/
            k1 = distortions[0]
            k2 = distortions[1]
            p1 = distortions[2]
            p2 = distortions[3] 
            k3 = distortions[4]

            xd = p_cam_distorted[0]
            yd = p_cam_distorted[1]

            x2 = xd*xd
            y2 = yd*yd
            xy = xd*yd
            r2 = x2 + y2
            r4 = r2*r2
            r6 = r2*r4

            a = 1.0 + k1*r2  + k2*r4 + k3*r6
            xu = a*xd + 2.0*p1*xy + p2*(r2 + 2.0*x2)
            yu = a*yd + p1*(r2+2.0*y2) + 2.0*p2*xy
    
        p_cam_distorted = np.array([xu,yu])

        return p_cam_distorted

    def icv_get_rectangles(self):
        # see icvGetRectangles, line 2460 in https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration.cpp
        N = 9
        x_step = self.w / (N-1)
        y_step = self.h / (N-1)

        # Get a grid over [w,h] image in original, distorted, coordinates
        xu = []
        yu = []
        xu_left = []
        xu_right = []
        yu_bottom = [] 
        yu_top = []         
        for y in range(N):
            yp = y*y_step
            for x in range(N):
                xp = x*x_step
                ps = np.array([xp,yp])
                pc_distorted = self.project_image_plane_to_camera_plane(ps)                # from 2d pixel coordinates to 2D camera plane
                pc_undistorted = self.undistort(pc_distorted)                              # undistors with inverse lens distortions

                x_undistorted = pc_undistorted[0]
                y_undistorted = pc_undistorted[1]

                xu.append(x_undistorted)
                yu.append(y_undistorted)
                if x == 0: xu_left.append(x_undistorted)
                if x == N-1: xu_right.append(x_undistorted)
                if y == 0: yu_top.append(y_undistorted)
                if y == N-1: yu_bottom.append(y_undistorted)


        xmin_o = np.min(xu)
        xmax_o = np.max(xu)
        ymin_o = np.min(yu)
        ymax_o = np.max(yu)
        outer = edict(x=xmin_o, y=ymin_o, width=xmax_o-xmin_o, height=ymax_o-ymin_o)


        xmin_i = np.max(xu_left)
        xmax_i = np.min(xu_right)
        ymin_i = np.max(yu_top)
        ymax_i = np.min(yu_bottom)
        inner = edict(x=xmin_i, y=ymin_i, width=xmax_i-xmin_i, height=ymax_i-ymin_i)

        return outer, inner

    def get_fov(self):
        # Zeliltsky 2.60
        fovx = 2 * np.rad2deg(np.arctan2(self.width , (2 * self.fx)))
        fovy = 2 * np.rad2deg(np.arctan2(self.height , (2 * self.fy)))

        return edict(fovx=fovx, fovy=fovy)

    def to_dict(self):
        return self.as_dict()

    def as_dict(self):
        asdict = dict(
            width=self.width,
            height=self.height,
            camera_model_name=self.camera_model_name,
            params=[float(p) for p in self.params.tolist()]
        )
        return asdict

    def to_json(self, json_file):
        write_json_file(self.as_dict(), json_file)


class Camera:
    def __init__(self, 
                camera_intrinsics: CameraIntrinsicts, 
                R: np.array,
                t: np.array):
        self.camera_intrinsics = camera_intrinsics
        self._K = camera_intrinsics.K(use_homogenous_coordinates=True)
        self.se3 = SE3(R, t)

        self._P = self._K @ self.extrinsics
        self._center = - R.T @ t
        self.img = None

    def append_image(self, img):
        self.img = img

    @property
    def extrinsics(self):
        """returns the 4x4 extrinsic matrix (in homogenous coordinates)

        Returns:
            np.array: 4x4 matrix transforming points from 3D homogenous coordinates (4d vectors) into the camera coordinate system 
        """
        return self.se3.as_matrix()

    @property
    def K(self):
        return self._K

    @property 
    def w(self):
        return self.camera_intrinsics.w

    @property 
    def width(self):
        return self.camera_intrinsics.w

    @property 
    def h(self):
        return self.camera_intrinsics.h

    @property 
    def height(self):
        return self.camera_intrinsics.h

    @property
    def center(self):
        return self._center

    @staticmethod
    def from_rotvec_trans(camera_intrinsics, rotation_vector, translation_vector):
        if len(rotation_vector.shape) == 2:
            rotation_vector = rotation_vector.squeeze()

        if len(translation_vector.shape) == 2:
            translation_vector = translation_vector.squeeze()

        R = SO3.from_rotvec(rotation_vector).as_matrix()

        return Camera(camera_intrinsics, R, translation_vector)


    def project_3d(self, pw: np.array):
        """project a 3D vector by a pure pinhole projection
           See Szeliski, 2.1

        Args:
            pw (np.array): 3D vector in homogenous coordinates, represented as 4D array

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            np.array: _description_
        """
        if len(pw.shape) == 1:
            if pw.shape[0] != 4:
                raise ValueError(f'Expecting 3D point in homogenous coordinates')
        elif len(pw.shape) == 2:
            if pw.shape[1] != 4:
                raise ValueError(f'Expecting 3D points in homogenous coordinates')

        p_out = self._P @ pw
        p_out = p_out / p_out[2]   # devide by Z coordinate

        p2d = p_out[:2]
        disparity = p_out[3]
        return p2d, disparity

    def project_point(self, pw: np.array):
        """project a 3D vector by a pure pinhole projection
           See Szeliski, 2.1

        Args:
            pw (np.array): 3D vector in homogenous coordinates, represented as 4D array

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            np.array: _description_
        """
        if len(pw.shape) == 1:
            if pw.shape[0] != 4:
                raise ValueError(f'Expecting 3D point in homogenous coordinates')
        elif len(pw.shape) == 2:
            if pw.shape[1] != 4:
                raise ValueError(f'Expecting 3D points in homogenous coordinates')

        # see line 854 in https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h

        pc_undistorted, disparity = self.project_3d_to_camera_plane(pw)       # from3d world coordinates to 2D camera plane cordinates
        pc_distorted = self.camera_intrinsics.distort(pc_undistorted)                           # distort point using distortion coefficients and model
        pu = self.camera_intrinsics.project_camera_plane_to_image_plane(pc_distorted)           # transform to 2D pixel coordinates

        return pu, disparity

    def reproject_point(self, pu: np.array, disparity=None):
        """project a 3D vector by a pure pinhole projection
           See Szeliski, 2.1

        Args:
            pw (np.array): 3D vector in homogenous coordinates, represented as 4D array

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            np.array: _description_
        """
        pc_distorted = self.camera_intrinsics.project_image_plane_to_camera_plane(pu)                # from 2d pixel coordinates to 2D camera plane
        pc_undistorted = self.camera_intrinsics.undistort(pc_distorted)                              # undistors with inverse lens distortions
        p3d = self.project_camera_plane_to_3d(pc_undistorted, disparity)           # from 2D camera plane to 3D point, usign the disparity (inverse depth)

        return p3d

    def project_3d_to_camera_plane(self, pw: np.array):
        """project a 3D vector
           See Szeliski, 2.1

        Args:
            pw (np.array): 4D vector of a world 3D point 
        """
        pc = self.extrinsics @ pw

        disparity = 1/pc[2]
        xc = pc[0]*disparity
        yc = pc[1]*disparity

        return np.array([xc,yc]), disparity

    def project_camera_plane_to_3d(self, pc: np.array, disparity=None):
        """project a 3D vector
           See Szeliski, 2.1

        Args:
            pw (np.array): 4D vector of a world 3D point 
        """

        pc_3d = np.array([pc[0]/disparity, pc[1]/disparity,1./disparity,1])

        p_3d = np.linalg.inv(self.extrinsics) @ pc_3d

        return p_3d

    def undistort_camera(self):
        # UndistortImage: See line 937 of https://github.com/colmap/colmap/blob/dev/src/base/undistortion.cc
        #   calls to UndistortCamera: (line 752) returning undistored camera
        #   clalls WarpImageBetweenCameras
        pass


    def plot_camera_axis(self, vis_scale):
        """ plot the axis of the camera

        Args:
            vis_scale (_type_): _description_
        """
        import ipyvolume as ipv

        x, y, z = self.center[0],  self.center[1],  self.center[2]
        inv_extrinsic = np.linalg.inv(self.extrinsics_matrix())

        x_arrow = inv_extrinsic @ (1*vis_scale, 0, 0,  1)
        y_arrow = inv_extrinsic @ (0, 1*vis_scale, 0,  1)
        z_arrow = inv_extrinsic @ (0, 0, 1*vis_scale,  1)
        
        ipv.plot([x, x_arrow[0]], [y, x_arrow[1]], [z, x_arrow[2]], color='red')
        ipv.plot([x, y_arrow[0]], [y, y_arrow[1]], [z, y_arrow[2]], color='blue')
        ipv.plot([x, z_arrow[0]], [y, z_arrow[1]], [z, z_arrow[2]], color='green')
        
        cam_sphere_size = 1
        ipv.scatter(np.array([x]), np.array([y]), np.array([z]), size=cam_sphere_size, marker="sphere", color='blue')

        return ipv


    def plot(self, vis_scale, show_image: bool = False):
        import ipyvolume as ipv

        """ Plots the wireframe of a camera's viewport. """
        x, y, z = self.center[0],  self.center[1],  self.center[2]

        extrinsic = self.extrinsics_matrix()
        inv_extrinsic = np.linalg.inv(extrinsic) 

        camera_aspect_ratio = self.w / self.h
        vis_cam_height = 1 
        vis_cam_width = vis_cam_height * camera_aspect_ratio
        wire_frame_depth = 1.2

        # determine the camera plane that is ib z=wire_frame_depth, in homogenous coordinates with a scale facti=or to allow viewing
        # Those coordinates are in the camera plane
        x0 = -vis_cam_width/2
        x1 = vis_cam_width/2
        y0 = -vis_cam_height/2
        y1 = vis_cam_height/2
        p0 = np.array((x0, y0, wire_frame_depth, 1/vis_scale)) * vis_scale
        p1 = np.array((x1, y0, wire_frame_depth, 1/vis_scale)) * vis_scale
        p2 = np.array((x0, y1, wire_frame_depth, 1/vis_scale)) * vis_scale
        p3 = np.array((x1, y1, wire_frame_depth, 1/vis_scale)) * vis_scale

       # Get left/right top/bottom wireframe coordinates
        # Use the inverse of the camera's extrinsic matrix to convert 
        # coordinates relative to the camera to world coordinates.
        p0w = inv_extrinsic @ p0
        p1w = inv_extrinsic @ p1
        p2w = inv_extrinsic @ p2
        p3w = inv_extrinsic @ p3

        # Connect camera projective center to wireframe extremities
        ipv.plot([x, p0w[0]], [y, p0w[1]], [z, p0w[2]], color='blue')
        ipv.plot([x, p1w[0]], [y, p1w[1]], [z, p1w[2]], color='blue')
        ipv.plot([x, p2w[0]], [y, p2w[1]], [z, p2w[2]], color='blue')
        ipv.plot([x, p3w[0]], [y, p3w[1]], [z, p3w[2]], color='blue')
        
        # Connect wireframe corners with a rectangle
        ipv.plot([p0w[0], p1w[0]], [p0w[1], p1w[1]], [p0w[2], p1w[2]], color='red')   
        ipv.plot([p1w[0], p3w[0]], [p1w[1], p3w[1]], [p1w[2], p3w[2]], color='red')
        ipv.plot([p3w[0], p2w[0]], [p3w[1], p2w[1]], [p3w[2], p2w[2]], color='red')
        ipv.plot([p2w[0], p0w[0]], [p2w[1], p0w[1]], [p2w[2], p0w[2]], color='red')

        cam_sphere_size = 1
        ipv.scatter(np.array([x]), np.array([y]), np.array([z]), size=cam_sphere_size, marker="sphere", color='blue')

        if show_image and self.img is not None:
            xx, yy = np.meshgrid(np.linspace(x0*vis_scale,  x1*vis_scale,  self.w), 
                                 np.linspace(y0*vis_scale, y1*vis_scale, self.h))
            zz = np.ones_like(yy) * wire_frame_depth * vis_scale
            coords = np.stack([xx, yy, zz, np.ones_like(zz)]) 
            coords = coords.reshape(4, -1) 
    
            # Convert canera relative coordinates to world relative coordinates
            coords = inv_extrinsic @ coords
            xx, yy, zz, ones = coords.reshape(4, self.h, self.w) 

            img = self.img.copy() / 255

            ipv.plot_surface(xx, yy, zz, color=img)

        return ipv

class Cameras:
    def __init__(self, cameras_list: list):
        self.cameras_list = cameras_list


    def plot(self, vis_scale, show_image: bool = False):
        import ipyvolume as ipv
        from slib.plot_3d import init_3d_plot

        xlist = [c.center[0] for c in self.cameras_list]
        ylist = [c.center[1] for c in self.cameras_list]
        zlist = [c.center[2] for c in self.cameras_list]
        xmin = np.min(xlist) - 2.0
        xmax = np.max(xlist) + 2.0
        ymin = np.min(ylist) - 2.0
        ymax = np.max(ylist) + 2.0
        zmin = np.min(zlist) - 2.0
        zmax = np.max(zlist) + 2.0
        fig = init_3d_plot(xmin, ymin, xmax, ymax, zmin, zmax)

        for c in self.cameras_list:
            ipv = c.plot(vis_scale, show_image)

        return ipv

    def get_new_camera(self):
        undistorted_camera = CameraIntrinsicts.from_pinhole_model(self.fx, self.fy, self.cx, self.cy, width=self.width, height=self.height)


def init_undistort_rectify_map(camera_inrinsics: CameraIntrinsicts, new_camera_intrinsic: CameraIntrinsicts):
    # See https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    # code - line 64 in https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp
    mapx = np.zeros((new_camera_intrinsic.h, new_camera_intrinsic.w))
    mapy = np.zeros((new_camera_intrinsic.h, new_camera_intrinsic.w))
    for u in range(new_camera_intrinsic.w):
        for v in range(new_camera_intrinsic.h):
            pi = np.array([float(u), float(v)])
            p_undistorted = new_camera_intrinsic.project_image_plane_to_camera_plane(pi)

            p_distorted = camera_inrinsics.distort(p_undistorted)
            pix = camera_inrinsics.project_camera_plane_to_image_plane(p_distorted)
            mapx[v,u] = pix[0]
            mapy[v,u] = pix[1]

    return mapx, mapy

def get_test_camera():
    import os
    import sys
    from pathlib import Path
    import cv2

    root_dir = Path(os.getcwd()).parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0,str(root_dir))
    else:
        print('PATH O.K.')

    img_dir = root_dir /'images' / 'calibration'
    img_file = img_dir / 'left03.jpg'
    img = cv2.imread(str(img_file))
    h,  w = img.shape[:2]

    # data rteken from OpenCv tutorial : https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html
    #  and data                        : https://github.com/opencv/opencv/tree/master/samples/data (left01.jpg - left14.j)
    distortions = np.array(
        [
        [-2.6637260909660682e-01], 
        [-3.8588898922304653e-02], 
        [1.7831947042852964e-03], 
        [-2.8122100441115472e-04], 
        [2.3839153080878486e-01]
        ]
    )

    mtx = np.array(
        [
            [5.3591573396163199e+02, 0.,                     3.4228315473308373e+02],
            [0.,                     5.3591573396163199e+02, 2.3557082909788173e+02],
            [0.,                     0.,                     1.]
        ]
    )

    fx = mtx[0,0]
    fy = mtx[1,1]
    cx = mtx[0,2]
    cy = mtx[1,2]
    ci = CameraIntrinsicts.from_opencv_model(fx,fy,cx,cy,distortions,w, h)

    return mtx, distortions, ci, img


# tests
# See also https://github.com/maxcrous/multiview_notebooks/blob/main/3_stereo_geometry.ipynb
#   https://learnopencv.com/understanding-lens-distortion/
#   https://github.com/mint-lab/3dv_tutorial
#   invitaiton to 3D: https://docs.google.com/viewer?url=https%3A%2F%2Fwww.eecis.udel.edu%2F~cer%2Farv%2Freadings%2Fold_mkss.pdf&pdf=true
# https://github.com/subeeshvasu/Awesome-Image-Distortion-Correction
#
# OpenCv remap vs pytorch grod_map
#    https://discuss.pytorch.org/t/cv2-remap-in-pytorch/99354/2
#    https://gist.github.com/sftekin/d5ddf65e54e3dfe89a308c624e62280b
#    https://www.programcreek.com/python/example/104458/torch.nn.functional.grid_sample
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
#    https://arxiv.org/abs/1506.02025
#      https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
if __name__ == '__main__':

    mtx, distortions, ci, img =  get_test_camera()
    h,  w = img.shape[:2]
    alpha = 0.0

    no_rot = np.array([[0.0], [0.0], [0.0]])
    no_trans = np.array([[0.0], [0.0], [0.0]])
    K = ci.K()[:3,:3]

    # Test that pixel to camere plane and distortiom/undistortion works as expected
    #   undistortPoints 
    print('Testing compatibility with OpenCv undistort, project_points')
    print('Testing that undistort is the reverse of distort')

    for xi in [0, 50, 100, 200, 300, 400, 640]:
        for yi in [0,100,200,400, 480]:
            pi  = np.array([float(xi), float(yi)])              # original (distorted) coordimnates

            pc = ci.project_image_plane_to_camera_plane(pi)     # to camera plane
            p_undistorted = ci.undistort(pc)                    # undistort
 
            # Test competability with OpenCv undistort 
            cv_undistorted =  cv2.undistortPoints(pi, K, distortions).squeeze()
            err_to_cv_undistort = np.linalg.norm(p_undistorted-cv_undistorted)
            if err_to_cv_undistort > 1e-4:
                print(p_undistorted, cv_undistorted, err_to_cv_undistort)

            # revert previous transformation
            pd = ci.distort(p_undistorted)                      # distort
            pi1 = ci.project_camera_plane_to_image_plane(pd)    # to image 

            # compare to OpenCv implementation
            pw = np.array([p_undistorted[0], p_undistorted[1], 1.0, 1.0])   # lift to wrld coordinates with disparty (z) 1, identity rigid tramsformation
            pp, _ =  cv2.projectPoints(pw[:3],                              # project to image
                                        no_rot,
                                        no_trans,
                                        mtx,
                                        distortions)

    
            dist_to_opencv = np.linalg.norm(pp-pi1)          # check that OpenCV projection from camera coordinate system is teh same as self
            if dist_to_opencv > 1e-5:
                print(f'error between opencv projection and self: {dist_to_opencv}')

            dist_undist_err = np.linalg.norm(pc-pd)          # check that undistort is the inverse of distort
            pixel_err = np.linalg.norm(pi-pi1)               # check the whole image --> camera plane and back 
            if dist_undist_err > 1e-4 or pixel_err > 0.01:
                print(f'dist err : {dist_undist_err}. PIxel errors {pixel_err}')

    print('Comparing impelementation of get_optimal_new_camera_matrix to OpenCv getOptimalNewCameraMatrix')
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        new_matrix_cv, roi = cv2.getOptimalNewCameraMatrix(
            mtx,
            distortions,
            (w,h),
            alpha=alpha
        )
        # print(new_matrix)
        # print(roi)

        new_camera = ci.get_optimal_new_camera_matrix(alpha=alpha)
        # print(new_camera)

        err_new_matrix = np.linalg.norm(new_camera.K()[:3,:3]-new_matrix_cv)
        if err_new_matrix > 0.02:
            print(err_new_matrix)

    # Test equivalence of undistortion maps to OpenCv
    map_x, map_y = init_undistort_rectify_map(ci, new_camera)

    opencv_mapx, opencv_mapy = cv2.initUndistortRectifyMap(
        mtx, 
        distortions,
        None,
        new_camera.K()[:3,:3],
        (new_camera.w,new_camera.h),
        cv2.CV_32FC1
    )

    ex = np.linalg.norm(map_x-opencv_mapx)
    ey = np.linalg.norm(map_y-opencv_mapy)
    if ex > 0.01 or ey > 0.01:
        print(map_x[0,0], opencv_mapx[0,0] )
        print(map_y[0,0], opencv_mapy[0,0] )
        print(f'err-x={ex}, err-y={ey}')