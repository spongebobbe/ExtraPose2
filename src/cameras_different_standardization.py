
"""Utilities to deal with the cameras of human3.6m"""
#from __future__ import division

from xml.dom import minidom
import h5py
import numpy as np
import matplotlib.pyplot as plt
import src.data_utils
import src.viz

def project_point_radial( P, R, T, f, c, k, p ):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal 
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2

  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) )
  tan = 2*p[0]*XX[1,:] + 2*p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2

def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.dot( P.T - T ) # rotate and translate

  return X_cam.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.T.dot( P.T ) + T # rotate and translate

  return X_cam.T

'''
def load_camera_params( hf, path ):
  """Load h36m camera parameters

  Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  R = hf[ path.format('R') ][:]
  R = R.T

  T = hf[ path.format('T') ][:]
  f = hf[ path.format('f') ][:]
  c = hf[ path.format('c') ][:]
  k = hf[ path.format('k') ][:]
  p = hf[ path.format('p') ][:]

  name = hf[ path.format('Name') ][:]
  name = "".join( [chr(item) for item in name] )

  return R, T, f, c, k, p, name

def load_cameras( bpath , subjects=[1,5,6,7,8,9,11] ):
  """Loads the cameras of h36m

  Args
    bpath: path to hdf5 file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = {}

  with h5py.File(bpath,'r') as hf:
    for s in subjects:
      for c in range(4): # There are 4 cameras in human3.6m
        rcams[(s, c+1)] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )

  return rcams
'''

CAMERA_ID_TO_NAME = {
  1: "54138969",
  2: "55011271",
  3: "58860488",
  4: "60457274",
}


def load_camera_params(w0, subject, camera):
  """Load h36m camera parameters
  Args
    w0: 300-long array read from XML metadata
    subect: int subject id
    camera: int camera id
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
  """

  # Get the 15 numbers for this subject and camera
  w1 = np.zeros(15)
  start = 6 * ((camera-1)*11 + (subject-1))
  w1[:6] = w0[start:start+6]
  w1[6:] = w0[(265+(camera-1)*9 - 1): (264+camera*9)]

  def rotationMatrix(r):    
    R1, R2, R3 = [np.zeros((3, 3)) for _ in range(3)]

    # [1 0 0; 0 cos(obj.Params(1)) -sin(obj.Params(1)); 0 sin(obj.Params(1)) cos(obj.Params(1))]
    R1[0:] = [1, 0, 0]
    R1[1:] = [0, np.cos(r[0]), -np.sin(r[0])]
    R1[2:] = [0, np.sin(r[0]),  np.cos(r[0])]

    # [cos(obj.Params(2)) 0 sin(obj.Params(2)); 0 1 0; -sin(obj.Params(2)) 0 cos(obj.Params(2))]
    R2[0:] = [ np.cos(r[1]), 0, np.sin(r[1])]
    R2[1:] = [0, 1, 0]
    R2[2:] = [-np.sin(r[1]), 0, np.cos(r[1])]

    # [cos(obj.Params(3)) -sin(obj.Params(3)) 0; sin(obj.Params(3)) cos(obj.Params(3)) 0; 0 0 1];%
    R3[0:] = [np.cos(r[2]), -np.sin(r[2]), 0]
    R3[1:] = [np.sin(r[2]),  np.cos(r[2]), 0]
    R3[2:] = [0, 0, 1]

    return (R1.dot(R2).dot(R3))
    
  R = rotationMatrix(w1)
  T = w1[3:6][:, np.newaxis]
  f = w1[6:8][:, np.newaxis]
  c = w1[8:10][:, np.newaxis]
  k = w1[10:13][:, np.newaxis]
  p = w1[13:15][:, np.newaxis]
  name = CAMERA_ID_TO_NAME[camera]

  return R, T, f, c, k, p, name


def load_cameras(bpath, subjects=[1,5,6,7,8,9,11]):
  """Loads the cameras of h36m
  Args
    bpath: path to xml file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = {}

  xmldoc = minidom.parse(bpath)
  string_of_numbers = xmldoc.getElementsByTagName('w0')[0].firstChild.data[1:-1]

  # Parse into floats
  w0 = np.array(list(map(float, string_of_numbers.split(" "))))

  assert len(w0) == 300

  for s in subjects:
    for c in range(4): # There are 4 cameras in human3.6m
      rcams[(s, c+1)] = load_camera_params(w0, s, c+1)

  return rcams
  
def normalize_camera_params(h,w,rcams, subjects=[1,5,6,7,8,9,11]):
  # normalize camera params using the method of 'Estimation of 3D Human Pose Using Prior Knowledge'
  #rcams norm are the same across the different subjects but I keep the same structure of rcmas
  rcams_norm = {}
  for s in subjects:
    for cam in range(4): # There are 4 cameras in human3.6m
      _, _, f, c, _, _, _ = rcams[ (s, cam+1) ]

      f_norm = np.mean(f) #there are 2 values of focal length. issue https://github.com/una-dinosauria/3d-pose-baseline/issues/167
      f_norm = (2*f_norm)/w
      focus = [2*(c[0]/w) -1, 2*(c[1]/w) - h/w] 

      rcams_norm[(s, cam+1)] = [focus,f_norm]
      
  rcams_norm = standardize_rcams_norm(rcams_norm)

  return rcams_norm

def standardize_rcams_norm(rcams_norm1, subjects=[1,5,6,7,8,9,11]):
  rcams_norm = {}
  #4 cams with same params across subj. I calculate mu and stdev on subj 1
  l_cams = list(rcams_norm1.values())
  mu_focus = np.mean([l_cams[0][0], l_cams[1][0], l_cams[2][0], l_cams[3][0]], axis = 0)
  stddev_focus = np.std([l_cams[0][0], l_cams[1][0], l_cams[2][0], l_cams[3][0]], axis = 0) 
  mu_f = np.mean([l_cams[0][1], l_cams[1][1], l_cams[2][1], l_cams[3][1]], axis = 0)
  stddev_f = np.std([l_cams[0][1], l_cams[1][1], l_cams[2][1], l_cams[3][1]], axis = 0)
  
  for s in subjects:
    for cam in range(4): # There are 4 cameras in human3.6m
      focus, f_norm = rcams_norm1[ (s, cam+1) ]
      rcams_norm[(s, cam+1)] =  [np.divide( (focus - mu_focus), stddev_focus ), np.divide( (f_norm - mu_f), stddev_f )]
  return rcams_norm

