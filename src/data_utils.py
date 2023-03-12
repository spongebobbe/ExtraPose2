
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
from statistics import stdev
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import src.cameras as cameras
import src.viz
import h5py
import glob
import copy
import cdflib
from numpy import linalg as LA

# Human3.6m IDs for training and testing


# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax' 
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'


CAMERA_NAME_TO_ID = {
  "54138969": 1,
  "55011271": 2,
  "58860488": 3,
  "60457274": 4,
}



def calculate_length_notNormalized(pose3d, subjects, action):
  bone_lengths = {} # for each subject, action and camera I calculate the bone lengths across frames
  pose3d_tmp = {}
  for key3d in pose3d.keys():
      (subj, b, fname) = key3d

      
      #(1, 'Directions', 'Directions 1..54138969.h5') => (1, 'Directions', '1')
      #(1, 'Directions', 'Directions ..54138969.h5') => (1, 'Directions', '1')
      # the first is overwritten by the second so in bone_lengths I have only one trial for each action instead of 2
      
      #find the correct camera 
      camera_name = fname.split('..')[-1].split('.')[0]
      camera_id =  CAMERA_NAME_TO_ID[camera_name]

      data_size = pose3d[key3d].shape[0]

      pose3d_tmp[key3d] = pose3d[key3d].reshape([data_size,-1,3])
    

      no_Neck = ([0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27])  
      pose3d_tmp2 = pose3d_tmp[key3d][:,no_Neck,:]

      parent = [0,1,2,0,4,5,0,7,8,8,10,11,8,13,14] 

      dists = pose3d_tmp2[:,1:,:] - pose3d_tmp2[:,parent,:] 

      boneLengths = LA.norm(dists, axis=2)
      #ax = plt.axes(projection='3d');ax.scatter3D(pose3d_tmp[0,:,0], pose3d_tmp[0,:,1], pose3d_tmp[0,:,2])
      bone_lengths[(subj,b,camera_id)] = boneLengths
    
  bone_lengths_final = calculate_mean_bone_lengths(bone_lengths, subjects, action)
  return bone_lengths_final

def calculate_mean_bone_lengths(bone_lengths, subjects, action):
  bone_lengths_final = {} #for eac subj and cam I calculate the mea bone lengths for action 'discussions'
  #the bone lengths are equal across different cameras but i keep this structure for convenience
  for s in subjects:
    for cam in range(4): # There are 4 cameras in human3.6m
      lengths = bone_lengths[ (s,action, cam+1) ]

      bone_lengths_final[ (s, cam+1) ] = np.mean(lengths,axis = 0)
    
  return bone_lengths_final

def standardize_bone_lengths(mu, stddev, bone_lengths_train, bone_lengths_test):
  
  # #calculate mean and standard dev
  # bone_lengths_tmp = list(bone_lengths_train.values())
  # bone_lengths_tmp.extend(list(bone_lengths_test.values()))
  # bone_lengths_tmp = np.array(bone_lengths_tmp)
  # mu = np.mean(bone_lengths_tmp, axis = 0)
  # #mu = np.tile(mu, (bone_lengths_tmp.shape[0], 1))
  # stddev = np.std(bone_lengths_tmp, axis = 0)
  # #stddev = np.tile(stddev, (bone_lengths_tmp.shape[0], 1))

  #standardize
  bone_lengths_train_norm = {}
  for key in bone_lengths_train.keys():  
    bone_lengths_train_norm[ key ] = np.divide( (bone_lengths_train[key] - mu), stddev )

  bone_lengths_test_norm = {}
  for key in bone_lengths_test.keys():  
    bone_lengths_test_norm[ key ] = np.divide( (bone_lengths_test[key] - mu), stddev )

  return bone_lengths_train_norm, bone_lengths_test_norm

def bone_normalization_stats(bone_lengths_train):
  # #calculate mean and standard dev
  bone_lengths_tmp = list(bone_lengths_train.values())
  bone_lengths_tmp = np.array(bone_lengths_tmp)
  mu = np.mean(bone_lengths_tmp, axis = 0)
  stddev = np.std(bone_lengths_tmp, axis = 0)
  return mu, stddev


def standardize_bone_lengths_old(bone_lengths):
  
  #calculate mean and standard dev
  #calculate mean and standard dev
  bone_lengths_tmp = list(bone_lengths.values())
  bone_lengths_tmp = np.array(bone_lengths_tmp)
  mu = np.mean(bone_lengths_tmp, axis = 0)
  #mu = np.tile(mu, (bone_lengths_tmp.shape[0], 1))
  stddev = np.std(bone_lengths_tmp, axis = 0)
  #stddev = np.tile(stddev, (bone_lengths_tmp.shape[0], 1))

  #standardize
  bone_lengths_norm = {}
  for key in bone_lengths.keys():  
    bone_lengths_norm[ key ] = np.divide( (bone_lengths[key] - mu), stddev )
  return bone_lengths_norm


def standardize_bone_lengths2(bone_lengths):
  
  #standardize per raw (in this way ratio across bone lengths is preserved)
  bone_lengths_norm = {}
  for key in bone_lengths.keys():  
    mu = np.mean(bone_lengths[key]),
    stddev = np.std(bone_lengths[key])
    bone_lengths_norm[ key ] = np.divide( (bone_lengths[key] - mu), stddev )
  return bone_lengths_norm

def load_data( bpath, subjects, actions, dim=3 ):
  """
  Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

  Args
    bpath: String. Path where to load the data from
    subjects: List of integers. Subjects whose data will be loaded
    actions: List of strings. The actions to load
    dim: Integer={2,3}. Load 2 or 3-dimensional data
  Returns:
    data: Dictionary with keys k=(subject, action, seqname)
      values v=(nx(32*2) matrix of 2d ground truth)
      There will be 2 entries per subject/action if loading 3d data
      There will be 8 entries per subject/action if loading 2d data
  """

  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data = {}

  for subj in subjects:
    for action in actions:

      # print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join(  bpath, 'S{0}/'.format(subj), 'MyPoseFeatures/{0}D_Positions/'.format(dim), '{0}*.cdf'.format(action) )
#       dpath =  bpath + 'S{0}'.format(subj) + '/MyPoseFeatures/{0}D_positions'.format(dim) + '/{0}*.cdf'.format(action) 
      # print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          # print("no load sittingDown when the action is sitting")

          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          # print( fname )
          loaded_seqs = loaded_seqs + 1

          cdf_file = cdflib.CDF(fname)
          poses = cdf_file.varget("Pose").squeeze()
          cdf_file.close()

          data[ (subj, action, seqname) ] = poses

      if dim == 2:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format( loaded_seqs )
      else:
        assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format( loaded_seqs )

  return data


def load_stacked_hourglass(data_dir, subjects, actions):
  """
  Load 2d detections from disk, and put it in an easy-to-acess dictionary.

  Args
    data_dir: string. Directory where to load the data from,
    subjects: list of integers. Subjects whose data will be loaded.
    actions: list of strings. The actions to load.
  Returns
    data: dictionary with keys k=(subject, action, seqname)
          values v=(nx(32*2) matrix of 2d stacked hourglass detections)
          There will be 2 entries per subject/action if loading 3d data
          There will be 8 entries per subject/action if loading 2d data
  """
  # Permutation that goes from SH detections to H36M ordering.
  SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
  assert np.all( SH_TO_GT_PERM == np.array([6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]) )

  data = {}

  for subj in subjects:
    for action in actions:

      print('Reading subject {0}, action {1}'.format(subj, action))

      dpath = os.path.join( data_dir, 'S{0}'.format(subj), 'StackedHourglass/{0}*.h5'.format(action) )
      print( dpath )

      fnames = glob.glob( dpath )

      loaded_seqs = 0
      for fname in fnames:
        seqname = os.path.basename( fname )
        seqname = seqname.replace('_',' ')

        # This rule makes sure SittingDown is not loaded when Sitting is requested
        if action == "Sitting" and seqname.startswith( "SittingDown" ):
          continue

        # This rule makes sure that WalkDog and WalkTogeter are not loaded when
        # Walking is requested.
        if seqname.startswith( action ):
          print( fname )
          loaded_seqs = loaded_seqs + 1

          # Load the poses from the .h5 file
          with h5py.File( fname, 'r' ) as h5f:
            poses = h5f['poses'][:]

            # Permute the loaded data to make it compatible with H36M
            poses = poses[:,SH_TO_GT_PERM,:]

            # Reshape into n x (32*2) matrix
            poses = np.reshape(poses,[poses.shape[0], -1])
            poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])

            
            dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
            dim_to_use_y = dim_to_use_x+1

            dim_to_use = np.zeros(len(SH_NAMES)*2,dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:,dim_to_use] = poses
            seqname = seqname+'-sh'
            data[ (subj, action, seqname) ] = poses_final

      # Make sure we loaded 8 sequences
      if (subj == 11 and action == 'Directions'): # <-- this video is damaged
        assert loaded_seqs == 7, "Expecting 7 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )
      else:
        assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead. S:{1} {2}".format(loaded_seqs, subj, action )

  return data


def normalization_stats(complete_data, dim, predict_14=False ):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={2,3} dimensionality of the data
    predict_14. boolean. Whether to use only 14 joints
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions not used in the model
    dimensions_to_use: list of dimensions used in the model
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_mean = np.mean(complete_data, axis=0)
  data_std  =  np.std(complete_data, axis=0)

  # Encodes which 17 (or 14) 2d-3d pairs we are predicting
  dimensions_to_ignore = []
  #THE PROBLEM IS THAT FOR 2D ELIMINATES NECK/NOSE WHILE FOR 3D ELIMINATES THE HIP (0)
  if dim == 2: 
    #actually they are now saying that the joint to eliminate is not neck/nose but spine. https://github.com/una-dinosauria/3d-pose-baseline/issues/185
    #but it's not a problem when using 2d projection.
    #dimensions_to_use    = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
    dimensions_to_use    = np.where(np.array([x != '' and x != 'Spine' for x in H36M_NAMES]))[0]
    dimensions_to_use    = np.sort( np.hstack( (dimensions_to_use*2, dimensions_to_use*2+1))) #np.hstack appends as columns
    #if I want to eliminate marker 2 (because is empty= '')I need to eliminate its 3 coordinates from the data. 
    # each row of the data is Xmarker1, Ymarker1, Zmarker1, ..., Xmarker32, Ymarker32, Zmarker32 
    # so i need to delete  index 3, 4, 5  
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*2), dimensions_to_use )
  else: # dim == 3
    dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
    dimensions_to_use = np.delete( dimensions_to_use, [0,7,9] if predict_14 else 0 )

    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3,
                                             dimensions_to_use*3+1,
                                             dimensions_to_use*3+2)))
    dimensions_to_ignore = np.delete( np.arange(len(H36M_NAMES)*3), dimensions_to_use )

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use





def transform_world_to_camera(poses_set, cams, ncams=4 ):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted( poses_set.keys() ):

      subj, action, seqname = t3dk
      t3d_world = poses_set[ t3dk ]

      for c in range( ncams ):
        R, T, f, _, k, p, name = cams[ (subj, c+1) ]
        camera_coord = cameras.world_to_camera_frame( np.reshape(t3d_world, [-1, 3]), R, T)
        camera_coord = np.reshape( camera_coord, [-1, len(H36M_NAMES)*3] )

        sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
        t3d_camera[ (subj, action, sname) ] = camera_coord

    return t3d_camera


def normalize_data(data, data_mean, data_std, dim_to_use ):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """
  data_out = {}

  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev )

  return data_out

def select_joints(data,dim_to_use ):
  data_out = {}
  for key in data.keys():
    data_out[ key ] = data[ key ][ :, dim_to_use ]    
  return data_out

def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_ignore):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = np.array([dim for dim in range(D)
                                if dim not in dimensions_to_ignore])

  orig_data[:, dimensions_to_use] = normalized_data

  #in 3d data no hip but neck nose. origin data is initialized as zero, 
  #then assign to the correct position the normalized data. In this way the hip is added back.
  #the other unused joints are also 000

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat
  return orig_data

def deselect_joints(data, data_mean, dimensions_to_ignore):
  
  T = data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = np.array([dim for dim in range(D)
                                if dim not in dimensions_to_ignore])

  orig_data[:, dimensions_to_use] = data

  return orig_data

def define_actions( action, actions):
  """
  Given an action string, returns a list of corresponding actions.

  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """


  if action == "All" or action == "all":
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def project_to_cameras( poses_set, cams, ncams=4 ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of cameras per subject
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      #t3d is in world coordinates
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )

      pts2d = np.reshape( pts2d, [-1, len(H36M_NAMES)*2] )
      sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
      t2d[ (subj, a, sname) ] = pts2d

  return t2d


def read_2d_predictions( actions, data_dir , TRAIN_SUBJECTS, TEST_SUBJECTS):
  """
  Loads 2d data from precomputed Stacked Hourglass detections

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
  Returns
    train_set: dictionary with loaded 2d stacked hourglass detections for training
    test_set: dictionary with loaded 2d stacked hourglass detections for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions)
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions)

  complete_train = copy.deepcopy( np.vstack(list(train_set.values()) ))
  data_mean, data_std,  dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def create_2d_data( actions, data_dir, rcams, TRAIN_SUBJECTS, TEST_SUBJECTS ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters. Also normalizes the 2d poses

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    rcams: dictionary with camera parameters
  Returns
    train_set: dictionary with projected 2d poses for training
    test_set: dictionary with projected 2d poses for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )

  train_set = project_to_cameras( train_set, rcams )
  test_set  = project_to_cameras( test_set, rcams )

  '''
  import src.viz_new as viz
  p2d = train_set[(1, 'Directions', 'Directions 1..55011271.h5')][0,:]
  ax1 = plt.axes();viz.show2Dpose( p2d, ax1)
  '''

  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack( list(train_set.values()) ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  '''
  import src.viz_new as viz
  p2d = train_set[(1, 'Directions', 'Directions 1..55011271.h5')][0,:]
  ax1 = plt.axes();viz.show2D_norm_pose( p2d, ax1)
  '''

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

"""
def create_2d_data_L( actions, data_dir, rcams ):
  
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters. Also normalizes the 2d poses

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    rcams: dictionary with camera parameters
  Returns
    train_set: dictionary with projected 2d poses for training
    test_set: dictionary with projected 2d poses for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  

  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )

  train_set = project_to_cameras( train_set, rcams )
  test_set  = project_to_cameras( test_set, rcams )

  train_set = add_noise(train_set, 20)

  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack(list( train_set.values()) ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  '''
  import src.viz_new as viz
  p2d = train_set[(1, 'Directions', 'Directions 1..54138969.h5')][0,:]
  ax1 = plt.axes();viz.show2Dpose( p2d, ax1)
  '''

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use
"""
def add_noise(train_set, sigma): 
  # to add noise to 2d inputs, used in create_2d_data_L
  for k in train_set.keys():
    frames = train_set[k]
    f_shape = frames.shape
    noise = np.random.normal(0,sigma , f_shape) # 64 columns of noise
    train_set[k] = train_set[k] + noise
  return train_set

def read_3d_data( actions, data_dir, camera_frame, rcams,TRAIN_SUBJECTS, TEST_SUBJECTS, predict_14=False, flag_bone_lengths = False, normalize_target = True ):
  """
  Loads 3d poses, zero-centres and normalizes them

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    camera_frame: boolean. Whether to convert the data to camera coordinates
    rcams: dictionary with camera parameters
    predict_14: boolean. Whether to predict only 14 joints
  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
    train_root_positions: dictionary with the 3d positions of the root in train
    test_root_positions: dictionary with the 3d positions of the root in test
  """
  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions, dim=3 )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions, dim=3 )
  '''
  import src.viz_new as viz
  p3d = train_set[(1, 'Directions', 'Directions 1.cdf')][0,:]
  ax1 = plt.axes(projection='3d');viz.show3Dpose( p3d, ax1)
  '''

  if camera_frame:
    train_set = transform_world_to_camera( train_set, rcams )
    test_set  = transform_world_to_camera( test_set, rcams )

  '''
  import src.viz_new as viz
  p3d = train_set[(1, 'Directions', 'Directions 1..54138969.h5')][0,:]
  ax2 = plt.axes(projection='3d');viz.show3Dpose( p3d, ax2)

   # other cameras ex. 1, 'Directions', 'Directions 1..55011271.h5'
  '''
    
  # Apply 3d post-processing (centering around root)
  train_set, train_root_positions = postprocess_3d( train_set )
  test_set,  test_root_positions  = postprocess_3d( test_set )
  '''
  import src.viz_new as viz
  p3d = train_set[(1, 'Directions', 'Directions 1..54138969.h5')][0,:]
  ax3 = plt.axes(projection='3d');viz.show3Dpose( p3d, ax3)
  '''
  #CALCULATE LENGTH 
  bone_lengths_train_norm = None
  bone_lengths_test_norm = None

  if flag_bone_lengths:
    #calculate lengths for action directions
    bone_lengths_train = calculate_length_notNormalized( train_set, TRAIN_SUBJECTS, action = 'Directions' )
    bone_lengths_test = calculate_length_notNormalized( test_set, TEST_SUBJECTS, action = 'Directions' )
    # bone_lengths_train_norm, bone_lengths_test_norm  = standardize_bone_lengths(bone_lengths_train, bone_lengths_test)
    bone_mean, bone_std = bone_normalization_stats(bone_lengths_train)
    bone_lengths_train_norm, bone_lengths_test_norm = standardize_bone_lengths(bone_mean, bone_std, bone_lengths_train, bone_lengths_test)

  # Compute normalization statistics
  complete_train = copy.deepcopy( np.vstack(list(train_set.values() )))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3, predict_14=predict_14 )
  
  if normalize_target:
    # Divide every dimension independently. (as original paper)
    train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
    test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )
  else:
    #if the target is not normalize, i need to select the correct number of joints
    train_set = select_joints( train_set,  dim_to_use )
    test_set  = select_joints( test_set,   dim_to_use )
  '''
  import src.viz_new as viz
  pose3d = train_set[(1, 'Directions', 'Directions 1..55011271.h5')][0]
  ax = plt.axes(projection='3d')
  viz.show3D_norm_pose( pose3d, ax)
  '''
  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions, bone_lengths_train_norm, bone_lengths_test_norm


def postprocess_3d( poses_set ):
  """
  Center 3d points around root

  Args
    poses_set: dictionary with 3d data
  Returns
    poses_set: dictionary with 3d data centred around root (center hip) joint
    root_positions: dictionary with the original 3d position of each pose
  """
  root_positions = {}
  for k in poses_set.keys():
    # Keep track of the global position
    root_positions[k] = copy.deepcopy(poses_set[k][:,:3])

    # Remove the root from the 3d position
    poses = poses_set[k]
    poses = poses - np.tile( poses[:,:3], [1, len(H36M_NAMES)] ) #poses[:,:3] are the hip coordinates => hip are now 0,0,0
    poses_set[k] = poses

  return poses_set, root_positions
