import os
import sys
import time
import json
from pprint import pprint
#from six.moves import xrange
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import random_split

from src.model import LinearModel,weight_init
import src.data_utils as data_utils
import src.cameras as cameras
import src.utils as utils
import src.procrustes as procrustes
import src.viz_new as viz
import wandb

#FIXED SETTINGS
opt = {
    "SimpleTest": False,
    "action": "All",
    "camera_frame": True,
    "cameras_path": "data/h36m/metadata.xml",
    "ckpt": "checkpoint/train/",
    "data_dir": "data/h36m/",
    "evaluateActionWise": True,
    "job": 8,
    "linear_size": 1024,
    "load": "checkpoint/train/ckpt_last.pth.tar",
    "lr_gamma": 0.96,
    "max_norm": True,
    "n_inputs": 16,
    "num_stage": 2,
    "predict_14": False,
    "procrustes": True,
    "use_hg": False,
    "TRAIN_SUBJECTS" : [1,5,6,7,8],
    "TEST_SUBJECTS" :[9,11],
    "actions": [
           "Directions",
           "Discussion",
           "Eating",
           "Greeting",
           "Phoning",
           "Photo",
           "Posing",
           "Purchases",
           "Sitting",
           "SittingDown",
           "Smoking",
           "Waiting",
           "WalkDog",
           "Walking",
           "WalkTogether"],
      
}
from types import SimpleNamespace 
opt = SimpleNamespace(**opt)
print(opt)
SUBJECT_IDS = [1,5,6,7,8,9,11]
actions = data_utils.define_actions(opt.action, opt.actions)
num_actions = len(actions)
criterion = nn.MSELoss(reduction='mean').cuda() #criterion for loss function

def read_load_data(opt):
    print(">>> loading data") 
    # Load camera parameters
    rcams = cameras.load_cameras(opt.cameras_path, SUBJECT_IDS)
     # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(actions, opt.data_dir, opt.camera_frame, rcams, opt.TRAIN_SUBJECTS, opt.TEST_SUBJECTS, opt.predict_14 )  
    # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
    if opt.use_hg:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, opt.data_dir, opt.TRAIN_SUBJECTS, opt.TEST_SUBJECTS)
    else: 
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, opt.data_dir, rcams, opt.TRAIN_SUBJECTS, opt.TEST_SUBJECTS)

    return  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d


def create_test_val_subsets(train_set_3d, train_set_2d):
    #create validation set from training set
    test_abs = int(len(train_set_3d) * 0.80) 
    keys_val_subset = random.sample(train_set_3d.keys(),len(train_set_3d) - test_abs)
    train_subset_3d = {}
    train_subset_2d = {}
    val_subset_3d = {}
    val_subset_2d = {}
    for key in train_set_3d.keys():
        if key in keys_val_subset:
            val_subset_3d[key] = train_set_3d[key] 
            val_subset_2d[key] = train_set_2d[key]
        else:
            train_subset_3d[key] = train_set_3d[key] 
            train_subset_2d[key] = train_set_2d[key]

    return train_subset_3d, val_subset_3d, train_subset_2d, val_subset_2d

def get_all_batches(opt,data_x, data_y, batch_size, training=True ):
    """
    Obtain a list of all the batches, randomly permutted
    Args
      data_x: dictionary with 2d inputs
      data_y: dictionary with 3d expected outputs
      camera_frame: whether the 3d data is in camera coordinates
      training: True if this is a training batch. False otherwise.

    Returns
      encoder_inputs: list of 2d batches
      decoder_outputs: list of 3d batches
    """

    # Figure out how many frames we have
    n = 0

    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d

    # 2d pos 具有 16个关节点
    encoder_inputs  = np.zeros((n, opt.n_inputs*2), dtype=float)
    # 3d pose 
    
    if opt.predict_14:
      decoder_outputs = np.zeros((n, 14*3), dtype=float)
    else: 
      decoder_outputs = np.zeros((n, 16*3),dtype=float)

    # Put all the data into big arrays
    idx = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if (opt.camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and opt.camera_frame else key3d

      n2d, _ = data_x[ key2d ].shape
      encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
      decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
      idx = idx + n2d


    if training:
      # Randomly permute everything
      idx = np.random.permutation( n )
      encoder_inputs  = encoder_inputs[idx, :]
      decoder_outputs = decoder_outputs[idx, :]

    # Make the number of examples a multiple of the batch size
    n_extra  = n % batch_size
    if n_extra > 0:  # Otherwise examples are already a multiple of batch size
      encoder_inputs  = encoder_inputs[:-n_extra, :]
      decoder_outputs = decoder_outputs[:-n_extra, :]

    n_batches = n // batch_size
    encoder_inputs  = np.split( encoder_inputs, n_batches )
    decoder_outputs = np.split( decoder_outputs, n_batches )

    return encoder_inputs, decoder_outputs


def evaluate_batches_test(opt,model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  encoder_inputs, decoder_outputs, batch_size, current_epoch=0 ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess
    model
    data_mean_3d
    data_std_3d
    dim_to_use_3d
    dim_to_ignore_3d
    data_mean_2d
    data_std_2d
    dim_to_use_2d
    dim_to_ignore_2d
    current_step
    encoder_inputs
    decoder_outputs
    current_epoch
  Returns

    total_err
    joint_err
    step_time
    loss
  """

  n_joints = 17 if not(opt.predict_14) else 14
  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )
    
    enc_in = torch.from_numpy(encoder_inputs[i]).float()
    dec_out = torch.from_numpy(decoder_outputs[i]).float()

    

    inputs = Variable(enc_in.cuda())
    targets = Variable(dec_out.cuda())

    outputs = model(inputs)
    
    # denormalize
    enc_in  = data_utils.unNormalizeData( enc_in,  data_mean_2d, data_std_2d, dim_to_ignore_2d )
    dec_out = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    poses3d = data_utils.unNormalizeData( outputs.data.cpu().numpy(), data_mean_3d, data_std_3d, dim_to_ignore_3d )

    # Keep only the relevant dimensions
    dtu3d = np.hstack( (np.arange(3), dim_to_use_3d) ) if not(opt.predict_14) else  dim_to_use_3d

    dec_out = dec_out[:, dtu3d]
    poses3d = poses3d[:, dtu3d]

    assert dec_out.shape[0] == batch_size
    assert poses3d.shape[0] == batch_size

    if opt.procrustes:
      # Apply per-frame procrustes alignment if asked to do so
      for j in range(batch_size):
        gt  = np.reshape(dec_out[j,:],[-1,3])
        out = np.reshape(poses3d[j,:],[-1,3])
        _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
        out = (b*out.dot(T))+c

        poses3d[j,:] = np.reshape(out,[-1,17*3] ) if not(opt.predict_14) else np.reshape(out,[-1,14*3] )

    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints*3, 3):get_all_batches
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx = dist_idx + 1

    all_dists.append(dists)
    assert sqerr.shape[0] == batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time

def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action

  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """
  return {k:v for k, v in poses_set.items() if k[1] == action} 

def evaluate_batches(opt,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  dec_out, outputs, batch_size):
  
  n_joints = 17 if not(opt.predict_14) else 14

  all_dists = []
    
  # denormalize
  dec_out2 = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
  poses3d = data_utils.unNormalizeData( outputs.data.cpu().numpy(), data_mean_3d, data_std_3d, dim_to_ignore_3d )

  # Keep only the relevant dimensions
  dtu3d = np.hstack( (np.arange(3), dim_to_use_3d) ) if not(opt.predict_14) else  dim_to_use_3d

  dec_out2 = dec_out2[:, dtu3d]
  poses3d = poses3d[:, dtu3d]

  assert dec_out2.shape[0] == batch_size
  assert poses3d.shape[0] == batch_size

  if opt.procrustes:
    # Apply per-frame procrustes alignment if asked to do so
    for j in range(batch_size):
      gt  = np.reshape(dec_out2[j,:],[-1,3])
      out = np.reshape(poses3d[j,:],[-1,3])
      _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
      out = (b*out.dot(T))+c

      poses3d[j,:] = np.reshape(out,[-1,17*3] ) if not(opt.predict_14) else np.reshape(out,[-1,14*3] )

  # Compute Euclidean distance error per joint
  sqerr = (poses3d - dec_out2)**2 # Squared error between prediction and expected output
  dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
  dist_idx = 0
  for k in np.arange(0, n_joints*3, 3):
    # Sum across X,Y, and Z dimenstions to obtain L2 distance
    dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
    dist_idx = dist_idx + 1

  all_dists.append(dists)
  assert sqerr.shape[0] == batch_size

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  # joint_err = np.mean( all_dists, axis=0 ) #mean joint errors along batches
  total_err = np.mean( all_dists )
  return total_err


def test_best_model(opt, best_config, best_result):
    best_trained_model = LinearModel(best_config.batch_size,opt.predict_14, best_config.p_dropout, linear_size=opt.linear_size, num_stage=opt.num_stage)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    # checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    checkpoint_path = best_result.checkpoint.to_directory() + '/MyCheckpoint.pth' 

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = read_load_data(opt)

    best_trained_model.eval()
    print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs
          
    cum_err = 0
    record = ''
    for action in actions:
        with torch.no_grad():            
            print("{0:<12} ".format(action), end="")
            # Get 2d and 3d testing data for this action
            # Get 2d and 3d testing data for this action
            action_test_set_2d = get_action_subset( test_set_2d, action )
            action_test_set_3d = get_action_subset( test_set_3d, action )
            encoder_inputs, decoder_outputs = get_all_batches(opt, action_test_set_2d, action_test_set_3d , best_config.batch_size, training=False)

            total_err, joint_err, step_time = evaluate_batches_test( opt, best_trained_model,
              data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
              data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
              encoder_inputs, decoder_outputs, best_config.batch_size )
            cum_err = cum_err + total_err

            print("{0:>6.2f}".format(total_err))


        avg_error = cum_err/float(len(actions) )
    print("Best trial test set error: {}".format(avg_error))

def train(config):
    wandb.init(config = config)
    # config = wandb.config

    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = load_data(opt)
    train_subset_3d, val_subset_3d, train_subset_2d, val_subset_2d = create_test_val_subsets(train_set_3d, train_set_2d)
    
    # CREATE MODEL
    print(">>> creating model")
    model = LinearModel(config.batch_size,opt.predict_14, config.p_dropout, linear_size=opt.linear_size, num_stage=opt.num_stage)
    model = model.cuda()
    model.apply(weight_init)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    glob_step = 0
    lr_init = lr=config.lr
    lr_now =lr_init
    # lr_decay = opt.lr_decay
    # before batch_size = 64 and lr_decay = 100'000
    lr_decay = 64000000/config.batch_size    
    lr_gamma = opt.lr_gamma

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

    
    # if checkpoint_dir:
    #   checkpoint_path = checkpoint_dir + '/MyCheckpoint.pth' 
    #   model_state, optimizer_state = torch.load(checkpoint_path)
    #       # os.path.join(checkpoint_dir, "MyCheckpoint"))
    #   model.load_state_dict(model_state)
    #   optimizer.load_state_dict(optimizer_state)

    cudnn.benchmark = True  #optimize when using fixed input
    current_epoch = 0

    #LOOP THROUGH THE EPOCHS
    while current_epoch < config.epochs:
        model.train()
        loss_train = 0
        current_steps = 0
        err_train = 0

        # Optional
        wandb.watch(model)

        encoder_inputs, decoder_outputs = get_all_batches(opt, train_subset_2d, train_subset_3d, config.batch_size, training=True )      
        nbatches = len( encoder_inputs )
        print("There are {0} train batches".format( nbatches ))
        #LOOP THROUGH THE EPOCHS
        for i in range( nbatches ): 

            if glob_step % lr_decay == 0 or glob_step == 1:
                lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, lr_gamma)
            
            enc_in = torch.from_numpy(encoder_inputs[i]).float()
            dec_out = torch.from_numpy(decoder_outputs[i]).float()            
            inputs = Variable(enc_in.cuda())
            targets = Variable(dec_out.cuda())
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)                                                    
            step_loss_train = criterion(outputs, targets)
            step_loss_train.backward()
            optimizer.step()

            step_err_train = evaluate_batches(opt,
            data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
            dec_out, outputs, config.batch_size)

            err_train += step_err_train
            loss_train += float(step_loss_train)

            # ENDS CURRENT BATCH
            current_steps += 1
            glob_step += 1

        loss_train = loss_train / current_steps
        err_train = err_train/ current_steps
        print("training epoch: [%d] train loss: %.3f train error: %.3f " % (current_epoch+1,loss_train , err_train ))
        
        # wandb.log({ "loss_train": loss_train, "err_train": err_train, "step" :current_epoch})
        # wandb.log({ "current_lr": lr_now,  "step" :current_epoch})

        


        # clear useless chache
        torch.cuda.empty_cache()
    
        model.eval()
        # VALIDATION 
        loss_val = 0.0
        current_steps = 0
        err_val = 0
        encoder_inputs, decoder_outputs = get_all_batches(opt, val_subset_2d, val_subset_3d, config.batch_size, training=True )
        nbatches = len( encoder_inputs )
        print("There are {0} validation batches".format( nbatches ))
        for i in range( nbatches ):
            with torch.no_grad():
                enc_in = torch.from_numpy(encoder_inputs[i]).float()
                dec_out = torch.from_numpy(decoder_outputs[i]).float()            
                inputs = Variable(enc_in.cuda())
                targets = Variable(dec_out.cuda())

                outputs = model(inputs)
                step_loss_val = criterion(outputs, targets)

                step_err_val = evaluate_batches(opt,
                data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d, dec_out, outputs, config.batch_size)
                
                err_val += step_err_val               
                loss_val  += float(step_loss_val)
                
                # ENDS CURRENT BATCH
                current_steps += 1

        loss_val = loss_val / current_steps
        err_val = err_val/ current_steps

        print("training epoch: [%d] val loss: %.3f val error: %.3f " % (current_epoch+1,loss_val , err_val ))
        wandb.log({ "loss_val": loss_val, "err_val": err_val, "loss_train": loss_train, "err_train": err_train, "step" :current_epoch, "current_lr": lr_now})
        
        str_model_params = "p_dropout=" + str(config.p_dropout) + "_" + "lr_init=" + str(config.lr)  + "_" + "batch_size="+ str(config.batch_size) + "_" + "n_epochs="+ str(config.epochs)+"_"
        str_model = str_model_params +  'ckpt.pth.tar'
        file_path = os.path.join(opt.ckpt, str_model)
        torch.save({'epoch': current_epoch,
                    'lr_now': lr_now,
                    'step': glob_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'batch_size': config.batch_size,
                    'lr': config.lr,
                    'tot_epochs': config.epochs,
                    'err_train':err_train,
                    'err_val':err_val,
                    'p_dropout': config.p_dropout}, file_path)
                    
        #ENDS CURRENT EPOCH
        current_epoch = current_epoch + 1
        # clear useless chache
        torch.cuda.empty_cache()
        
    print("Finished Training")
      

if __name__ == "__main__":  
  config={}
  print("START")
  print(sys.argv)
  config["batch_size"] = int(sys.argv[1].split("=")[1])
  config["epochs"] = int(sys.argv[2].split("=")[1])
  config["lr"] = float(sys.argv[3].split("=")[1])
  config["p_dropout"] = float(sys.argv[4].split("=")[1])
  config = SimpleNamespace(**config)
  print(config)

  train(config)


