import torch
from src.model import LinearModel,weight_init

checkpoint_path = "fold1_best_tuning.pth.tar"
ckpt = torch.load(checkpoint_path)
batch_size = ckpt['batch_size']
p_dropout = ckpt['p_dropout']

best_trained_model = LinearModel(batch_size, True, p_dropout, linear_size=opt.linear_size, num_stage=opt.num_stage)

best_trained_model.load_state_dict(ckpt['state_dict'])