import os
import argparse
import time
from datetime import datetime
import torch
import numpy as np
import utils as utils

from data_loader import data_loader
from cooperative_diff_game import masdes


parser = argparse.ArgumentParser("Cooperative Differential Game for Time-Series")
parser.add_argument("-gpu_num", type=int, default=0, help="Number of GPU to use.")
parser.add_argument("-load", type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument("-model_load", default=False, help="Import Pretrained Model")
parser.add_argument("-data_set", type=str, default="air_quality", help="[physionet,speech,air_quality]")
parser.add_argument("-n_stages", type=int, default=500, help="Number of stages")
parser.add_argument("-random_seed", type=int, default=2022, help="Random_seed")
parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("-batch_size", type=int, default=128, help="Batch size")
parser.add_argument("-T_p", type=int, default=48, help="Number of Total time steps")
parser.add_argument("-T_o", type=int, default=36, help="Observable Time interval [0, T_o)")
parser.add_argument("-A", type=int, default=36, help="Number of players")
parser.add_argument("-D", type=int, default=6, help="Number of data dimensions")
parser.add_argument("-L", type=int, default=2, help="Number of Piecewise layers")
parser.add_argument("-T", type=float, default=1.0, help="Total time interval")
parser.add_argument('-c_gap', type=int, default=1, help='Time gap between Piecewise layers')
parser.add_argument('-r_gap', type=int, default=3, help='Manage Area for each Piecewise layers')
parser.add_argument('-AdaIN', default=False, help='On-Off AdaIN')
parser.add_argument("-hidden_dim", type=int, default=128, help="Dimension of Piecewise FCN layers")
parser.add_argument("-hidden_weight", type=int, default=36, help="Dimension of Piecewise Aggregation layers")
parser.add_argument("-sigma_high", type=float, default=5.0, help="Maximum volatility")
parser.add_argument("-sigma_low",  type=float, default=0.1, help="Minumum volatility")
parser.add_argument("-sde_type",  type=str, default="vanilla", help="SDE Type")

args = parser.parse_args()


def main():
    device = "cuda:{}".format(args.gpu_num)
    args.device = device
    torch.cuda.set_device(device)
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    now = datetime.now()
    experimentID = now.strftime("%b%d-%H%M%S")
    args.ID = experimentID

    save_ckpt_path = "chpt/{}/layer={}_dim={}_gpu={}".format(args.data_set, args.L, args.hidden_dim,args.gpu_num)
    save_log_path = "logs/{}/".format(args.data_set)
    utils.makedirs(save_ckpt_path)
    utils.makedirs(save_log_path)
    
    ckpt_path = os.path.join(save_ckpt_path, "experiment_" + str(args.ID) + ".ckpt")
    log_path = save_log_path + str(args.ID) + ".log"

    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath("__file__"))
    for o in vars(args):
        logger.info("{} : {}".format(o, getattr(args,o)))
    logger.info("Experiments" + str(experimentID))


    train_loader, test_loader = data_loader(args)

    model = masdes(args, logger)
    model = model.to(args.device)
    
    model.train_and_eval(train_loader, test_loader)
    

if __name__ == "__main__":
    main()  