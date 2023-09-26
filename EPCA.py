import json
import torch
import os
import argparse
import time
import random

import body as attack # 




from attack_utils import get_model, read_imagenet_data_specify, save_results
from foolbox.distances import l2
import numpy as np
from PIL import Image
import torch_dct

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

global sd 
sd = 20
torch.manual_seed(sd)
if torch.cuda.is_available():
	torch.cuda.manual_seed(sd)
	torch.cuda.manual_seed_all(sd)
np.random.seed(sd)
random.seed(sd)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", default="results", help="Output folder") # , "-o",
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet-18",
        help="The name of model you want to attack(resnet-18, inception-v3, vgg-16, resnet-50, resnet-101, densenet-121)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="images",
        help="The path of dataset"
    )
    parser.add_argument(
         "--csv",
        type=str,
        default=" ", 
        help="The path of csv information about dataset"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=sd,
        help='The random seed you choose'
    )
    parser.add_argument(
        '--max_queries',
        type=int,
        default=1000,
        help='The max number of queries in model'
    )
    parser.add_argument(
        '--ratio_mask',
        type=float,
        default=0.1,        
        help='ratio of mask'
    )
    parser.add_argument(
        '--dim_num',
        type=int,
        default=1,    
        help='the number of picked dimensions'
    )
    parser.add_argument(
        '--max_iter_num_in_2d',                  
        type=int,
        default=1,                               
        help='the maximum iteration number of attack algorithm in 2d subspace'
    )
    parser.add_argument(
        '--init_alpha',
        type=float,
        default=np.pi/2,
        help='the initial angle of alpha'
    )
    parser.add_argument(                               
        '--init_beta',
        type=float,
        default=np.pi/24,                        
        help='the initial angle of beta'
    )
    parser.add_argument(                               
        '--u',
        type=float,
        default=,    # 0 or 1                              
        help='momentum decay factor'
    )    
    parser.add_argument(                               
        '--step',
        type=float,
        default=0.1,                            
        help='control update step size for beta'
    )
    
    #################################
    ############# skip ##############
    
    parser.add_argument(                               
        '--lambd',
        type=float,
        default=0.1,                            
        help='control when to skip '
    )
    parser.add_argument(                               
        '--window_size',
        type=float,
        default=100,                            
        help='window size'
    )
    parser.add_argument(                               
        '--threshold',
        type=float,
        default=2.0,                           
        help='distance condition to activate skip '
    )
    parser.add_argument(                               
        '--low_ratio',
        type=float,
        default=0,                            
        help='start'
    )
    parser.add_argument(                               
        '--up_ratio',
        type=float,
        default=0.2,                          
        help='stop'
    )
    parser.add_argument(                               
        '--low_bound',
        type=float,
        default=6,                            
        help='low bound'
    )
    parser.add_argument(                               
        '--fremix_num',
        type=int,
        default=5,                            
        help='iteration number of fremix'
    )
        
    parser.add_argument("--results_name", default=" ", help="Output file name")    

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    if args.model_name=='inception-v3':
        args.side_length=299
    else:
        args.side_length=224


    ###############################
    print("Load Model: %s" % args.model_name)
    fmodel = get_model(args,device)

    ###############################
    
    

    ###############################
    print("Load Data")

    images, labels, selected_paths = read_imagenet_data_specify(args, device) 

    
    print("{} images loaded with the following labels: {}".format(len(images), labels))

    ###############################
    print("Attack !")
    time_start = time.time()

    epca = attack.EPCA(fmodel, input_device=device) #

    my_advs, q_list, my_intermediates, max_length, ac, l2_list = epca.attack(args,images, labels) 

    
    
    print('EPCA Attack Done')
    print("{:.2f} s to run".format(time.time() - time_start))
    print("Results of:", args.model_name)
    print('Top1 ASR:', ac[0]/len(images), 'Top2 ASR:', ac[1]/len(images), 'Top3 ASR:',ac[2]/len(images))
    print('Mean l2:', torch.mean(torch.tensor(l2_list)))
    
    my_labels_advs = fmodel(my_advs).argmax(1)
    my_advs_l2 = l2(images, my_advs) 


    
    save_results(args,my_intermediates, len(images))

    
    for image_i, adv in enumerate(my_advs):

        adv_i = np.array(my_advs[image_i] * 255).astype(np.uint8)              
        img_adv_i = Image.fromarray(adv_i.transpose(1, 2, 0), mode="RGB") 
        img_adv_i.save(os.path.join('./adv', "{}_adversarial.jpg".format(image_i)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    