from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
import sys
import torch_dct
from attack_utils import *
import time

global device


# initialize an adversarial example with uniform noise
def get_x_adv(x_o: torch.Tensor, label: torch.Tensor, model) -> torch.Tensor:
    criterion = get_criterion(label)
    init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=100) # 
    x_adv = init_attack.run(model, x_o, criterion)
    return x_adv


# compute the difference
def get_difference(x_o: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    difference = x_adv - x_o
    if torch.norm(difference, p=2) == 0:
        raise ('difference is zero vector!')
        return difference
    return difference 


def rotate_in_2d(x_o2x_adv: torch.Tensor, direction: torch.Tensor, theta: float = np.pi / 8) -> torch.Tensor:
    alpha = torch.sum(x_o2x_adv * direction) / torch.sum(x_o2x_adv * x_o2x_adv)
    orthogonal = direction - alpha * x_o2x_adv 
    direction_theta = x_o2x_adv * np.cos(theta) + torch.norm(x_o2x_adv, p=2) / torch.norm(orthogonal,p=2) * orthogonal * np.sin(theta)
    direction_theta = direction_theta / torch.norm(direction_theta) * torch.norm(x_o2x_adv) 
    return direction_theta


# 
def gt_1dim_line(args, x_o2x_adv: torch.Tensor, n, ratio_size_mask=0.3, if_left=1) -> torch.Tensor:

    random.seed(args.seed)      
    
    zero_mask = torch.zeros(size=[args.side_length, args.side_length], device=device) 
    size_mask = int(args.side_length * ratio_size_mask) 
    if if_left:                                         
        zero_mask[:size_mask, :size_mask] = 1            

    else:
        zero_mask[-size_mask:, -size_mask:] = 1         

    to_choose = torch.where(zero_mask == 1)              
    x = to_choose[0]                                   
    y = to_choose[1]                                    



    select = np.random.choice(len(x), size=1, replace=False)       
    mask1 = torch.zeros_like(zero_mask)                          
    mask1[x[select], y[select]] = 1                                
    mask1 = mask1.reshape(-1, args.side_length, args.side_length)  

    rdu = torch.rand(1)
    if rdu < torch.tensor(1/3):
        mask = torch.cat([mask1, torch.zeros_like(zero_mask).reshape(-1, args.side_length, args.side_length) , torch.zeros_like(zero_mask).reshape(-1, args.side_length, args.side_length) ], dim=0).expand([1, 3, args.side_length, args.side_length])
    elif torch.tensor(1/3) < rdu and rdu < torch.tensor(2/3):      
        mask = torch.cat([torch.zeros_like(zero_mask).reshape(-1, args.side_length, args.side_length) , mask1, torch.zeros_like(zero_mask).reshape(-1, args.side_length, args.side_length) ], dim=0).expand([1, 3, args.side_length, args.side_length])
    elif torch.tensor(2/3) < rdu:      
        mask = torch.cat([torch.zeros_like(zero_mask).reshape(-1, args.side_length, args.side_length) , torch.zeros_like(zero_mask).reshape(-1, args.side_length, args.side_length) , mask1], dim=0).expand([1, 3, args.side_length, args.side_length])
                                             
    direction = rotate_in_2d(x_o2x_adv, mask, theta=np.pi / 2)

    return direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv, p=2), mask 

    
def get_x_hat_in_2d(args, x_o: torch.Tensor, x_adv: torch.Tensor, axis_unit1: torch.Tensor, axis_unit2: torch.Tensor,
                    net: torch.nn.Module, queries, original_label, max_iter=2, init_alpha = np.pi/2):

    if not hasattr(get_x_hat_in_2d, 'alpha'):  
        get_x_hat_in_2d.alpha = init_alpha     

    if not hasattr(get_x_hat_in_2d, 'theta'):  
        get_x_hat_in_2d.theta = args.init_beta

    if not hasattr(get_x_hat_in_2d, 'v'): 
        get_x_hat_in_2d.v = 0.0

    d = torch.norm(x_adv - x_o, p=2)           

    theta = get_x_hat_in_2d.theta

    x_hat = torch_dct.idct_2d(x_adv)               
    
    right_theta = np.pi - get_x_hat_in_2d.alpha    
    x = x_o + (axis_unit1 * np.cos(theta) + axis_unit2 * np.sin(theta)) * d / np.sin(get_x_hat_in_2d.alpha) * np.sin(
        get_x_hat_in_2d.alpha + theta)             
                                                       
    x = torch_dct.idct_2d(x)                       
    
    get_x_hat_in_2d.total += 1                                   
    get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0) 
    x = torch.clamp(x, 0, 1)
    label = get_label(net(x))
    queries += 1
    if label != original_label:       
        x_hat = x                     
        left_theta = theta           
        flag = 1                      
        
    else:                       
    
        x = x_o + d * (axis_unit1 * np.cos(theta) - axis_unit2 * np.sin(theta)) / np.sin(
            get_x_hat_in_2d.alpha) * np.sin(
            get_x_hat_in_2d.alpha + theta) 
        x = torch_dct.idct_2d(x)
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)
        label = get_label(net(x))
        queries += 1
        if label != original_label:  
            x_hat = x
            left_theta = theta
            flag = -1
            
        else:                    

            get_x_hat_in_2d.v = args.u * get_x_hat_in_2d.v - 1.0
            get_x_hat_in_2d.theta = get_x_hat_in_2d.theta + args.step * np.sign(get_x_hat_in_2d.v) # sign
            get_x_hat_in_2d.theta = max(args.init_beta, get_x_hat_in_2d.theta)
            get_x_hat_in_2d.theta = min(np.pi - args.init_alpha - 0.0001, get_x_hat_in_2d.theta)

            return x_hat, queries, False   

    #### binary search for beta ####
    theta = (left_theta + right_theta) / 2
    for i in range(max_iter):            
        x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) / np.sin(
            get_x_hat_in_2d.alpha) * np.sin(
            get_x_hat_in_2d.alpha + theta)
        x = torch_dct.idct_2d(x)     
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)
        label = get_label(net(x))
        queries += 1
        if label != original_label:  
            left_theta = theta       
            x_hat = x
                          
            get_x_hat_in_2d.v = args.u * get_x_hat_in_2d.v + 1.0
            get_x_hat_in_2d.theta = get_x_hat_in_2d.theta + args.step * np.sign(get_x_hat_in_2d.v)
            get_x_hat_in_2d.theta = max(args.init_beta, get_x_hat_in_2d.theta)
            get_x_hat_in_2d.theta = min(np.pi - args.init_alpha - 0.0001, get_x_hat_in_2d.theta)
            
            return x_hat, queries, True                 

        else:        

            right_theta = theta                     
        theta = (left_theta + right_theta) / 2        
    
    get_x_hat_in_2d.v = args.u * get_x_hat_in_2d.v + 1.0
    get_x_hat_in_2d.theta = get_x_hat_in_2d.theta + args.step * np.sign(get_x_hat_in_2d.v)
    get_x_hat_in_2d.theta = max(args.init_beta, get_x_hat_in_2d.theta)
    get_x_hat_in_2d.theta = min(np.pi - args.init_alpha - 0.0001, get_x_hat_in_2d.theta)
    return x_hat, queries, True


def get_x_hat_arbitary(args,x_o: torch.Tensor, net: torch.nn.Module, original_label, init_x=None,dim_num=5):

    
    if get_label(net(x_o)) != original_label:      

        return x_o, 0, [[0.0, 0.0, 0.0]]     

    if init_x is None:  
        x_adv = get_x_adv(x_o, original_label, net)   
    else:
        x_adv = init_x 
    x_hat = x_adv
    queries = 0.
    dist = torch.norm(x_o - x_adv)
    intermediate = []
    

    intermediate.append([0, dist.item(), get_x_hat_in_2d.theta])  

    while queries < args.max_queries : 

        x_o2x_adv = torch_dct.dct_2d(get_difference(x_o, x_adv))  
        axis_unit1 = x_o2x_adv / torch.norm(x_o2x_adv)            
        
        direction, mask = gt_1dim_line(args,x_o2x_adv, dim_num, args.ratio_mask, args.dim_num)
        
        axis_unit2 = direction / torch.norm(direction)
        
        x_hat, queries, changed = get_x_hat_in_2d(args, torch_dct.dct_2d(x_o), torch_dct.dct_2d(x_adv), 
                                                  axis_unit1, axis_unit2, 
                                                  net, queries, original_label, max_iter=args.max_iter_num_in_2d,
                                                  init_alpha=args.init_alpha) # pi/2
        
        
        x_adv = x_hat  
        dist = torch.norm(x_hat - x_o) 
        

        intermediate.append([queries, dist.item(), get_x_hat_in_2d.theta]) 

        if queries >= args.max_queries:
            break
    return x_hat, queries, intermediate 


class EPCA:
    def __init__(self, model, input_device):    
        self.net = model                        
        global device
        device = input_device

    def attack(self, args,inputs, labels):      
        x_adv_list = torch.zeros_like(inputs)   
        queries = []
        intermediates = []
        init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50) 
        criterion = get_criterion(labels.long())                                          
        best_advs = init_attack.run(self.net, inputs, criterion, early_stop=None) 

        
        final_l2 = []
        max_length = 0          
        acc = [0., 0., 0.]    
        for i, [input, label] in enumerate(zip(inputs, labels)):    
            print('[{}/{}]:'.format(i + 1, len(inputs)), end='')      
            global probability                                       
            probability = np.ones(input.shape[1] * input.shape[2])    
            global is_visited_1d
            is_visited_1d = torch.zeros(input.shape[0] * input.shape[1] * input.shape[2])  
            global selected_h
            global selected_w
            selected_h = input.shape[1]
            selected_w = input.shape[2]
            get_x_hat_in_2d.alpha = np.pi / 2          
            
            get_x_hat_in_2d.theta = args.init_beta     
            get_x_hat_in_2d.v = 0.0
                                 
            
            get_x_hat_in_2d.total = 0                
            get_x_hat_in_2d.clamp = 0               

            x_adv, q, intermediate = get_x_hat_arbitary(args,input[np.newaxis, :, :, :], self.net,
                                                        label.reshape(1, ).to(device),
                                                        init_x=best_advs[i][np.newaxis, :, :, :], dim_num=args.dim_num)


            
            #  RMSE
            x_adv_list[i] = x_adv[0] # [1,3,h,w] -> [3,h,w]
            diff = torch.norm(x_adv[0] - input, p=2) / (args.side_length * np.sqrt(3)) 
            final_l2.append(torch.norm(x_adv[0] - input, p=2))
            if diff <= 0.1:
                acc[0] += 1
            if diff <= 0.05:
                acc[1] += 1
            if diff <= 0.01:
                acc[2] += 1
                
            # 
            print("Top-1 Acc:{} Top-2 Acc:{} Top-3 Acc:{}".format(acc[0] / (i + 1), acc[1] / (i + 1),
                                                                                     acc[2] / (i + 1)))
            queries.append(q) #
            intermediates.append(intermediate)
            if max_length < len(intermediate):
                max_length = len(intermediate)
        queries = np.array(queries)
        return x_adv_list, queries, intermediates, max_length, acc, final_l2
