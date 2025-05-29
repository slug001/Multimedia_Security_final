# -*- coding = utf-8 -*-
import numpy as np
import torch
import copy
import time
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from models.Update import LocalUpdate
import heapq

def cos(a, b):
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def fltrust(params, central_param, global_parameters, args):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


import torch

def kernel_function(x, y):
    sigma = 1.0
    return torch.exp(-torch.norm(x - y) ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y):
    # Compute the MMD between two tensors x and y
    # x and y should have the same number of samples
    m = x.size(0)
    n = y.size(0)
    # Compute the kernel matrices for x and y
    xx_kernel = torch.zeros((m, m))
    yy_kernel = torch.zeros((n, n))
    xy_kernel = torch.zeros((m, n))
    for i in range(m):
        for j in range(i, m):
            xx_kernel[i, j] = xx_kernel[j, i] = kernel_function(x[i], x[j])

    for i in range(n):
        for j in range(i, n):
            yy_kernel[i, j] = yy_kernel[j, i] = kernel_function(y[i], y[j])

    for i in range(m):
        for j in range(n):
            xy_kernel[i, j] = kernel_function(x[i], y[j])
    # Compute the MMD statistic
    mmd = (torch.sum(xx_kernel) / (m * (m - 1))) + (torch.sum(yy_kernel) / (n * (n - 1))) - (2 * torch.sum(xy_kernel) / (m * n))
    return mmd


def flare(w_updates, w_locals, net, central_dataset, dataset_test, global_parameters, args):
    w_feature=[]
    temp_model = copy.deepcopy(net)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    
    for client in w_locals:
        net.load_state_dict(client)
        local = LocalUpdate(
                args=args, dataset=dataset_test, idxs=central_dataset)
        feature = local.get_PLR(
            net=copy.deepcopy(net).to(args.device))
        w_feature.append(feature)
    distance_list=[[] for i in range(len(w_updates))]
    # distance_list=[list(len(w_updates)) for i in range(len(w_updates))]
    for i in range(len(w_updates)):
        for j in range(i+1, len(w_updates)):
            score = compute_mmd(w_feature[i], w_feature[j])
            distance_list[i].append(score.item())
            distance_list[j].append(score.item())
    print('defense line121 distance_list', distance_list)
    vote_counter=[0 for i in range(len(w_updates))]
    k = round(len(w_updates)*0.5)
    for i in range(len(w_updates)):
        IDs = np.argsort(distance_list[i])
        for j in range(len(IDs)):
            # client_id is the index of client i-th client voting for
            # distance_list[] only records score with other clients without itself
            # so distance_list[i][i] should be itself
            # client_id = j + 1 after j >= i
            if IDs[j] >= i:
                client_id = IDs[j] + 1 
            else:
                client_id = IDs[j]
            vote_counter[client_id] += 1
            if j + 1 >= k:  # first 𝑘 elements in 𝐼 𝐷𝑠 and vote for it
                break

    trust_score = [x/sum(vote_counter) for x in vote_counter]
    # print('defense line188 len trust_score', trust_score)
    
    w_avg = copy.deepcopy(global_parameters)
    for k in w_avg.keys():
        for i in range(0, len(w_updates)):
            try:
                w_avg[k] += w_updates[i][k] * trust_score[i]
            except:
                print("Fed.py line17 type_as", 'w_updates[i][k].type():', w_updates[i][k].type(), k)
                w_updates[i][k] = w_updates[i][k].type_as(w_avg[k]).long()
                w_avg[k] = w_avg[k].long() + w_updates[i][k] * trust_score[i]
    return w_avg


def log_layer_wise_distance(updates):
    # {layer_name, [layer_distance1, layer_distance12...]}
    layer_distance = {}
    for layer, val in updates[0].items():
        if 'num_batches_tracked' in layer:
            continue
        # for each layer calculate distance among models
        for model in updates:
            temp_layer_dis = 0
            for model2 in updates:
                temp_norm = torch.norm((model[layer] - model2[layer]))
                temp_layer_dis += temp_norm
            if layer not in layer_distance.keys():
                layer_distance[layer] = []
            layer_distance[layer].append(temp_layer_dis.item())
    return layer_distance
    
        
def layer_krum(gradients, n_attackers, args, multi_k=False):
    new_global = {}
    for layer in gradients[0].keys():
        if layer.split('.')[-1] == 'num_batches_tracked' or layer.split('.')[-1] == 'running_mean' or layer.split('.')[-1] == 'running_var':
            new_global[layer] = gradients[-1][layer]
        else:
            layer_gradients = [x[layer] for x in gradients]
            new_global[layer] = layer_multi_krum(layer_gradients, n_attackers, args, multi_k)
    return new_global

def layer_flatten_grads(gradients):
    flat_epochs = []
    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        flat_epochs.append(gradients[n_user].cpu().numpy().flatten().tolist())
    flat_epochs = np.array(flat_epochs)
    return flat_epochs

def layer_multi_krum(layer_gradients, n_attackers, args, multi_k=False):
    grads = layer_flatten_grads(layer_gradients)
    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))
    score_record = None
    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        if args.log_distance == True and score_record == None:
            print('defense.py line149 (krum distance scores):', scores)
            score_record = scores
            args.krum_distance.append(scores)
            layer_distance_dict = log_layer_wise_distance(gradients)
            args.krum_layer_distance.append(layer_distance_dict)
            # print('defense.py line149 (layer_distance_dict):', layer_distance_dict)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    agg_layer = 0
    for selected_layer in candidate_indices:
        agg_layer += layer_gradients[selected_layer]
    agg_layer /= len(candidate_indices)
    return agg_layer

def multi_krum(gradients, n_attackers, args, multi_k=False):
    grads = flatten_grads(gradients)
    print("[DEBUG] multi_krum start: len(grads)=", len(grads), "n_attackers=", n_attackers)
    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    # 确保 grads 不空
    if len(remaining_updates) == 0:
        raise RuntimeError("[DEBUG] multi_krum: no updates to defend against!")
    all_indices = np.arange(len(grads))
    
    score_record = None

    entered = False
    while len(remaining_updates) > 2 * n_attackers + 2:
        entered = True
        print("[DEBUG] Entering krum loop; remaining_updates=", len(remaining_updates))
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        print("[DEBUG] Computed scores:", scores)
        
        if args.log_distance == True and score_record == None:
            print('defense.py line149 (krum distance scores):', scores)
            score_record = scores
            args.krum_distance.append(scores)
            layer_distance_dict = log_layer_wise_distance(gradients)
            args.krum_layer_distance.append(layer_distance_dict)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    args.turn+=1
    for selected_client in candidate_indices:
        if selected_client < num_malicious_clients:
            args.wrong_mal += 1

    # 在使用 scores 之前再检查一次
    if 'scores' not in locals():
        raise RuntimeError("[DEBUG] multi_krum: scores was never computed!")
    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]
    
    return np.array(candidate_indices)



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2

def get_update2(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        update2[key] = update[key] - model[key]
    return update2


def fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp):
    pred_update = []
    distance = []
    for i in range(len(old_update_list)):
        pred_update.append((old_update_list[i] + hvp).view(-1))
        
    
    pred_update = torch.stack(pred_update)
    local_update_list = torch.stack(local_update_list)
    old_update_list = torch.stack(old_update_list)
    
    distance = torch.norm((old_update_list - local_update_list), dim=1)
    print('defense line219 distance(old_update_list - local_update_list):',distance)

    distance = torch.norm((pred_update - local_update_list), dim=1)
    distance = distance / torch.sum(distance)
    return distance

def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(100)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/100
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    # print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    # print(silhouette_score(score.reshape(-1, 1), label_pred))
    print('defence.py line233 label_pred (0 = malicious pred)', label_pred)
    return label_pred

def detection1(score):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    select_k = 2  # default detect attacks
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1

def RLR(global_model, agent_updates_list, args):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    print('defense.py line 430 lr_vector == -1', lr_vector[lr_vector==-1].shape[0]/lr_vector.view(-1).shape[0])
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs.to(args.gpu)
   
    


def flame(local_model, update_params, global_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            print(i)
            f.write('\n')
        f.write('\n')
        f.write("--------Round--------")
        f.write('\n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    print(benign_client)
   
    for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal+=1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn+=1

    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model


def flame_analysis(local_model, args, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    if debug==True:
        filename = './' + args.save + '/flame_analysis.txt'
        f = open(filename, "a")
        for i in cos_list:
            f.write(str(i))
            f.write('/n')
        f.write('/n')
        f.write("--------Round--------")
        f.write('/n')
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
    return benign_client

def lbfgs(args, S_k_list, Y_k_list, v):
    curr_S_k = nd.concat(*S_k_list, dim=1)
    curr_Y_k = nd.concat(*Y_k_list, dim=1)
    S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k.asnumpy())
    L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(args.gpu))
    sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = nd.diag(S_k_time_Y_k)
    upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
    mat = nd.concat(*[upper_mat, lower_mat], dim=0)
    mat_inv = nd.linalg.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
    approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def lbfgs_torch(args, S_k_list, Y_k_list, v):
    curr_S_k = torch.stack(S_k_list)
    curr_S_k = curr_S_k.transpose(0, 1).cpu() #(10,xxxxxx)
    print('------------------------')
    print('curr_S_k.shape', curr_S_k.shape)
    curr_Y_k = torch.stack(Y_k_list)
    curr_Y_k = curr_Y_k.transpose(0, 1).cpu() #(10,xxxxxx)
    S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k
    S_k_time_Y_k = S_k_time_Y_k.cpu()


    S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k
    S_k_time_S_k = S_k_time_S_k.cpu()
    print('S_k_time_S_k.shape', S_k_time_S_k.shape)
    R_k = np.triu(S_k_time_Y_k.numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
    sigma_k = Y_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1) / (S_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1))
    sigma_k=sigma_k.cpu()
    
    D_k_diag = S_k_time_Y_k.diagonal()
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = mat.inverse()
    print('mat_inv.shape',mat_inv.shape)
    v = v.view(-1,1).cpu()

    approx_prod = sigma_k * v
    print('approx_prod.shape',approx_prod.shape)
    print('v.shape',v.shape)
    print('sigma_k.shape',sigma_k.shape)
    print('sigma_k',sigma_k)
    p_mat = torch.cat([curr_S_k.transpose(0, 1) @ (sigma_k * v), curr_Y_k.transpose(0, 1) @ v], dim=0)
    
    approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat
    print('approx_prod.shape',approx_prod.shape)

    return approx_prod.T

from torch.utils.data import DataLoader, Subset
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from CrowdGuard.CrowdGuardClientValidation import CrowdGuardClientValidation

def create_cluster_map_from_labels(expected_number_of_labels, clustering_labels):
    """
    Converts a list of labels into a dictionary where each label is the key and
    the values are lists/np arrays of the indices from the samples that received
    the respective label
    :param expected_number_of_labels number of samples whose labels are contained in
    clustering_labels
    :param clustering_labels list containing the labels of each sample
    :return dictionary of clusters
    """
    assert len(clustering_labels) == expected_number_of_labels

    clusters = {}
    for i, cluster in enumerate(clustering_labels):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(i)
    return {index: np.array(cluster) for index, cluster in clusters.items()}


def determine_biggest_cluster(clustering):
    """
    Given a clustering, given as dictionary of the form {cluster_id: [items in cluster]}, the
    function returns the id of the biggest cluster
    """
    biggest_cluster_id = None
    biggest_cluster_size = None
    for cluster_id, cluster in clustering.items():
        size_of_current_cluster = np.array(cluster).shape[0]
        if biggest_cluster_id is None or size_of_current_cluster > biggest_cluster_size:
            biggest_cluster_id = cluster_id
            biggest_cluster_size = size_of_current_cluster
    return biggest_cluster_id

def print_vote_matrix(votes, malicious_list, idxs_users, are_attackers):
    m = votes.shape[0]
    assert len(are_attackers) == len(idxs_users)
    print("\n[CrowdGuard] === Validator Global Mapping ===")
    # idxs_users 是本輪選中的全局用戶 ID 列表，長度 m
    # local j -> global idxs_users[j]
    for j in range(m):
        global_uid = idxs_users[j]
        behavior = "ATTACK" if are_attackers[j] else "NO ATTACK"
        true_type = "MALICIOUS" if global_uid in malicious_list else "BENIGN"
        print(f"Validator {j} → Global User {global_uid}: True={true_type} with {behavior}")
    print("[CrowdGuard] ============================\n")
    print("[CrowdGuard] === Vote Matrix Summary ===")
    header = "Validator (Global UID): " + " ".join(f"{uid:>3}" for uid in idxs_users)
    print(header)
    for j in range(m):
        vote_row = votes[j]
        votes_str = " ".join(f"{v:>3}" for v in vote_row)
        print(f"Validator {j} (Global {idxs_users[j]}): {votes_str}")
    print("[CrowdGuard] ============================\n")

def save_file_vote_matrix(votes, malicious_list, idxs_users, are_attackers, f):
    m = votes.shape[0]
    assert len(are_attackers) == len(idxs_users)
    f.write("\n[CrowdGuard] === Validator Global Mapping ===\n")
    # idxs_users 是本輪選中的全局用戶 ID 列表，長度 m
    # local j -> global idxs_users[j]
    for j in range(m):
        global_uid = idxs_users[j]
        true_type = "MALICIOUS" if global_uid in malicious_list else "BENIGN"
        f.write(f"Validator {j} → Global User {global_uid}: True={true_type}\n")
    f.write("[CrowdGuard] ============================\n\n")
    f.write("[CrowdGuard] === Vote Matrix Summary ===\n")
    for j in range(m):
        f.write(f"Validator {j} (Global {idxs_users[j]}): {votes[j].tolist()}\n")
    f.write("[CrowdGuard] ============================\n\n")

# CrowdGuard defense using the utility functions
def crowdguard(w_updates, global_model_copy, dataset_train, dict_users, idxs_users, malicious_list, are_attackers, args, debug=False):
    if debug:
        print("[CrowdGuard] Running defense with debug info ON")
    m = len(w_updates)

    # === 1) 重建模型 & DataLoader ===
    models, loaders = [], []
    uid_to_model_index = {}
    for local_pos, delta in enumerate(w_updates):
        real_uid = idxs_users[local_pos]
        # Li = Gt + Δi
        model = copy.deepcopy(global_model_copy).to(args.device)
        sd = model.state_dict()
        for k in sd:
            sd[k] = sd[k] + delta[k].to(args.device)
        model.load_state_dict(sd)
        model.eval()
        models.append(model)

        # Di：用 real_uid 取子集
        indices = sorted(list(dict_users[real_uid]))
        subset = Subset(dataset_train, indices)
        #if debug:
        #    print(f"[DEBUG] Client {real_uid} has {len(indices)} samples")
        loaders.append(DataLoader(subset, batch_size=args.local_bs, shuffle=False))

        uid_to_model_index[real_uid] = local_pos

    # === 2) HLBIM 分析 & 投票 ===
    votes = np.zeros((m, m), dtype=int)
    global_model=copy.deepcopy(global_model_copy).to(args.device)
    for uid in idxs_users:
        model_index = uid_to_model_index[uid]
        if debug:
            print(f"[CrowdGuard] [Validator {model_index} → Global User {uid}] start validating against global model")
        poisoned = CrowdGuardClientValidation.validate_models(
            global_model=global_model,
            models=models,
            own_client_index=model_index,
            local_data=loaders[model_index],
            device=args.device,
            debug=debug
        )
        if debug:
            print(f"[CrowdGuard] [Validator {model_index} → Global User {uid}] detected poisoned models: {poisoned} → by Global Users {[idxs_users[i] for i in poisoned]}")
        # Build vote row: 1 for benign (including self), 0 for poisoned
        for i in range(m):
            votes[model_index, i] = 1 if (i == model_index or i not in poisoned) else 0
    # 在 crowdguard() 的 votes 建構後，直接印出對應關係
    if debug:
        print_vote_matrix(votes, malicious_list, idxs_users, are_attackers)

    # === 3) 堆疊式聚类 & 最終投票 ===
    # 3.1 Agglomerative → 選出 majority_validators
    agg_labels = AgglomerativeClustering(n_clusters=2, distance_threshold=None,
                                       compute_full_tree=True,
                                       metric="euclidean", memory=None, connectivity=None,
                                       linkage='single',
                                       compute_distances=True).fit_predict(votes)
    if debug:
        print(f"[CrowdGuard] Agglomerative labels: {agg_labels}")
    agg_map = create_cluster_map_from_labels(m, agg_labels)
    if debug:
        print(f"[CrowdGuard] Agglomerative Clustering: {agg_map}")
    major_label = determine_biggest_cluster(agg_map)
    majority_validators = agg_map[major_label]
    if debug:
        print(f"[CrowdGuard] Validators kept (after clustering) as DBScan Input: {majority_validators}")

    # 3.2 DBSCAN → 從 majority_validators 挑出最穩定的那一群
    sub_votes = votes[majority_validators]
    db_labels = DBSCAN(eps=0.5, min_samples=1).fit_predict(sub_votes)
    if debug:
        print(f"[CrowdGuard] DBSCAN core labels: {db_labels}")
    db_map = create_cluster_map_from_labels(len(majority_validators), db_labels)
    top_label = determine_biggest_cluster(db_map)
    core_indices = db_map[top_label]
    if debug:
        print(f'[CrowdGuard] DBScan Clustering: {core_indices}')
    single_sample_idx = core_indices[0]

    # Final per-model vote
    final_votes = sub_votes[single_sample_idx]
    if debug:
        print(f"[CrowdGuard] Final votes (0=poisoned, 1=benign): {final_votes}")

    # === 4) 過濾 & 回傳 ===
    kept = [i for i, v in enumerate(final_votes) if v==1]
    filtered_updates = [w_updates[i] for i in kept]
    if debug:
        print(f"[CrowdGuard] Kept indices: {kept} → Kept Global Users {[idxs_users[i] for i in kept]}")
        pruned = [i for i in range(m) if i not in kept]
        print(f"[CrowdGuard] Pruned count: {len(pruned)}, Pruned indices: {pruned} → Pruned Global Users {[idxs_users[i] for i in pruned]}")
    with open(f'./{args.save}/crowdguard_log.txt', 'a') as f:
        f.write("=== CrowdGuard Round Info ===\n")
        save_file_vote_matrix(votes, malicious_list, idxs_users, are_attackers, f)
        f.write(f"Validators kept (after clustering) as DBScan Input: {majority_validators}\n")
        f.write(f"Final kept indices: {kept} → Kept Global Users {[idxs_users[i] for i in kept]}\n")
        pruned = [i for i in range(m) if i not in kept]
        f.write(f"Pruned count: {len(pruned)}, Pruned indices: {pruned} → Pruned Global Users {[idxs_users[i] for i in pruned]}\n\n")
    
    import gc
    import torch
    # 清空 GPU 快取
    torch.cuda.empty_cache()
    # 釋放模型和 loader 資源
    del models
    del loaders
    del global_model
    gc.collect()
    return filtered_updates, kept
