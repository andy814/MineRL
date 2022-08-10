#from Agent import Agent
from AgentMultiLabel import Agent

from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import gym
import minerl
import os
from torch.utils.tensorboard import SummaryWriter
from ResidualModules import ResidualModelAdditional,ResidualModel,LabelSmoothing,ResidualModelAdditionalMultiLabel,ResidualModelAdditionalMultiLabelRegression
import torch.nn.functional as F
import copy
import random
import time
from ForestWorld import Forest

from Utils import *
LEARNING_RATE= 5e-5 # avgloss: lower than 1.5
EPOCHS=200 # we use number of updates instead of epochs
MAX_ITER=200_000 # save the model for every MAX_ITER updates

# 390min 100runs
RUNS=10
NBR_CHECKPOINTS=10

#NBR_CHECKPOINTS=1
#RUNS=100

BATCH_SIZE = 16
random.seed(2022)
np.random.seed(2022)
th.manual_seed(2022)

# only used for ObtainDiamond/ObtainIronPickaxe model
def train(model,dataset):
    DATA_DIR = os.getenv('MINERL_DATA_ROOT', "/Users/cody/Code/il-representations/data/minecraft")
    data = minerl.data.make(dataset,  data_dir=DATA_DIR, num_workers=1)
    network=None
    if model=="ResidualModel":
        network=ResidualModel((3, 64, 64)).cuda()
    elif model=="ResidualModelAdditional":
        network=ResidualModelAdditional((3, 64, 64)).cuda()
    elif model=="ResidualModelAdditionalMultiLabel":
        network=ResidualModelAdditionalMultiLabel((3, 64, 64)).cuda()

    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    loss_function=LabelSmoothing(smoothing=0.005)
    # optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    # loss_function=nn.NLLLoss()

    iter_count = 0
    losses = []
    writer = SummaryWriter()
    #file_name=model+"_"+dataset+".pth"
    file_name="obtaindiamond_models/"+model+"_"+dataset+".pth"
    model_nbr=1

    # NBR_CHECKPOINTS=5
    # network=th.load("obtaindiamond_models_1/ResidualModelAdditionalMultiLabel_MineRLObtainDiamond-v0_2000000.pth")
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE, seq_len=1)):
        pov=dataset_obs["pov"]

        # Actions need bit more work
        actions = dataset_action_batch_to_actions_MultiLabael(dataset_actions)
        
        # Remove samples that had no corresponding action
        mask = [True]*len(pov) # we don't filter noop actions for now
        #mask = actions != -1
        pov = pov[mask]
        for key in actions:
            actions[key]=actions[key][mask]
            
        if len(pov)==0:
            continue
        
        for key in dataset_obs["inventory"]:
            dataset_obs["inventory"][key]=dataset_obs["inventory"][key][mask]
        inventory=dataset_obs["inventory"]
        damage=dataset_obs['equipped_items.mainhand.damage'][mask]
        max_damage=dataset_obs['equipped_items.mainhand.maxDamage'][mask]
        item_type=dataset_obs['equipped_items.mainhand.type'][mask]

        dic={
            "pov":pov,
            "inventory":inventory,
            "damage":damage,
            "max_damage":max_damage,
            "item_type":item_type
        }

        output = network(dic)
        loss = criterion(loss_function, output, actions ) # for cross_entropy/NLL

        for key in dic.keys(): # free memory
            dic[key]=[]
        dic.clear()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_count += 1
        losses.append(loss.item())
        
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()
            writer.add_scalar('Loss/train', mean_loss, iter_count)

        if (iter_count % 10000) == 0: # checkpoint
            th.save(network, file_name)

        if iter_count>=MAX_ITER*model_nbr:
            model_nbr+=1
            th.save(network, "obtaindiamond_models/"+model+"_"+dataset+"_"+str(iter_count)+".pth")

        if model_nbr>NBR_CHECKPOINTS:
            break
            
    th.save(network, file_name)
    del data

# only used for multi-label agent on obtiandiamond/ironpickaxe
def train_val(model,dataset): 
    DATA_DIR = os.getenv('MINERL_DATA_ROOT', "/Users/cody/Code/il-representations/data/minecraft")
    data = minerl.data.make(dataset,  data_dir=DATA_DIR, num_workers=1)
    network=None
    if model=="ResidualModel":
        network=ResidualModel((3, 64, 64)).cuda()
    elif model=="ResidualModelAdditional":
        network=ResidualModelAdditional((3, 64, 64)).cuda()
    elif model=="ResidualModelAdditionalMultiLabel":
        network=ResidualModelAdditionalMultiLabel((3, 64, 64)).cuda()
    elif model=="ResidualModelAdditionalMultiLabelRegression":
        network=ResidualModelAdditionalMultiLabelRegression((3, 64, 64)).cuda()

    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    loss_function=LabelSmoothing(smoothing=0.005)
    iter_count = 0
    train_losses = []
    val_losses = []

    writer = SummaryWriter()
    file_name=model+"_"+dataset+".pth"
    
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE, seq_len=1)):
        pov=dataset_obs["pov"]
        actions = dataset_action_batch_to_actions_MultiLabael(dataset_actions,regression=False)
        mask = [True]*len(pov) # we don't filter noop actions for now
        
        val_num=4
        train_mask=mask[val_num:]
        val_mask=mask[:val_num]
        if sum(train_mask)==0 or sum(val_mask)==0:
            continue

        #pov=data_augmentation1(pov)

        pov_train = pov[val_num:][train_mask]
        pov_val = pov[:val_num][val_mask]

        actions_train=actions
        actions_val=copy.deepcopy(actions_train)
        for key in actions_train:
            actions_train[key]=actions_train[key][val_num:][train_mask]
        for key in actions_val:
            actions_val[key]=actions_val[key][:val_num][val_mask]

        train_inventory_dict=dataset_obs["inventory"]
        val_inventory_dict=copy.deepcopy(train_inventory_dict)
        for key in train_inventory_dict:
            train_inventory_dict[key]=train_inventory_dict[key][val_num:][train_mask]
        inventory_train=train_inventory_dict
        damage_train=dataset_obs['equipped_items.mainhand.damage'][val_num:][train_mask]
        max_damage_train=dataset_obs['equipped_items.mainhand.maxDamage'][val_num:][train_mask]
        item_type_train=dataset_obs['equipped_items.mainhand.type'][val_num:][train_mask]
        dic_train={
            "pov":pov_train,
            "inventory":inventory_train,
            "damage":damage_train,
            "max_damage":max_damage_train,
            "item_type":item_type_train
        }

        for key in val_inventory_dict:
            val_inventory_dict[key]=val_inventory_dict[key][:val_num][val_mask]
        inventory_val=val_inventory_dict
        damage_val=dataset_obs['equipped_items.mainhand.damage'][:val_num][val_mask]
        max_damage_val=dataset_obs['equipped_items.mainhand.maxDamage'][:val_num][val_mask]
        item_type_val=dataset_obs['equipped_items.mainhand.type'][:val_num][val_mask]
        dic_val={
            "pov":pov_val,
            "inventory":inventory_val,
            "damage":damage_val,
            "max_damage":max_damage_val,
            "item_type":item_type_val
        }

        output_train = network(dic_train)
        loss_train = criterion(loss_function, output_train, actions_train,regression=False) # for cross_entropy/NLL
        output_val = network(dic_val)
        loss_val = criterion(loss_function, output_val, actions_val, regression=False) # for cross_entropy/NLL

        # free memory
        if 'inventory_train' in dic_train:
            for key in dic_train['inventory_train']:
                dic_train['inventory_train'][key]=[]
        if 'inventory_val' in dic_val:
            for key in dic_val['inventory_val']:
                dic_val['inventory_val'][key]=[]
        for key in dic_train.keys(): 
           dic_train[key]=[]
        dic_train.clear()
        for key in dic_val.keys():
            dic_val[key]=[]
        dic_val.clear()
        del dic_train,dic_val
        for key in actions_train.keys(): 
           actions_train[key]=[]
        actions_train.clear()
        for key in actions_val.keys():
            actions_val[key]=[]
        actions_val.clear()
        del actions_train,actions_val
        for arr in [pov_train,pov_val,pov,actions,mask,train_mask,val_mask]:
           arr=[]

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_count += 1
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        
        if (iter_count % 1000) == 0:
            mean_loss_train = sum(train_losses) / len(train_losses)
            mean_loss_val = sum(val_losses) / len(val_losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}. Val Loss {:<10.3f}. ".format(iter_count, mean_loss_train, mean_loss_val))
            
            train_losses.clear()
            val_losses.clear()

            writer.add_scalar('Loss/train', mean_loss_train, iter_count)
            writer.add_scalar('Loss/val', mean_loss_val, iter_count)

        if (iter_count % 10000) == 0: # checkpoint
            th.save(network, file_name)

    th.save(network, file_name)
    del data


def enjoy():
    file_name_treechop="treechop_models/ResidualModel_MineRLTreechop-v0_400000.pth"
    network_treechop = th.load(file_name_treechop).cuda()

    #forest=Forest()
    #forest.register()
    #env2=gym.make("ForestWorld-v0")
    env2=gym.make("MineRLObtainDiamond-v0")
    env = ActionShaping(env2, always_attack=True)
    MultiLabel=ActionShapingMultiLabelNonWrapper(env2,always_attack=True,regression=False,remove_uncommon=True)

    avg_reward=0
    
    for i in range(1,NBR_CHECKPOINTS+1):
        # model_nbr=i*MAX_ITER
        model_nbr=1800000
        file_name_multiLabel="obtaindiamond_models/ResidualModelAdditionalMultiLabel_MineRLObtainDiamond-v0_"+str(model_nbr)+".pth"
        network_multiLabel = th.load(file_name_multiLabel).cuda()
        agent=Agent(network_treechop,network_multiLabel,env,env2,stuck_threshold=20,use_treechop=True,use_script=True)
        
        avg_reward=0
        rewards=[]
        progresses=[]
        for game_i in range(RUNS):
            obs = env.reset()
            done = False
            reward_sum = 0
            agent.clear()
            while not done: 
                action,action_type=agent.act(obs)
                if action_type==0:
                    obs, reward, done, info = env2.step(action)
                elif action_type==1:
                    obs, reward, done, info = env.step(action)
                else:
                    action=MultiLabel.action(action)
                    obs, reward, done, info = env2.step(action)
                reward_sum += reward
                agent.update(reward)
                env.render()
            avg_reward+=reward_sum
            rewards.append(reward_sum)
            if reward_sum>=1024:
                agent.progress+=1
            progresses.append(agent.progress)

        avg_reward/=RUNS
        print("average reward for",model_nbr,":",avg_reward)
        print(rewards)
        print(progresses)
    #env2.close()

def train_treechop(model,dataset):
    DATA_DIR = os.getenv('MINERL_DATA_ROOT', "/Users/cody/Code/il-representations/data/minecraft")
    data = minerl.data.make(dataset,  data_dir=DATA_DIR, num_workers=1)
    network=None
    if model=="ResidualModel":
        network=ResidualModel((3, 64, 64)).cuda()

    # optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # loss_function=LabelSmoothing(smoothing=0.005)
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function=nn.NLLLoss()

    iter_count = 0
    losses = []
    writer = SummaryWriter()
    file_name="treechop_models/"+model+"_"+dataset+".pth"
    model_nbr=1
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs=EPOCHS, batch_size=BATCH_SIZE, seq_len=1)):
        
        pov=dataset_obs["pov"]

        # Actions need bit more work
        actions = dataset_action_batch_to_actions(dataset_actions)
        
        # Remove samples that had no corresponding action
        #mask = [True]*len(pov) # we don't filter noop actions for now
        mask = actions != -1
        pov = pov[mask]
        actions=actions[mask]
            
        if len(pov)==0:
            continue

        dic={}
        dic["pov"]=dataset_obs["pov"][mask]
        
        logits = network(dic)
        loss = loss_function(logits, th.from_numpy(actions).long().cuda()) # for cross_entropy/NLL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_count += 1
        losses.append(loss.item())
        
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()
            writer.add_scalar('Loss/train', mean_loss, iter_count)

        if (iter_count % 10000) == 0: # checkpoint
            th.save(network, file_name)
        
        if iter_count>=MAX_ITER*model_nbr:
            model_nbr+=1
            th.save(network, "treechop_models/"+model+"_"+dataset+"_"+str(iter_count)+".pth")

        if model_nbr>NBR_CHECKPOINTS:
            break
            
    th.save(network, file_name)
    del data

def enjoy_treechop():
    env2=gym.make("MineRLTreechop-v0")
    env = ActionShaping(env2, always_attack=True)
    action_list = np.arange(7)
    for i in range(1,NBR_CHECKPOINTS+1):
        model_nbr=i*MAX_ITER
        file_name_treechop="treechop_models/ResidualModel_MineRLTreechop-v0_"+str(model_nbr)+".pth"
        network_treechop = th.load(file_name_treechop).cuda()
        avg_reward=0
        rewards=[]
        for game_i in range(RUNS):
            obs = env.reset()
            done = False
            reward_sum = 0
            while not done: 
                probabilities = th.exp(network_treechop(obs))[0] # for KL_DIV/NLL
                probabilities /= probabilities.sum()
                probabilities = probabilities.detach().cpu().numpy()
                action = np.random.choice(action_list, p=probabilities)
                obs, reward, done, info = env.step(action)
                reward_sum += reward
                env.render()
            avg_reward+=reward_sum
            rewards.append(reward_sum)

        avg_reward/=RUNS
        print("average reward for",model_nbr,":",avg_reward)
        print(rewards)
    
    #env2.close()

if __name__ == "__main__":    
    train("ResidualModelAdditionalMultiLabel","MineRLObtainDiamond-v0")
    train_val("ResidualModelAdditionalMultiLabelRegression","MineRLObtainDiamond-v0")
    enjoy()
    train_treechop("ResidualModel","MineRLTreechop-v0")
    enjoy_treechop() 