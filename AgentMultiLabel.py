import time
from git import refresh
from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import gym
import minerl
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from ResidualModules import onehot_mapping
from collections import deque
from enum import Enum

from Utils import *
import random

class Agent:

    @staticmethod
    def equip(item):
        action_sequence = []
        action_sequence += ['equip:'+item]
        return iter(action_sequence)

    @staticmethod
    def wood_dig():
        """
        Specify the action sequence for the scripted part of the agent.
        """
        action_sequence = []
        action_sequence += ['camera:[0,10]'] * 18
        action_sequence += ['forward']*100
        action_sequence += ['camera:[-10,0]'] * 18
        action_sequence += ['attack'] * 20
        action_sequence += ['craft:planks'] * 3
        action_sequence += ['craft:stick'] * 1
        action_sequence += ['craft:crafting_table']
        action_sequence += ['camera:[10,0]'] * 18
        action_sequence += ['attack'] * 20
        action_sequence += [''] * 10
        action_sequence += ['jump']
        action_sequence += [''] * 5
        action_sequence += ['place:crafting_table']
        action_sequence += [''] * 10

        action_sequence += ['camera:[-1,0]']
        action_sequence += ['nearbyCraft:wooden_pickaxe']
        action_sequence += ['camera:[1,0]']
        action_sequence += [''] * 10
        action_sequence += ['equip:wooden_pickaxe']
        action_sequence += [''] * 10
        action_sequence += ['camera:[1,0]']
        
        action_sequence += ['attack'] * 100
        action_sequence += ['forward']*4
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*4
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*4
        action_sequence += ['attack'] * 400 # 400 for 13
        action_sequence += ['camera:[-10,0]']*9

        return iter(action_sequence)

    @staticmethod
    def stone_dig():
        """
        Specify the action sequence for the scripted part of the agent.
        """
        action_sequence = []
        action_sequence += ['forward']*100
        action_sequence += ['camera:[-10,0]'] * 18
        action_sequence += ['attack'] * 20
        action_sequence += ['craft:stick'] * 2
        action_sequence += ['camera:[10,0]'] * 18
        action_sequence += ['attack'] * 20
        action_sequence += [''] * 10
        action_sequence += ['jump']
        action_sequence += [''] * 5
        action_sequence += ['place:crafting_table']
        action_sequence += [''] * 10

        action_sequence += ['camera:[-1,0]']
        action_sequence += ['nearbyCraft:stone_pickaxe']
        action_sequence += ['camera:[1,0]']
        action_sequence += [''] * 10
        action_sequence += ['equip:stone_pickaxe']
        action_sequence += [''] * 10
        action_sequence += ['camera:[1,0]']

        action_sequence += ['attack'] * 100
        action_sequence += ['forward']*4
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*4
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*4
        action_sequence += ['attack'] * 350 # 350 for 20

        action_sequence += ['jump']
        action_sequence += [''] * 5
        action_sequence += ['place:crafting_table']
        action_sequence += ['camera:[-1,0]']
        action_sequence += ['craft:planks'] * 3
        action_sequence += ['craft:stick'] * 5
        action_sequence += ['nearbyCraft:stone_pickaxe']*1
        action_sequence += ['attack'] * 100

        action_sequence += ['camera:[-10,0]']*9

        return iter(action_sequence)

    @staticmethod
    def iron_dig():
        """
        Specify the action sequence for the scripted part of the agent.
        """
        action_sequence = []
        action_sequence += ['forward']*100
        action_sequence += ['camera:[-10,0]'] * 18
        action_sequence += ['attack'] * 20
        action_sequence += ['craft:stick'] * 1
        action_sequence += ['camera:[10,0]'] * 18
        action_sequence += ['attack'] * 20
        action_sequence += [''] * 10
        action_sequence += ['jump']
        action_sequence += [''] * 5
        action_sequence += ['place:crafting_table']
        action_sequence += [''] * 10

        action_sequence += ['camera:[-1,0]']
        action_sequence += ['nearbyCraft:furnace']
        action_sequence += ['jump']
        action_sequence += [''] * 5
        action_sequence += ['place:furnace']
        action_sequence += ['nearbySmelt:iron_ingot']*3
        action_sequence += ['nearbyCraft:iron_pickaxe']
        action_sequence += [''] * 10

        action_sequence += ['camera:[1,0]']
        action_sequence += [''] * 10
        action_sequence += ['equip:iron_pickaxe']
        action_sequence += [''] * 10
        action_sequence += ['camera:[1,0]']

        #action_sequence += ['attack'] * 500
        action_sequence += ['attack'] * 100
        action_sequence += ['forward']*4
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*4
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*8
        action_sequence += ['camera:[0,10]']*9
        action_sequence += ['forward']*4
        action_sequence += ['attack'] * 250 # 30% faster?
        action_sequence += ['camera:[-10,0]']*9

        return iter(action_sequence)

    @staticmethod
    def get_out(strategy=0):
        action_sequence = []
        if strategy==0: # turnAround, face horizontally
            action_sequence += ['camera:[0,10]'] * 18
            action_sequence += ['camera:[-10,0]'] * 9
            action_sequence += ['camera:[10,0]'] * 6

        elif strategy==1: # random turn
            yaw=random.randrange(6,31)
            pitch=random.randrange(3,7)
            action_sequence += ['camera:[0,10]'] * yaw
            action_sequence += ['camera:[10,0]'] * 18
            action_sequence += ['camera:[-10,0]'] * pitch

        else: # place some stones
            action_sequence += ['forward']*100
            steps=2
            tries=2
            for i in range(tries):
                action_sequence += ['camera:[-10,0]'] * 18
                action_sequence += ['attack'] * 100
                action_sequence += ['camera:[10,0]'] * 18
                for j in range(steps):
                    action_sequence += ['jump']
                    action_sequence += [''] * 5
                    action_sequence += ['place:cobblestone']
                    action_sequence += [''] * 5 
            action_sequence += ['camera:[-10,0]'] * 9

        return iter(action_sequence)

    scriptDict={
        "wood_dig":wood_dig.__func__,
        "stone_dig":stone_dig.__func__,
        "iron_dig":iron_dig.__func__,
        "get_out":get_out.__func__,
        "equip":equip.__func__
    }

    progressItems=["log","planks","crafting_table","wooden_pickaxe",
                    "cobblestone","stone_pickaxe","iron_ore","iron_pickaxe"]
    
    class Stage(Enum): # stage= level of current item
        START= 0
        WOOD = 1 # once we get enough logs
        STONE = 2 # once we get enough cobblestones
        IRON = 3 # once we get enough iron

    def __init__(self,network_treechop,network_multiLabel,
                env,env2,stuck_threshold=5,refresh_threshold=1,num_actions=7,use_script=True,use_treechop=True): # second(s)
        self.network_treechop=network_treechop
        #self.network_iron=network_iron
        #self.network_diamond=network_diamond
        self.network_multiLabel=network_multiLabel
        self.env=env
        self.env2=env2 # unwrapped environment
        self.stuck_threshold=stuck_threshold
        self.num_actions=num_actions
        self.refresh_threshold=refresh_threshold
        self.inventory_max_len=stuck_threshold//refresh_threshold
        self.use_script=use_script
        self.use_treechop=use_treechop

        self.progress=-1
        self.getout_timer=0
        #self.dig_timer=0
        self.refresh_timer=0
        self.prev_inventory=deque()
        self.curr_inventory=None
        self.prev_time=time.time()
        self.script_iterator=None
        self.mainhand="air"

        self.stage=self.Stage.START 
        self.script=None

    def act(self,obs): # act using pov and additional features
        self.curr_inventory=obs["inventory"]
        self.mainhand=obs["equipped_items"]["mainhand"]["type"]
        obs=decode_obs(obs)
        #pov=obs['pov']
        #pov = th.from_numpy(pov.transpose(2, 0, 1)[None].astype(np.float32) / 255).cuda()
        action_list = np.arange(self.num_actions)
        action_type=0 # 0 1 2 for no_script, single_label_output,multi_label_output
        if self.script and self.use_script: # perform the script actions
            try:
                action=str_to_act(self.env2,next(self.script_iterator))
            except StopIteration:
                if self.script=="get_out":
                    self.exit_get_out()
                elif self.script=="equip":
                    self.exit_equip_item()
                elif self.script=="wood_dig":
                    self.exit_wood_dig()
                elif self.script=="stone_dig":
                    self.exit_stone_dig()
                elif self.script=="iron_dig":
                    self.exit_iron_dig()
                action=self.env2.action_space.noop()
            action_type=0
        else: # We can use multiple networks to solve different tasks
            #if None:
            if self.stage==self.Stage.START and self.use_treechop: # Treechop model for Treechop task
                #probabilities = th.softmax(self.network_treechop(obs), dim=1)[0] # for Cross_Entropy
                probabilities = th.exp(self.network_treechop(obs))[0] # for KL_DIV/NLL
                probabilities /= probabilities.sum()
                probabilities = probabilities.detach().cpu().numpy()
                action = np.random.choice(action_list, p=probabilities)
                action_type=1
            else:
                action_dict=self.network_multiLabel(obs)
                action=dict()
                for key in action_dict:
                    action_list = np.arange(len(action_dict[key][0]))
                    probabilities=th.exp(action_dict[key])[0]
                    probabilities /= probabilities.sum()
                    probabilities = probabilities.detach().cpu().numpy()
                    action[key] = np.random.choice(action_list, p=probabilities)
                    # Sample action according to the probabilities
                action_type=2

        return action,action_type

    def clear(self): # clear timer

        self.getout_timer=0
        self.refresh_timer=0
        self.script_iterator=None
        self.script=None
        self.prev_inventory=deque()
        self.curr_inventory=None
        self.prev_time=time.time()
        self.stage=self.Stage.START
        #self.stage=self.Stage.WOOD
        self.mainhand="air"
        self.progress=-1

    def update(self,reward=-1):
        self.refresh_timer+=time.time()-self.prev_time
        for i,item in enumerate(Agent.progressItems):
            if self.curr_inventory[item]>0 and self.progress<i:
                self.progress=i

        if self.refresh_timer>=self.refresh_threshold: # trigger refresh
            self.prev_inventory.append(self.curr_inventory)
            if len(self.prev_inventory)>self.inventory_max_len:
                self.prev_inventory.popleft()
            self.refresh_timer=0
        self.getout_timer+=time.time()-self.prev_time
        inventory_changed=False
        if self.getout_timer>=self.stuck_threshold:
            for key in self.prev_inventory[-1]:
                if self.prev_inventory[-1][key]!=self.prev_inventory[0][key]:
                    inventory_changed=True
                    break
            if not inventory_changed:
                self.enter_get_out()
            else:
                self.getout_timer=0
        self.check_item()
        self.check_stage()
        self.prev_time=time.time()

    def check_stage(self):
        if self.stage==self.Stage.START:
            if np.squeeze(self.curr_inventory["log"])>=10:
                if self.use_script:
                    self.enter_wood_dig()
                else:
                    self.stage=Agent.Stage.WOOD
            
        elif self.stage==self.Stage.WOOD:
            if np.squeeze(self.curr_inventory["cobblestone"])>=10:
                self.enter_stone_dig()
        elif self.stage==self.Stage.STONE:       
            if np.squeeze(self.curr_inventory["iron_ore"])>=3 and np.squeeze(self.curr_inventory["coal"])>=3:
                self.enter_iron_dig()

    def check_item(self):
        pickaxes=["iron_pickaxe","stone_pickaxe","wooden_pickaxe"]
        if self.mainhand not in set( pickaxes ):
            for pickaxe in pickaxes:
                if np.squeeze(self.curr_inventory[pickaxe])>=1:
                    self.enter_equip_item(pickaxe)
                    return

    def enter_get_out(self,strategy=0): # perform get_out script
        # print("enter_get_out")
        # print(self.curr_inventory)
        # print(self.mainhand)
        self.getout_timer=0
        if self.script: # you can only be in one script at a time
            return
        self.script="get_out"
        if np.squeeze(self.curr_inventory["cobblestone"])>=20:
        #if np.squeeze(self.curr_inventory["cobblestone"])>=200:
            self.script_iterator=Agent.scriptDict[self.script](2)
        elif self.stage==self.Stage.WOOD or self.stage==self.Stage.STONE or self.stage==self.Stage.IRON:
            self.script_iterator=Agent.scriptDict[self.script](1)
        else:
            self.script_iterator=Agent.scriptDict[self.script](0)

    def exit_get_out(self): 
        # print("exit_get_out")
        # print(self.curr_inventory)
        # print(self.mainhand)
        self.getout_timer=0
        self.script=None
        self.script_iterator=None

    def enter_equip_item(self,item):
        # print("equipping:",item)
        if self.script and self.script!="get_out": # you can interrupt get_out script
            return
        self.script="equip"
        self.script_iterator=Agent.scriptDict[self.script](item)

    def exit_equip_item(self):
        # print("equipped")
        self.script=None
        self.script_iterator=None


    def enter_wood_dig(self): # perform craft script
        # print("enter_wood_dig")
        if self.script and self.script!="get_out": # you can interrupt get_out script
            return
        self.script="wood_dig"
        self.script_iterator=Agent.scriptDict[self.script]()
        self.stage=self.Stage.WOOD

    def exit_wood_dig(self):
        # print("exit_wood_dig")
        # print(self.curr_inventory)
        # print(self.mainhand)
        self.script=None
        self.script_iterator=None

    def enter_stone_dig(self): # perform craft script
        # print("enter_stone_dig")
        if self.script and self.script!="get_out": # you can interrupt get_out script
            return
        self.script="stone_dig"
        self.script_iterator=Agent.scriptDict[self.script]()
        self.stage=self.Stage.STONE

    def exit_stone_dig(self):
        # print("exit_stone_dig")
        # print(self.curr_inventory)
        # print(self.mainhand)
        self.script=None
        self.script_iterator=None

    def enter_iron_dig(self): # perform craft script
        # print("enter_iron_dig")
        if self.script and self.script!="get_out": # you can interrupt get_out script
            return
        self.script="iron_dig"
        self.script_iterator=Agent.scriptDict[self.script]()
        self.stage=self.Stage.IRON

    def exit_iron_dig(self):
        # print("exit_iron_dig")
        # print(self.curr_inventory)
        # print(self.mainhand)
        self.script=None
        self.script_iterator=None
        