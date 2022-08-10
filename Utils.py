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
import random
# we use ordinal encoding for enum type actions. The order follows the one in MineRL documentation
craft_options=["crafting_table","none","planks","stick","torch"]
equip_options=["air","iron_axe","iron_pickaxe","none","stone_axe","stone_pickaxe","wooden_axe","wooden_pickaxe"]
nearbyCraft_options=["furnace","iron_axe","iron_pickaxe","none","stone_axe","stone_pickaxe","wooden_axe","wooden_pickaxe"]
nearbySmelt_options=["coal","iron_ingot","none"]
place_options=["cobblestone","crafting_table","dirt","furnace","none","stone","torch"]

class PovOnlyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']

class ActionShaping(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            #             [('back', 1)],
            #             [('left', 1)],
            #             [('right', 1)],
            #             [('jump', 1)],
            #             [('forward', 1), ('attack', 1)],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],

        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action): # given parameter action, return the action in the action space
        return self.actions[action]

# this class inherited from ActionWrapper won't work
class ActionShapingMultiLabel(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack

        self.action_space = gym.spaces.Tuple((
        gym.spaces.Tuple((gym.spaces.Discrete(3),gym.spaces.Discrete(3))),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(len(craft_options)),
        gym.spaces.Discrete(len(equip_options)),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(len(nearbyCraft_options)),
        gym.spaces.Discrete(len(nearbySmelt_options)),
        gym.spaces.Discrete(len(place_options)),
        gym.spaces.Discrete(2),
        gym.spaces.Discrete(2)))

        #self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action): # given parameter action, return the action in the action space
        ret=self.env.action_space.noop()

        ret["camera"]=[(np.squeeze(action["pitch"])-1)*self.camera_angle,(np.squeeze(action["yaw"])-1)*self.camera_angle]
        ret["attack"]=np.squeeze(action["attack"])
        ret["craft"]=np.squeeze(craft_options[action["craft"]])
        ret["equip"]=np.squeeze(equip_options[action["equip"]])
        ret["moveForward"]=np.squeeze(action["moveForward"])
        ret["jump"]=np.squeeze(action["jump"])
        ret["nearbyCraft"]=np.squeeze(nearbyCraft_options[action["nearbyCraft"]])
        ret["nearbySmelt"]=np.squeeze(nearbySmelt_options[action["nearbySmelt"]])
        ret["place"]=np.squeeze(place_options[action["place"]])
        ret["sneak"]=np.squeeze(action["sneak"])
        ret["sprint"]=np.squeeze(action["sprint"])
        
        return ret

# we define our own wrapper (without inheritance)
class ActionShapingMultiLabelNonWrapper():
    def __init__(self, env, camera_angle=10, always_attack=True, remove_uncommon=True, regression=False):
        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self.env=env
        self.regression=regression
        self.remove_uncommon=remove_uncommon

    def action(self, action): # given parameter action, return the action in the action space
        ret=self.env.action_space.noop()

        if self.regression:
            ret["camera"]=[np.squeeze(action["pitch"]),np.squeeze(action["yaw"])]
            ret["camera"]=np.array(ret["camera"]).astype(np.float32)
            #print(ret["camera"])
        else:
            ret["camera"]=[(np.squeeze(action["pitch"])-1)*self.camera_angle,(np.squeeze(action["yaw"])-1)*self.camera_angle]
            ret["camera"]=np.array(ret["camera"]).astype(np.float32)

        ret["attack"]=action["attack"]
        ret["craft"]=craft_options[action["craft"]]
        ret["equip"]=equip_options[action["equip"]]
        ret["forward"]=action["moveForward"]
        ret["jump"]=action["jump"]
        ret["nearbyCraft"]=nearbyCraft_options[action["nearbyCraft"]]
        ret["nearbySmelt"]=nearbySmelt_options[action["nearbySmelt"]]
        ret["place"]=place_options[action["place"]]
        ret["sneak"]=action["sneak"]
        ret["sprint"]=action["sprint"]

        if self.always_attack:
            ret["attack"]=1
        if self.remove_uncommon:
            if ret["nearbyCraft"] in set(["iron_axe","stone_axe","wooden_axe"]):
                ret["nearbyCraft"]="none"
            if ret["equip"]=="air":
                ret["equip_options"]="none"
        return ret

def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=np.int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first)
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = -1
    return actions


def dataset_action_batch_to_actions_MultiLabael(dataset_actions, camera_margin=5,regression=False): # also return np.array
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    # for binary actions, we does not change the meaning presented in the original training data
    # for camera, we use 0 1 2 for -10,0,10

    attack_actions = dataset_actions["attack"].squeeze()
    camera_actions = dataset_actions["camera"].squeeze()
    craft_actions = dataset_actions["craft"].squeeze()
    equip_actions = dataset_actions["equip"].squeeze()
    moveForward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    nearbyCraft_actions = dataset_actions["nearbyCraft"].squeeze()
    nearbySmelt_actions = dataset_actions["nearbySmelt"].squeeze()
    place_actions = dataset_actions["place"].squeeze()
    sneak_actions = dataset_actions["sneak"].squeeze()
    sprint_actions = dataset_actions["sprint"].squeeze()

    batch_size = len(camera_actions)
    #actions = np.zeros((batch_size,), dtype=np.int)
    
    # actions=dict(key=str,value=(batch_size,))
    actions={   
        "attack":[0]*batch_size,
        "pitch":[1]*batch_size, # 0 1 2 for -10,0,10
        "yaw":[1]*batch_size,
        "craft":[0]*batch_size,
        "equip":[0]*batch_size,
        "moveForward":[0]*batch_size,
        "jump":[0]*batch_size,
        "nearbyCraft":[0]*batch_size,
        "nearbySmelt":[0]*batch_size,
        "place":[0]*batch_size,
        "sneak":[0]*batch_size,
        "sprint":[0]*batch_size
    }

    for i in range(batch_size):
        if regression: 
            actions["pitch"][i]=camera_actions[i][0]
            actions["yaw"][i]=camera_actions[i][1]
        else:
            # Moving camera is most important (horizontal first)
            if camera_actions[i][0] < -camera_margin:
                actions["pitch"][i] = 0
            if camera_actions[i][0] > camera_margin:
                actions["pitch"][i] = 2
            if camera_actions[i][1] > camera_margin:
                #actions["yaw"][i] = 0
                actions["yaw"][i] = 2
            if camera_actions[i][1] < -camera_margin:
                #actions["yaw"][i] = 2
                actions["yaw"][i] = 0

        actions["attack"][i]=attack_actions[i]
        actions["craft"][i]=craft_options.index(craft_actions[i])
        actions["equip"][i]=equip_options.index(equip_actions[i])
        actions["moveForward"][i]=moveForward_actions[i]
        actions["jump"][i]=jump_actions[i]
        actions["nearbyCraft"][i]=nearbyCraft_options.index(nearbyCraft_actions[i])
        actions["nearbySmelt"][i]=nearbySmelt_options.index(nearbySmelt_actions[i])
        actions["place"][i]=place_options.index(place_actions[i])
        actions["sneak"][i]=sneak_actions[i]
        actions["sprint"][i]=sprint_actions[i]

    for key in actions:
        actions[key]=np.array(actions[key])
    #print(actions)
    #print(camera_actions)
    #print(actions["pitch"],actions["yaw"])
    return actions


def str_to_act(env, actions):
    """
    Simplifies specifying actions for the scripted part of the agent.
    Some examples for a string with a single action:
        'craft:planks'
        'camera:[10,0]'
        'attack'
        'jump'
        ''
    There should be no spaces in single actions, as we use spaces to separate actions with multiple "buttons" pressed:
        'attack sprint forward'
        'forward camera:[0,10]'

    :param env: base MineRL environment.
    :param actions: string of actions.
    :return: dict action, compatible with the base MineRL environment.
    """
    act = env.action_space.noop()
    for action in actions.split():
        if ":" in action:
            k, v = action.split(':')
            if k == 'camera':
                act[k] = eval(v)
            else:
                act[k] = v
        else:
            act[action] = 1
    return act

def decode_obs(obs):
    pov=obs["pov"]
    #pov=np.expand_dims(pov,axis=0)
    inventory=obs["inventory"]
    damage=obs["equipped_items"]["mainhand"]["damage"]
    max_damage=obs["equipped_items"]["mainhand"]["maxDamage"]
    item_type=obs["equipped_items"]["mainhand"]["type"]

    for key in inventory:
        inventory[key]=np.expand_dims(np.array([inventory[key]]),axis=0)
    damage=np.expand_dims(np.array([damage]),axis=0)
    max_damage=np.expand_dims(np.array([max_damage]),axis=0)
    item_type=np.expand_dims(np.array([item_type]),axis=0)

    dic={
        "pov":pov,
        "inventory":inventory,
        "damage":damage,
        "max_damage":max_damage,
        "item_type":item_type
    }
    return dic

def criterion(loss_func,outputs,labels,regression=False):
    losses = 0
    for key in outputs:
        # we use MSE for regression
        if (key=="pitch" or key=="yaw") and regression: 
            mse=nn.MSELoss()
            losses+=mse(th.squeeze(outputs[key]),th.from_numpy(labels[key]).float().cuda())
        else:
            losses += loss_func(outputs[key],th.from_numpy(labels[key]).long().cuda())
    return losses

def data_augmentation1(pov): # very very slow, not used
    pov=np.squeeze(pov).astype(np.float32)
    b,h,w,c=pov.shape
    for i in range(b):
        pov[i]/=255
        brightness=0.02*np.random.rand()
        pov[i]+=brightness
        for j in range(h):
            for k in range(w):
                #noise=np.random.normal(0,0.02)
                noise=np.random.rand()
                pov[i][j][k]+=noise
                for m in range(c):
                    strength=1+0.02*np.random.rand() # strengeth=0.02
                    pov[i][j][k][m]*=strength
    return pov

def data_augmentation2(pov): # very slow, not used
    pov=np.squeeze(pov).astype(np.float32)
    b,h,w,c=pov.shape
    for i in range(b):
        pov[i]/=255
        brightness=0.02*random.random()
        pov[i]+=brightness
        for j in range(h):
            for k in range(w):
                noise=0.02*random.random()
                pov[i][j][k]+=noise
    return pov