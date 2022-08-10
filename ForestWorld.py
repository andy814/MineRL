from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List

Forest_DOC = 'ForestWorld-v0'

# testing environment

#Forest_LENGTH = 8000
Forest_LENGTH = 50000

class Forest(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'ForestWorld-v0'

        super().__init__(*args,
                        max_episode_steps=Forest_LENGTH, reward_threshold=1024.0,
                        **kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.BiomeGenerator("forest"),
            handlers.DrawingDecorator("""
                <DrawCuboid x1="0" y1="0" z1="0" x2="0" y2="0" z2="0" type="dirt"/>
            """)
        ]

    def create_agent_start(self) -> List[Handler]:
        return [
            # make the agent start with these items
            handlers.SimpleInventoryAgentStart([
            # dict(type="stone_axe", quantity=1),

            # {'type':'iron_pickaxe', 'quantity':50},
            # {'type':'planks', 'quantity':10},
            # {'type':'stick', 'quantity':10},
            # {'type':'log', 'quantity':20},
            # {'type':'iron_ore', 'quantity':4},
            # {'type':'iron_ingot', 'quantity':4},
            # {'type':'coal', 'quantity':4},
            # {'type':'stone', 'quantity':10},
            # {'type':'cobblestone', 'quantity':30}
                
            ]),
            # make the agent start 90 blocks high in the air
            #handlers.AgentStartPlacement(0, 0, 0, 0, 0)
        ]

    def create_rewardables(self) -> List[Handler]:
        return [
            # reward the agent for touching a certain item (but only once)
            handlers.RewardForCollectingItemsOnce([
                # {'type':'gold_block', 'behaviour':'onceOnly', 'reward':'50'},
                {'amount':1,'reward':1,'type':"log" },
                {'amount':1,'reward':2,'type':"planks" },
                {'amount':1,'reward':4,'type':"stick" },
                {'amount':1,'reward':4,'type':"crafting_table" },
                {'amount':1,'reward':8,'type':"wooden_pickaxe" },
                {'amount':1,'reward':16,'type':"cobblestone" },
                {'amount':1,'reward':32,'type':"furnace" },
                {'amount':1,'reward':32,'type':"stone_pickaxe" },
                {'amount':1,'reward':64,'type':"iron_ore" },
                {'amount':1,'reward':128,'type':"iron_ingot" },
                {'amount':1,'reward':256,'type':"iron_pickaxe" },
                {'amount':1,'reward':1024,'type':"diamond" },
            ]),
            handlers.RewardForMissionEnd(0)
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            # make the agent quit when it gets a gold block in its inventory
            handlers.AgentQuitFromPossessingItem([
                dict(type="diamond", amount=1)
            ])
        ]
    
    def create_actionables(self) -> List[Handler]:
        craftItems=["crafting_table","none","planks","stick","torch"]
        equipItems=["air","iron_axe","iron_pickaxe","none","stone_axe","stone_pickaxe","wooden_axe","wooden_pickaxe"]
        craftNearbyItems=["furnace","iron_axe","iron_pickaxe","none","stone_axe","stone_pickaxe","wooden_axe","wooden_pickaxe"]
        smeltNearbyItems=["coal","iron_ingot","none"]
        placeItems=["cobblestone","crafting_table","dirt","furnace","none","stone","torch"]
        return super().create_actionables() + [
            # allow agent to place water
            handlers.KeybasedCommandAction("use"),
            # also allow it to do different jobs

            #handlers.PlaceBlock( [ "diamond_pickaxe" ] ),
            handlers.CraftAction( craftItems,_other="other",_default="none" ), # craftAction: __init__: default value of "_other" and "_default" parameters will cause errors
            handlers.EquipAction( equipItems ),
            handlers.CraftNearbyAction( craftNearbyItems,_other="other",_default="none" ),
            handlers.SmeltItemNearby( smeltNearbyItems,_other="other",_default="none" ),
            handlers.PlaceBlock( placeItems,_other="other",_default="none" ) # ... and PlaceBlock is errornous, too
        ]

    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            # current location and lifestats are returned as additional
            # observations
            handlers.FlatInventoryObservation(["coal","cobblestone","crafting_table","dirt","furnace","iron_axe","iron_ingot","iron_ore","iron_pickaxe","log","planks","stick","stone","stone_axe","stone_pickaxe","torch","wooden_axe","wooden_pickaxe"]),
            handlers.EquippedItemObservation(["air","iron_axe","iron_pickaxe","none","other","stone_axe","stone_pickaxe","wooden_axe","wooden_pickaxe"])
        ]
    
    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            # Sets time to morning and stops passing of time
            #handlers.TimeInitialCondition(False, 23000)
            # Sets time to morning and enables passing of time
            handlers.TimeInitialCondition(True, 23000)
        ]

    def create_server_quit_producers(self):
        return []
    
    def create_server_decorators(self) -> List[Handler]:
        return []

    # the episode can terminate when this is True
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'forest'

    def get_docstring(self):
        return Forest_DOC