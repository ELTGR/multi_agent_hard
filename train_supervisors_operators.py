import os
from time import sleep
from copy import deepcopy
import ray
from ray.rllib.algorithms.ppo import (
    PPOConfig,
    PPOTF1Policy,
    PPOTF2Policy,
    PPOTorchPolicy,
)
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import random


class MobileSupervisor:
    def __init__(self):
        # Initialisation de la classe MobileSupervisor
        self._supervisor_x = None
        self._supervisor_y = None

    def get_x_pos(self):
        return self._supervisor_x

    def get_y_pos(self):
        return self._supervisor_y

    def set_x_pos(self, nouvelle_valeur):
        self._supervisor_x = nouvelle_valeur

    def set_y_pos(self, nouvelle_valeur):
        self._supervisor_y = nouvelle_valeur

    def moveTo(self):
        # Méthode de la classe MobileSupervisor
        pass

    def doThis(self):
        # Méthode de la classe MobileSupervisor
        pass

    def doThat(self):
        # Méthode de la classe MobileSupervisor
        pass

class MobileOperator:
    def __init__(self):
        # Initialisation de la classe MobileOperator
        self._supervisor_x = None
        self._supervisor_y = None

    def get_x_pos(self):
        return self._supervisor_x

    def get_y_pos(self):
        return self._supervisor_y

    def set_x_pos(self, nouvelle_valeur):
        self._supervisor_x = nouvelle_valeur

    def set_y_pos(self, nouvelle_valeur):
        self._supervisor_y = nouvelle_valeur

    def moveTo(self):
        # Méthode de la classe MobileOperator
        pass

    def doThis(self):
        # Méthode de la classe MobileOperator
        pass

    def doThat(self):
        # Méthode de la classe MobileOperator
        pass


class MultiAgentsSupervisorOperatorsEnv(MultiAgentEnv):
    def __init__(self, env_config):
        #print("******__init__*******")
        self.subzones_width = env_config["subzones_width"]
        self.largeur_grille = env_config["num_boxes_grid_width"] #Nombre de colonnes de la grille
        self.hauteur_grille = env_config["num_boxes_grid_height"]  #Nombre de lignes de la grille

        #definition des goals et des centres des sous zones

        self.subzones = { 0:[0,0], 1:[3,0], 2:[6,0], 3:[0,3], 4:[3,3], 5:[6,3], 6:[0,6], 7:[3,6], 8:[6,6], 9:[0,9], 10:[3,9], 11:[6,9]}

        self.subzones_goals = {0 : None, 1 : None, 2 : None,
                               3 : None, 4 : None, 5 : None,
                               6 : None, 7 : None, 8 : None,
                               9 : None,10 : None,11 : None,}

        self.subzones_center = {0 : None, 1 : None, 2 : None,
                               3 : None, 4 : None, 5 : None,
                               6 : None, 7 : None, 8 : None,
                               9 : None,10 : None,11 : None,}


        for i in range(0, len(self.subzones)) :

            goals= ([0+self.subzones[i][0],1+self.subzones[i][0],2+self.subzones[i][0]],[0+self.subzones[i][1],1+self.subzones[i][1],2+self.subzones[i][1]])
            self.subzones_goals[i]=(goals[0],goals[1])


        for i in range(0, len(self.subzones)) :

            center = (1+self.subzones[i][0],1+self.subzones[i][1])
            self.subzones_center[i]=(center[0],center[1])


        self.agents_ids = ["supervisor","operator_0","operator_1","operator_2"]
        #creation des instance des agents
        supervisor= MobileSupervisor()
        operator_0=MobileOperator()
        operator_1=MobileOperator()
        operator_2=MobileOperator()

        #link between instance and name
        self.mobile_agents = { self.agents_ids[0] : supervisor
                              ,self.agents_ids[1] : operator_0
                              ,self.agents_ids[2] : operator_1
                              ,self.agents_ids[3] : operator_2 }



        self.agents_goals = {
         self.agents_ids[0] : ([self.subzones_center[0][0],self.subzones_center[1][0],self.subzones_center[2][0]],[ self.subzones_center[0][1],self.subzones_center[1][1],self.subzones_center[2][1]])
        ,self.agents_ids[1] : self.subzones_goals[0]
        ,self.agents_ids[2] : self.subzones_goals[1]
        ,self.agents_ids[3] : self.subzones_goals[2]
        }
        #print(self.agents_goals)
        #print("1111")
        self.check_goal = deepcopy(self.agents_goals)

        self.operator_end = {     self.agents_ids[1]: 0,
                                  self.agents_ids[2]: 0,
                                  self.agents_ids[3]: 0,}
        #print(self.operator_end)
        self.observation_space ={ self.agents_ids[0]: spaces.Box(low=0, high=11, shape=(11,)),
            self.agents_ids[1]: spaces.Box(low=0, high=11, shape=(8,)),
            self.agents_ids[2]: spaces.Box(low=0, high=11, shape=(8,)),
            self.agents_ids[3]: spaces.Box(low=0, high=11, shape=(8,)),
            }
        super().__init__()


    def reset(self):
        #print("******reste*******")

        self.step_counter=0
        self.zone_counter=0
        self.change_goal=False
        self.zone_center_counter=0
        self.supervisor_check = False
        
        for id in self.agents_ids:
            self.mobile_agents[id].set_x_pos(0)
            self.mobile_agents[id].set_y_pos(0)
        #print("11111")
        self.agents_goals = {
         self.agents_ids[0] : ([self.subzones_center[0][0],self.subzones_center[1][0],self.subzones_center[2][0]],[ self.subzones_center[0][1],self.subzones_center[1][1],self.subzones_center[2][1]])
        ,self.agents_ids[1] : self.subzones_goals[0]
        ,self.agents_ids[2] : self.subzones_goals[1]
        ,self.agents_ids[3] : self.subzones_goals[2]
        }
        #print("111111")
        self.check_goal = deepcopy(self.agents_goals)

        self.operator_end = {self.agents_ids[1]: 0,
                             self.agents_ids[2]: 0,
                             self.agents_ids[3]: 0,}

        observations={ self.agents_ids[0]:None,
            self.agents_ids[1]:None,
            self.agents_ids[2]:None,
            self.agents_ids[3]:None,
            }

        for id in self.agents_ids:
            observations[id] = self._get_observation(id)
        #print(observations)
        return observations

    def _get_observation(self,agent_id):


        #print("self.agents_goals = " + str(self.agents_goals))
        if agent_id == "supervisor" :
            observation =   [self.mobile_agents[agent_id].get_x_pos(),
                            self.mobile_agents[agent_id].get_y_pos(), self.operator_end[self.agents_ids[1]], self.operator_end[self.agents_ids[2]], self.operator_end[self.agents_ids[3]]]

        else :
            observation =   [self.mobile_agents[agent_id].get_x_pos(),
                             self.mobile_agents[agent_id].get_y_pos(),]
            #print("observation+=self.agents_goals[agent_id][0]" + str(self.agents_goals[agent_id][0]))
            #print("observation+=self.agents_goals[agent_id][1]" + str(self.agents_goals[agent_id][1]))

        observation.extend(self.agents_goals[agent_id][0])
        observation.extend(self.agents_goals[agent_id][1])
        #print("self.agents_goals[" + str(agent_id)+"] = " + str(self.agents_goals[agent_id]))
        return observation



    def step(self, action_dict):
        #print("******step***********")
        self.step_counter += 1
        self.operator_end_task = 0
        #print("step : " + str(self.step_counter))

        observations={self.agents_ids[0]:None,
                      self.agents_ids[1]:None,
                      self.agents_ids[2]:None,
                      self.agents_ids[3]:None, }

        rewards={     self.agents_ids[0]:None,
                      self.agents_ids[1]:None,
                      self.agents_ids[2]:None,
                      self.agents_ids[3]:None,}

        terminateds={ self.agents_ids[0]:None,
                      self.agents_ids[1]:None,
                      self.agents_ids[2]:None,
                      self.agents_ids[3]:None,
                      "__all__" : False }

        infos={       self.agents_ids[0]:None,
                      self.agents_ids[1]:None,
                      self.agents_ids[2]:None,
                      self.agents_ids[3]:None,}

        for agent_id, action in action_dict.items() :

            if agent_id != "supervisor":
                #=================Move========================#
                pre_x = self.mobile_agents[agent_id].get_x_pos()
                pre_y =self.mobile_agents[agent_id].get_y_pos()

                if action == 0:  # UP
                    self.mobile_agents[agent_id].set_y_pos(min(self.hauteur_grille-1, pre_y+1))
                elif action == 1:  # DOWN
                    self.mobile_agents[agent_id].set_y_pos(max(0, pre_y-1 ))
                elif action == 2:  # LEFT
                    self.mobile_agents[agent_id].set_x_pos(max(0, pre_x-1))
                elif action == 3:  # RIGHT
                    self.mobile_agents[agent_id].set_x_pos(min(self.largeur_grille-1, pre_x+1))
                else:
                    raise Exception("action: {action} is invalid")

                now_x = self.mobile_agents[agent_id].get_x_pos()
                now_y =self.mobile_agents[agent_id].get_y_pos()

                if self.operator_end[agent_id]==1 : 
                        #print("in")
                        terminateds[agent_id]=False
                        rewards[agent_id]=0
                #=================Check goals========================#
                for i in range(0,len(self.check_goal[agent_id][0])) :
                                        
                       
                    goal_x = self.check_goal[agent_id][0][i]
                    goal_y = self.check_goal[agent_id][1][i]


                    if (now_x,now_y)==(goal_x,goal_y) :

                        rewards[agent_id]=10
                        del self.check_goal[agent_id][0][i]
                        del self.check_goal[agent_id][1][i]
                        goal_uncheck = len(self.check_goal [agent_id][0])
                        # print("goal = " + str(self.agents_goals))
                        # print("subzone goal = " + str(self.subzones_goals))
                        # print("subzone = " + str(self.subzones_goals))
                        # print("check_goal = " + str(self.check_goal))

                        if goal_uncheck == 0 :

                            self.operator_end[agent_id]=1
                            terminateds[agent_id]=False
                            rewards[agent_id]=100

                        else :

                            terminateds[agent_id]=False
                            rewards[agent_id]=0
                            
                        break
                    else :
                        terminateds[agent_id]=False
                        rewards[agent_id]=0

                if self.step_counter >= 4000:
                    rewards[agent_id]=-100
                    terminateds["__all__"]=True

                if self.operator_end[agent_id] == 1 :
                    self.operator_end_task+=1

                if self.operator_end_task == 3 :
                    self.supervisor_check = True
                infos[agent_id]={}
                observations[agent_id]=self._get_observation(agent_id)

            elif agent_id == "supervisor":

                action_move = action[0]

                subzone_for_operator_0 = action[0]
                subzone_for_operator_1 = action[1]
                subzone_for_operator_2 = action[2]
                # print("action= " + str(action_move))
                # print("subzone_for_operator_0 = " + str(subzone_for_operator_0))
                # print("subzone_for_operator_1 = " + str(subzone_for_operator_1))
                # print("subzone_for_operator_2 = " + str(subzone_for_operator_2))
                #=================Move========================#
                pre_x = self.mobile_agents[agent_id].get_x_pos()
                pre_y =self.mobile_agents[agent_id].get_y_pos()

                if action_move == 0:  # UP
                    self.mobile_agents[agent_id].set_y_pos(min(self.hauteur_grille-1, pre_y+1))
                elif action_move == 1:  # DOWN
                    self.mobile_agents[agent_id].set_y_pos(max(0, pre_y-1 ))
                elif action_move == 2:  # LEFT
                    self.mobile_agents[agent_id].set_x_pos(max(0, pre_x-1))
                elif action_move == 3:  # RIGHT
                    self.mobile_agents[agent_id].set_x_pos(min(self.largeur_grille-1, pre_x+1))
                else:
                    raise Exception("action_move: {action_move} is invalid")

                now_x = self.mobile_agents[agent_id].get_x_pos()
                now_y =self.mobile_agents[agent_id].get_y_pos()


                #=================Check goals========================#
                for i in range(0,len(self.check_goal[agent_id][0])) :

                    goal_x = self.check_goal[agent_id][0][i]
                    goal_y = self.check_goal[agent_id][1][i]

                    if (now_x,now_y)==(goal_x,goal_y) and self.supervisor_check :

                        rewards[agent_id]=10
                        del self.check_goal[agent_id][0][i]
                        del self.check_goal[agent_id][1][i]
                        goal_uncheck = len(self.check_goal [agent_id][0])


                        if goal_uncheck == 0 :
                            self.change_goal = True
                            terminateds[agent_id]=False
                            rewards[agent_id]=100

                        else :

                            terminateds[agent_id]=False
                            rewards[agent_id]=0
                        break
                    else :
                        terminateds[agent_id]=False
                        rewards[agent_id]=0

                if self.step_counter >= 4000:
                    rewards[agent_id]=-100
                    terminateds["__all__"]=True

            if self.change_goal :
                self.zone_center_counter+=1
                case_suivante = self.zone_center_counter*3
                self.zone_counter+=3
                #self.agents_goals[self.agents_ids[0]] = self.subzones[subzone_for_operator_2]
                
                self.agents_goals[self.agents_ids[0]] = ([self.subzones_center[0+case_suivante][0],self.subzones_center[1+case_suivante][0],self.subzones_center[2+case_suivante][0]],
                                                         [self.subzones_center[0+case_suivante][1],self.subzones_center[1+case_suivante][1],self.subzones_center[2+case_suivante][1]])
                self.agents_goals[self.agents_ids[1]] = self.subzones_goals[subzone_for_operator_0]
                self.agents_goals[self.agents_ids[2]] = self.subzones_goals[subzone_for_operator_1]
                self.agents_goals[self.agents_ids[3]] = self.subzones_goals[subzone_for_operator_2]
                self.change_goal=False
                print(" self.change_goal = " + str(self.change_goal))
                print(" goals " + str(self.agents_goals))
                print("subzone_for_supervisor= " + str(self.zone_center_counter))
                print("subzone_for_operator_0 = " + str(subzone_for_operator_0))
                print("subzone_for_operator_1 = " + str(subzone_for_operator_1))
                print("subzone_for_operator_2 = " + str(subzone_for_operator_2))

                terminateds["__all__"]=True

            infos[agent_id]={}
            observations[agent_id]=self._get_observation(agent_id)

            if self.zone_counter > 12 :
                        rewards[agent_id]={-1000}

            # if terminateds[agent_id] == None :

            #     del observations[agent_id]
            #     del rewards[agent_id]
            #     del terminateds[agent_id]
            #     del infos[agent_id]

        # print("observations = " + str(observations))
        # print("goals uncheck = " + str(self.check_goal))
        # print("rewards = " + str(rewards))
        # print("terminateds = " + str(terminateds))
        # print("operator task end = " + str(self.operator_end))
        return observations, rewards, terminateds, infos



if __name__ == "__main__":

    ray.init()

    def select_policy(algorithm, framework):
        if algorithm == "PPO":
            if framework == "torch":
                return PPOTorchPolicy
            elif framework == "tf":
                return PPOTF1Policy
            else:
                return PPOTF2Policy
        else:
            raise ValueError("Unknown algorithm: ", algorithm)

    taille_map_x = 12
    taille_map_y = 9
    ppo_config = (
        PPOConfig()
        # or "corridor" if registered above
        .environment(MultiAgentsSupervisorOperatorsEnv,
                    env_config={
                        "num_boxes_grid_width":taille_map_x,
                        "num_boxes_grid_height":taille_map_y,
                        "subzones_width":3,
                    })
        .environment(disable_env_checking=True)

        .framework("torch")

        # disable filters, otherwise we would need to synchronize those
        # as well to the DQN agent
        .rollouts(observation_filter="MeanStdFilter")
        .training(
            model={"vf_share_layers": True},
            vf_loss_coeff=0.01,
            num_sgd_iter=6,
            _enable_learner_api=False,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=0)
        #.rollouts(num_rollout_workers=1)
        .rl_module(_enable_rl_module_api=False)

    )

    obs_supervisor = spaces.Box(low=0, high=taille_map_x, shape=(11,))
    obs_operator = spaces.Box(low=0, high=taille_map_x, shape=(8,))

    action_supervisor  = spaces.MultiDiscrete([4, 12, 12, 12])
    action_operator  = spaces.Discrete(4)

    policies = {
        "supervisor_policy": (None,obs_supervisor,action_supervisor, {}),
        "operator_policy": (None,obs_operator,action_operator, {}),
        #"operator_1": (None,obs_operator,acti, {}),
        #"operator_2": (None,obs_operator,acti, {}),
    }


    def policy_mapping_fn(agent_id, episode, worker, **kwargs):

        if agent_id == "supervisor" :

            return "supervisor_policy"

        else :

            return "operator_policy"


    ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
    ppo = ppo_config.build()


    i=0
    intervalle=100
    last_i=0

    while True :

        i+=1
        print("== Iteration", i, "==")
        print("-- PPO --")
        result_ppo = ppo.train()
        print(pretty_print(result_ppo))
        if (i-last_i)==intervalle :


            checkpoint = ppo.save()
            print(checkpoint)


        # restored_algo = Algorithm.from_checkpoint(checkpoint)
