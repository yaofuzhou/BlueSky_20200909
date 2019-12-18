import os
import numpy as np
import random
import time
from copy import copy
from collections import deque
import argparse
import tensorflow as tf
import tensorflow.keras.backend as K
import geopy.distance
from bluesky.tools import geo
from operator import itemgetter
from shapely.geometry import LineString
import numba as nb

################################
##                            ##
##      Marc Brittain         ##
##  marcbrittain.github.io    ##
##                            ##
################################


LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-4
HIDDEN_SIZE = 32

import time



@nb.njit()
def discount(r,discounted_r,cumul_r):
    """ Compute the gamma-discounted rewards over an episode
    """
    for t in range(len(r)-1,-1,-1):
        cumul_r = r[t] + cumul_r * 0.99
        discounted_r[t] = cumul_r
    return discounted_r

def dist_goal(states,traf,i):
    olat,olon=states
    ilat,ilon =traf.ap.route[i].wplat[0],traf.ap.route[i].wplon[0]
    dist = geo.latlondist(olat,olon,ilat,ilon)/geo.nm
    return dist


def getClosestAC_Distance(self,state,traf,route_keeper):

    olat,olon,ID = state[:3]
    index = int(ID[2:])
    rte = int(route_keeper[index])
    lat,lon,glat,glon,h = self.positions[rte]
    size = traf.lat.shape[0]
    index = np.arange(size).reshape(-1,1)
    ownship_obj = LineString([[olon,olat,31000],[glon,glat,31000]])
    d  = geo.latlondist_matrix(np.repeat(olat,size),np.repeat(olon,size),traf.lat,traf.lon)
    d = d.reshape(-1,1)

    dist = np.concatenate([d,index],axis=1)

    dist = sorted(np.array(dist),key=itemgetter(0))[1:]
    if len(dist) > 0:
        for i in range(len(dist)):

            index = int(dist[i][1])
            ID_ = traf.id[index]
            index_route = int(ID_[2:])

            rte_int = route_keeper[index_route]
            lat,lon,glat,glon,h = self.positions[rte_int]
            int_obj = LineString([[traf.lon[index],traf.lat[index],31000],[glon,glat,31000]])

            if not ownship_obj.intersects(int_obj):
                continue


            if not rte_int in self.intersection_distances[rte].keys() and rte_int != rte:
                continue

            if dist[i][0] > 100:
                continue



            return dist[i][0]


    else:
        return np.inf


    return np.inf


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss





# initalize the PPO agent
class PPO_Agent:
    def __init__(self,state_size,action_size,num_routes,numEpisodes,positions):

        self.state_size = state_size
        self.action_size = action_size
        self.positions = positions
        self.gamma = 0.99    # discount rate
        self.numEpisodes = numEpisodes
        self.max_time = 500

        self.episode_count = 0
        self.speeds = np.array([156,0,346])
        self.max_agents = 0
        self.num_routes = num_routes
        self.experience = {}
        self.dist_close = {}
        self.dist_goal = {}
        self.tas_max = 253.39054470774
        self.tas_min = 118.54804803287088
        self.lr = 0.0001
        self.value_size = 1
        self.getRouteDistances()
        self.model_check = []
        self.model = self._build_A2C()

        self.count = 0



    def getRouteDistances(self):
        self.intersections = {}
        self.intersection_distances = {}
        self.route_distances = []
        self.conflict_routes = {}
        for i in range(len(self.positions)):
            olat,olon,glat,glon,h = self.positions[i]
            _, d = geo.qdrdist(olat,olon,glat,glon)
            self.route_distances.append(d)
            own_obj = LineString([[olon,olat,31000],[glon,glat,31000]])
            self.conflict_routes[i] = []
            for j in range(len(self.positions)):
                if i == j: continue
                olat,olon,glat,glon,h = self.positions[j]
                other_obj = LineString([[olon,olat,31000],[glon,glat,31000]])
                self.conflict_routes[i].append(j)
                if own_obj.intersects(other_obj):
                    intersect = own_obj.intersection(other_obj)
                    try:
                        Ilon,Ilat,alt = list(list(intersect.boundary[0].coords)[0])
                    except:
                        Ilon,Ilat,alt = list(list(intersect.coords)[0])

                    try:
                        self.intersections[i].append([j,[Ilat,Ilon]])
                    except:
                        self.intersections[i] = [[j,[Ilat,Ilon]]]



        for route in self.intersections.keys():
            olat,olon,glat,glon,h = self.positions[i]

            for intersections in self.intersections[route]:
                conflict_route,location = intersections
                Ilat,Ilon = location
                _,d = geo.qdrdist(Ilat,Ilon,glat,glon)
                try:
                    self.intersection_distances[route][conflict_route] = d
                except:
                    self.intersection_distances[route] = {conflict_route:d}

        self.max_d = max(self.route_distances)






    def normalize_that(self,value,what,context=False,state=False,id_=None):


        if what=='spd':

            if value > self.tas_max:
                self.tas_max = value

            if value < self.tas_min:
                self.tas_min = value
            return (value-self.tas_min)/(self.tas_max-self.tas_min)

        if what=='rt':
            return value/(self.num_routes-1)



        if what=='state':

            dgoal = self.dist_goal[id_]/self.max_d
            spd = self.normalize_that(value[2],'spd')
            rt = self.normalize_that(value[3],'rt')
            acc = value[4]+0.5
            rt_own = int(value[3])
            norm_array = np.array([dgoal,spd,rt,acc,3/self.max_d])

            return norm_array


        if what == 'context':

            rt_own = int(state[3])
            dgoal = self.dist_goal[id_]/self.max_d
            spd = self.normalize_that(context[2],'spd')
            rt = self.normalize_that(context[3],'rt')
            acc = context[4]+0.5
            rt_int = int(context[3])

            if rt_own == rt_int:
                dist_away = abs(value[0]-dgoal)
                dist_own_intersection = 0
                dist_int_intersection = 0#

            else:
                dist_own_intersection = abs(self.intersection_distances[rt_own][rt_int]/self.max_d - value[0])
                dist_int_intersection = abs(self.intersection_distances[rt_int][rt_own]/self.max_d - dgoal)
                d  = geo.latlondist(state[0],state[1],context[0],context[1])/geo.nm
                dist_away = d/self.max_d


            context_arr = np.array([dgoal,spd,rt,acc,dist_away,dist_own_intersection,dist_int_intersection])

            return context_arr.reshape(1,1,7)


    def _build_A2C(self):

        I = tf.keras.layers.Input(shape=(self.state_size,),name='states')

        context = tf.keras.layers.Input(shape=(None,7),name='context')
        empty = tf.keras.layers.Input(shape=(HIDDEN_SIZE,),name='empty')

        advantage = tf.keras.layers.Input(shape=(1,),name='A')
        old_prediction = tf.keras.layers.Input(shape=(self.action_size,),name='old_pred')


        # encoding other_state into 32 values
        H1_int = tf.keras.layers.LSTM(HIDDEN_SIZE,activation='tanh')(context,initial_state=[empty,empty])
        # now combine them
        combined = tf.keras.layers.concatenate([I,H1_int], axis=1)


        H2 = tf.keras.layers.Dense(256,activation='relu')(combined)
        H3 = tf.keras.layers.Dense(256,activation='relu')(H2)

        output = tf.keras.layers.Dense(self.action_size+1,activation=None)(H3)

        # Split the output layer into policy and value
        policy = tf.keras.layers.Lambda(lambda x: x[:,:self.action_size],output_shape=(self.action_size,))(output)
        value = tf.keras.layers.Lambda(lambda x: x[:,self.action_size:],output_shape=(self.value_size,))(output)

        # now I need to apply activation
        policy_out = tf.keras.layers.Activation('softmax',name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear',name='value_out')(value)

        # Using Adam optimizer, RMSProp's successor.
        opt = tf.keras.optimizers.Adam(lr=self.lr)

        model = tf.keras.models.Model(inputs=[I,context,empty,advantage,old_prediction], outputs=[policy_out,value_out])

        self.predictor = tf.keras.models.Model(inputs=[I,context,empty], outputs=[policy_out,value_out])

        # The model is trained on 2 different loss functions
        model.compile(optimizer=opt, loss={'policy_out':proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction), 'value_out':'mse'})

        print(model.summary())

        return model


    def store(self,state,action,next_state,traf,id_,route_keeper,term=0):
        reward = 0
        done = False

        if term == 0:
            lat = traf.lat[traf.id2idx(id_)]
            lon = traf.lon[traf.id2idx(id_)]

            dist = self.dist_close[id_]

            if dist < 10 and dist > 3:
                reward = -0.1 + 0.05*(dist/10)


        if term == 1:

            reward= -1
            done = True

        if term == 2:
            reward = 0
            done = True


        state,context = state
        state = state.reshape((1,5))
        context = context.reshape((1,-1,7))

        self.max_agents = max(self.max_agents,context.shape[1])

        if not id_ in self.experience.keys():
            self.experience[id_] = {}

        try:
            self.experience[id_]['state'] = np.append(self.experience[id_]['state'],state,axis=0)

            if self.max_agents > self.experience[id_]['context'].shape[1]:
                self.experience[id_]['context'] = np.append(tf.keras.preprocessing.sequence.pad_sequences(self.experience[id_]['context'],self.max_agents,dtype='float32'),context,axis=0)
            else:
                self.experience[id_]['context'] = np.append(self.experience[id_]['context'],tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32'),axis=0)

            self.experience[id_]['action'] = np.append(self.experience[id_]['action'],action)
            self.experience[id_]['reward'] = np.append(self.experience[id_]['reward'],reward)
            self.experience[id_]['done'] = np.append(self.experience[id_]['done'],done)




        except:
            self.experience[id_]['state'] = state
            if self.max_agents > context.shape[1]:
                self.experience[id_]['context'] = tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32')
            else:
                self.experience[id_]['context'] = context

            self.experience[id_]['action'] = [action]
            self.experience[id_]['reward'] = [reward]
            self.experience[id_]['done'] = [done]



    def train(self):

        """Grab samples from batch to train the network"""

        total_state = []
        total_reward = []
        total_A = []
        total_advantage = []
        total_context = []
        total_policy = []

        total_length = 0

        for transitions in self.experience.values():
            episode_length = transitions['state'].shape[0]
            total_length += episode_length

            state = transitions['state']#.reshape((episode_length,self.state_size))
            context = transitions['context']
            reward = transitions['reward']
            done = transitions['done']
            action  = transitions['action']

            discounted_r, cumul_r = np.zeros_like(reward), 0
            discounted_rewards = discount(reward,discounted_r, cumul_r)
            policy,values = self.predictor.predict({'states':state,'context':context,'empty':np.zeros((len(state),HIDDEN_SIZE))},batch_size=256)
            advantages = np.zeros((episode_length, self.action_size))
            index = np.arange(episode_length)
            advantages[index,action] = 1
            A = discounted_rewards - values[:,0]

            if len(total_state) == 0:

                total_state = state
                if context.shape[1] == self.max_agents:
                    total_context = context
                else:
                    total_context = tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32')
                total_reward = discounted_rewards
                total_A = A
                total_advantage = advantages
                total_policy = policy

            else:
                total_state = np.append(total_state,state,axis=0)
                if context.shape[1] == self.max_agents:
                    total_context = np.append(total_context,context,axis=0)
                else:
                    total_context = np.append(total_context,tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32'),axis=0)
                total_reward = np.append(total_reward,discounted_rewards,axis=0)
                total_A = np.append(total_A,A,axis=0)
                total_advantage = np.append(total_advantage,advantages,axis=0)
                total_policy = np.append(total_policy,policy,axis=0)


        total_A = (total_A - total_A.mean())/(total_A.std() + 1e-8)
        self.model.fit({'states':total_state,'context':total_context,'empty':np.zeros((total_length,HIDDEN_SIZE)),'A':total_A,'old_pred':total_policy}, {'policy_out':total_advantage,'value_out':total_reward}, shuffle=True,batch_size=total_state.shape[0],epochs=8, verbose=0)


        self.max_agents = 0
        self.experience = {}




    def load(self, name):
        print('Loading weights...')
        self.model.load_weights(name)
        print('Successfully loaded model weights from {}'.format(name))


    def save(self,best=False,case_study='A'):


        if best:

            self.model.save_weights('best_model_{}.h5'.format(case_study))


        else:

            self.model.save_weights('model_{}.h5'.format(case_study))


    # action implementation for the agent
    def act(self,state,context):


        context = context.reshape((state.shape[0],-1,7))
        policy,value = self.predictor.predict({'states':state,'context':context,'empty':np.zeros((state.shape[0],HIDDEN_SIZE))},batch_size=state.shape[0])


        return policy

    def update(self,traf,index,route_keeper):
        """calulate reward and determine if terminal or not"""
        T = 0
        type_ = 0
        dist = getClosestAC_Distance(self,[traf.lat[index],traf.lon[index],traf.id[index]],traf,route_keeper)
        if dist < 3:
            T = True
            type_ = 1


        self.dist_close[traf.id[index]] = dist



        d_goal = dist_goal([traf.lat[index],traf.lon[index]],traf,index)

        if d_goal < 5 and T == 0:
            T = True
            type_ = 2

        self.dist_goal[traf.id[index]] = d_goal

        return T,type_
