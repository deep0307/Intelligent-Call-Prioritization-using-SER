
import random
import time
import numpy as np
import pandas as pd
import pprint

class call_center:

    encoding = {
    'ang':1,
    'hap':2,
    'sad':3,
    'neu':4
    }

    multi_fact = {
        'ang':8,
        'hap':5,
        'sad':3,
        'neu':1
    }

    waiting_queue=[]
    agent_suitability_list=[]

    def __init__(self, num):
        self.num_agents = num
        self.agent_list = []
        self.agent_status = dict.fromkeys(list(range(num)))
        self.agent_waiting_queue = dict.fromkeys(list(range(num)))
        self.emotion_matrix = []
        self.serves = dict.fromkeys(list(range(num)))
        for i in range(num - 1):
            agent = dict()
            agent['agent_id'] = i
            agent['preference'] = []
            from random import randrange
            zeros = randrange(4)
            agent['preference'] = random.sample(range(1, 5), 4 - zeros)
            for j in range(zeros):
                agent['preference'].append(0)
            self.agent_list.append(agent)
            self.emotion_matrix.append(agent['preference'])
            self.agent_status[i]="free"
            self.serves[i]=0
        self.agent_list.append({'agent_id': num - 1, 'preference': [4, 3, 1, 2]})
        self.agent_status[num-1]="free"
        self.serves[num-1]=0
        self.emotion_matrix.append([4, 3, 1, 2])
        self.emotion_matrix = np.array(self.emotion_matrix)

    def leaveQueue(self,id):
        for user in self.waiting_queue:
            if user["user_id"]==id:
                self.waiting_queue.remove(user)
        for key,value in self.agent_waiting_queue.items():
            for user in value:
              if user["user_id"]==id:
                value.remove(user)

    def updateQueue(self,d,waiting_queue,rearrange=False):
        # print("\n\n\n\d is\n")
        # print(d)
        if(rearrange==False):
            waiting_queue.append(d)
        
        for user in waiting_queue:
            user['temp_score']= (user['score']*self.multi_fact[user['emotion']]+user['loyalty']+user['callback'])+(time.time()-user['arrival_time'])/60.0
        waiting_queue = sorted(waiting_queue, key=lambda dic: dic['temp_score'], reverse=True) 
        # print(waiting_queue)
        return waiting_queue

    def agent_suitability(self):
        print("\n\nagent suitability list")
        n=self.num_agents
        for user in self.waiting_queue:
            asl = []
            for i in range(4):
                for j in range(n):
                    if(self.encoding[user['emotion']]==self.emotion_matrix[j][i]):
                        self.agent_suitability_list.append(self.agent_list[j])
                        asl.append(self.agent_list[j]['agent_id'])
                        user_temp = user.copy()
                        user_temp['score'] = user_temp['score']*(1-(len(asl)-1)/n)
                        flag=0
                        try:
                            for u in self.agent_waiting_queue[self.agent_list[j]['agent_id']]:
                                if u["user_id"] == user_temp["user_id"]:
                                    flag=1                       
                            if flag==0:
                                self.agent_waiting_queue[self.agent_list[j]['agent_id']].append(user_temp)
                        except:
                            self.agent_waiting_queue[self.agent_list[j]['agent_id']]= []
                            for u in self.agent_waiting_queue[self.agent_list[j]['agent_id']]:
                                if u["user_id"] == user_temp["user_id"]:
                                    flag=1
                            if flag==0:
                                self.agent_waiting_queue[self.agent_list[j]['agent_id']].append(user_temp)

            # print(agent_suitability_list)
            for agent in self.agent_list:
                if(agent['agent_id'] in asl):
                    continue
                else:
                    user_temp = user.copy()
                    user_temp['score'] = user_temp['score']*(1-len(asl)/n)
                    flag=0
                    try:
                        for u in self.agent_waiting_queue[agent['agent_id']]:
                            if u["user_id"] == user_temp["user_id"]:
                                flag=1
                        if flag==0:
                            self.agent_waiting_queue[agent['agent_id']].append(user_temp)
                    except:
                        self.agent_waiting_queue[agent['agent_id']] = []
                        for u in self.agent_waiting_queue[agent['agent_id']]:
                            if u["user_id"] == user_temp["user_id"]:
                                flag=1
                        if flag==0:
                            self.agent_waiting_queue[agent['agent_id']].append(user_temp)
            print(user['user_id'],"\t",self.agent_suitability_list,"\n\n")

    def print_queues(self):
        print("\n\nwaiting queue")
        pprint.pprint(self.waiting_queue)
        # print("\n\nagent suitability queue")
        # pprint.pprint(self.agent_suitability_list)
        print("\n\nagent waiting queue")
        pprint.pprint(self.agent_waiting_queue)
        print("\n\n")

    def allocator(self):
      for key,value in self.agent_status.items():
        if value=="free":
          try:
            user_temp=self.agent_waiting_queue[key].pop(0)
            self.agent_status[key]="busy"
            self.serves[key]=user_temp["user_id"]
            self.leaveQueue(user_temp["user_id"])
            self.agent_suitability()
            for key, value in self.agent_waiting_queue.items():
              self.agent_waiting_queue[key] = self.updateQueue({},value,rearrange=True)
          except:
            return