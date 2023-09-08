import json  
from call_center import call_center
center=call_center(3)
from calls import calls
import pprint
import time
   
num_agents=3

active_calls=[]

def incoming_call(u):
  print("\n\n\n\n")
  print(u)
  print(u["score"])
  center.waiting_queue=center.updateQueue(u,center.waiting_queue)
  center.print_queues()
  center.agent_suitability()
  for key, value in center.agent_waiting_queue.items():
      center.agent_waiting_queue[key] = center.updateQueue({},value,rearrange=True)
  center.allocator()
  return str(center.waiting_queue),str(center.agent_waiting_queue)

def disconnect(id):
  for key,value in center.serves.items():
      if value==id:
          center.serves[key]=0
          center.agent_status[key]="free"
  center.leaveQueue(id)
  center.agent_suitability()
  for key, value in center.agent_waiting_queue.items():
      center.agent_waiting_queue[key] = center.updateQueue({},value,rearrange=True)
  center.allocator()
  return str(center.waiting_queue),str(center.agent_waiting_queue)


def get_agent_status(id):
  return center.agent_status[id],center.serves[id]

def print_agents():
    print("\n\n\n")
    for i in range(num_agents):
        a_status, a_client = get_agent_status(i)
        print("Agent "+str(i)+" is "+a_status+" serving Client "+str(a_client))

def end_call():
    for i in range(num_agents):
        a_status, a_client = get_agent_status(i)
        if a_status=="busy":
            for ac in  active_calls:
                if ac["user"]["user_id"]==a_client:
                    if ac["call_time"]<(time.time()-ac["user"]["arrival_time"]):
                        ac["end_time"]=time.time()-ac["user"]["arrival_time"]
                        wq,aq=disconnect(a_client)
                        pprint.pprint(wq)
                        pprint.pprint(aq)

    
def print_active_call():
    print("\n\n\nActive Calls")
    for ac in active_calls:
        pprint.pprint(ac)

for call in calls.call_list:
    call["user"]["arrival_time"]=time.time()
    pprint.pprint(call)
    active_calls.append(call)
    wq,aq=incoming_call(call["user"])
    print("\n\n\nWaiting queue")
    pprint.pprint(list(eval(wq)))
    print("\n\n\nAgent queue")
    pprint.pprint(eval(aq))
    print_agents()
    print_active_call()

