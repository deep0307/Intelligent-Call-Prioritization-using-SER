from threading import main_thread
import time
from call_center import call_center
import pprint

users = [{
        'arrival_time': 1650225430.704303,
        'callback': 0,
        'emotion': 'hap',
        'loyalty': 0,
        'score': 1.0373152842364424,
        'temp_score': 0,
        'user_id': 0,
        'wav_file_name': 'happy'
    }, {
        'arrival_time': 1650225430.7044072,
        'callback': 0,
        'emotion': 'neu',
        'loyalty': 1,
        'score': 2.3948040658765426,
        'temp_score': 0,
        'user_id': 1,
        'wav_file_name': 'angry'
    }, {
        'arrival_time': 1650225430.704485,
        'callback': 0,
        'emotion': 'sad',
        'loyalty': 2,
        'score': 3.828070276340351,
        'temp_score': 0,
        'user_id': 2,
        'wav_file_name': 'sad'
    }]
    

class allocator:
    

    def main():
        global users
        center = call_center(5)

        # center.print_queues()

        for u in users:
            # print("THIS IS USER BEFORE CALL : \n\n",u)
            center.updateQueue(d=u,waiting_queue=center.waiting_queue)

        center.updateQueue({'arrival_time': time.time(),
            'callback': 1,
            'emotion': 'hap',
            'loyalty': 1,
            'score': 10,
            'temp_score': 0,
            'user_id': 3,
            'wav_file_name': 'this is trial'},center.waiting_queue)        

        center.updateQueue({'arrival_time': time.time(),
            'callback': 0,
            'emotion': 'ang',
            'loyalty': 1,
            'score': 5,
            'temp_score': 0,
            'user_id': 4,
            'wav_file_name': 'this is trial2'},center.waiting_queue)

        print("BEFORE UPDATING AGENT QUEUES")
        center.print_queues()
        center.agent_suitability()
        for key, value in center.agent_waiting_queue.items():
            center.agent_waiting_queue[key] = center.updateQueue({},value,rearrange=True)
            print("\n")
        print("AFTER UPDATING AGENT QUEUES")
        center.print_queues()


    if __name__ == "__main__":
        main()