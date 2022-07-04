
#. First Come First Served Approach

from utils_parameters import PRIORITY, DATASET_PATH, INP_RESOURCES, NUM_TIME_STEPS, REQUEST_CLASSES
import  utils_nextg
import numpy as np
import os, shutil
from tensorboardX import SummaryWriter


name = 'fcfs_priorityQ' if PRIORITY else 'fcfs_FCFSQ'
dset_name = DATASET_PATH.split('/')[-1][:-4]
logname = f'../logs/{dset_name}/{name}_{0}'

#Straightforward pass through the environment, accepts every request until there is no more space
def simulate_steps(env,writer,dset_path):

    state = env.reset()
    episode_reward = 0; reward =0; action = 0
    timestep_total_reward = np.empty(NUM_TIME_STEPS,dtype=float) 
    for time_step in range(NUM_TIME_STEPS):
        print(f"Time Step {time_step} -> Queue Size: {len(env.p_queue.queue)} | Last Reward: {reward} | Executing: {len(env.executing)} | Completed: {len(env.completed)}", end='\r')
    

        if(len(env.p_queue) > 0):
            #Accept as long as there is space in InP
            action = 1
            for resource in utils_nextg.RESOURCES:
                if(env.p_queue[0].resource_quantity[resource] > env.free_inp_resources[resource]):
                    action = 0
                    break

        next_state, reward = env.step(action,time_step) #apply action
        episode_reward += reward
        timestep_total_reward[time_step] = episode_reward


    aux_name = dset_path.split("/")[-1][:-4]
    np.save(f'../output/FCFS_ts_reward_{aux_name}.npy',timestep_total_reward)
    with open(f"../output/FCFS_reward_{aux_name}.txt",'w') as f:
        f.write(str(episode_reward))


    accepted_jobs = len(env.executing)+len(env.completed)
    timeouts = env.p_queue.num_timeouts
    average_waiting = env.p_queue.get_average_waiting_time()
    acc_pct = {clazz:0.0 for clazz in REQUEST_CLASSES}
    for c,clazz in enumerate(REQUEST_CLASSES):
        acc_pct[clazz]= env.accepted_per_class[c]/env.total_num_requests[c]
    print('\nMoving average score: {:.2f}\t'.format(episode_reward))


    for epoch in range(100):
        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        writer.add_scalar('Accepted Jobs', accepted_jobs, global_step=epoch)
        writer.add_scalar('Timeouts', timeouts, global_step=epoch)
        writer.add_scalar('Average Waiting Time', average_waiting, global_step=epoch)
    
    keys = list(acc_pct.keys())
    metrics = (episode_reward, accepted_jobs, timeouts, average_waiting, acc_pct[keys[0]],acc_pct[keys[1]],acc_pct[keys[2]])

    return metrics

if(__name__=='__main__'):

    env = utils_nextg.Environment(INP_RESOURCES,NUM_TIME_STEPS,loadpath=DATASET_PATH,priorityq=PRIORITY)
    
    if(os.path.exists(logname)):
        shutil.rmtree(logname)
    writer = SummaryWriter(logname)

    simulate_steps(env,writer,DATASET_PATH)
    
    writer.close()







