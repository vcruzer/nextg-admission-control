
import torch
import torch.nn.functional as F
import utils_replay_buffers,utils_models, utils_nextg
from utils_parameters import RESOURCES, PRIORITY, DATASET_PATH, INP_RESOURCES, NUM_TIME_STEPS, REQUEST_CLASSES
from utils_nextg import OVERBOOK
from utils import check_resources, get_params
import numpy as np
import random
import os, shutil
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INITIAL_EPSILON = 0.9 
FINAL_EPSILON = 0.1 

OPTIMAL = False #use parameters from file
CNN = True #use CNN with Image environment
DUELING = True #use Dueling DQN architecture
SAVE_MODEL = False #checkpoint save the model

#initialize environment and models
def initialize(dset_path,height,cnn=True,dueling=True):
    if(cnn):
        env = utils_nextg.ImageEnvironment(INP_RESOURCES,NUM_TIME_STEPS,height=height,loadpath=dset_path, priorityq=PRIORITY)
        input_size = env.state_size
        num_actions = env.state_size[1]

        if(dueling):
            main_ddqn = utils_models.CNN_DDQN(input_size).to(device)
            target_ddqn = utils_models.CNN_DDQN(input_size).to(device)     
            best_ddqn = utils_models.CNN_DDQN(input_size).to(device)
        else:
            main_ddqn = utils_models.CNN_DQN(input_size).to(device)
            target_ddqn = utils_models.CNN_DQN(input_size).to(device)     
            best_ddqn = utils_models.CNN_DQN(input_size).to(device)

    else:
        env = utils_nextg.Environment(INP_RESOURCES,NUM_TIME_STEPS,loadpath=dset_path, priorityq=PRIORITY)
        input_size = env.state_size
        num_actions = 2

        if(dueling):
            main_ddqn = utils_models.DDQN(input_size,num_actions).to(device)
            target_ddqn = utils_models.DDQN(input_size,num_actions).to(device)
            best_ddqn = utils_models.DDQN(input_size,num_actions).to(device)
        #else:
        #    main_ddqn = utils_models.DQN(input_size).to(device)
        #    target_ddqn = utils_models.DQN(input_size).to(device)     
        #    best_ddqn = utils_models.DQN(input_size).to(device)
        
        target_ddqn.load_state_dict(main_ddqn.state_dict()) #loads weights from main ddqn to target
    best_ddqn.load_state_dict(main_ddqn.state_dict())

    return env,input_size,num_actions,main_ddqn,target_ddqn,best_ddqn

#getting 'ground_truth' from target dqn
def get_ytrue(target_ddqn,main_ddqn, batch_next_state,batch_reward,gamma):

    with torch.no_grad():

        if(DUELING): #for Double DQN
            targetQ_next = target_ddqn(batch_next_state)

            mainQ_next = main_ddqn(batch_next_state)
            main_max_action = torch.argmax(mainQ_next, dim=1, keepdim=True) #picking action with max reward

            next_batch_actions = targetQ_next.gather(1, main_max_action.long()) #uses index from main network on the Q values from target network

            #y = batch_reward + (gamma * next_batch_actions)
        else:   #for classic DQN
            future_Qs = target_ddqn(batch_next_state)
            future_actions = torch.argmax(future_Qs, dim=1, keepdim=True) 
            next_batch_actions = future_Qs.gather(1, future_actions.long())

        y = batch_reward + (gamma * next_batch_actions)

    return y


#training DQN in batch of data from replay buffer
def train_model(main_ddqn,target_ddqn, batch, optimizer,params,scheduler=None,is_weights=None):

    batch_state, batch_next_state, batch_action, batch_reward = zip(*batch)

    batch_state = torch.FloatTensor(batch_state).to(device)
    batch_next_state = torch.FloatTensor(batch_next_state).to(device)
    batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)

    y = get_ytrue(target_ddqn,main_ddqn,batch_next_state,batch_reward,params['gamma'])

    optimizer.zero_grad()

    #These are the actions which would've been taken for each batch state according to main_ddqn
    y_pred = main_ddqn(batch_state).gather(1, batch_action.long())

    if(params['replay_type']=="priority"):
        weights_tensor  = torch.FloatTensor(is_weights).unsqueeze(1).to(device)
        loss =  (weights_tensor * F.mse_loss(y_pred, y,reduction='none')).mean()
    else:
        loss = F.mse_loss(y_pred, y)
    
    loss.backward()
    if(params['max_norm'] > 0): # if clipping enabled
        torch.nn.utils.clip_grad_norm_(main_ddqn.parameters(), max_norm=params['max_norm']) #gradient clipping to improve network convergence
    optimizer.step()
    #scheduler.step()

    return loss.item(), y_pred, y


#simulate steps within the environment with the current DQN model
def simulate_steps(env,memory_replay,models,optimizer,writer,num_actions,params,reward_to_beat=0):

    epsilon = INITIAL_EPSILON; epochs = range(params['epochs'])
    learn_steps = 0; best_reward = 0; is_weights = None

    #return metrics
    reward_per_epoch = np.empty(len(epochs),dtype=float)
    accepted_jobs = np.empty(len(epochs),dtype=int)
    timeouts= np.empty_like(accepted_jobs)
    average_waiting = np.empty_like(reward_per_epoch)
    acc_pct = {clazz:np.empty_like(reward_per_epoch) for clazz in REQUEST_CLASSES}

    #main_ddqn, target_ddqn = models
    main_ddqn, target_ddqn, best_ddqn = models
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in epochs:
        state = env.reset(); episode_reward = 0; action = 0; reward =0; last_loss=0; accum_loss=0
        counter = 0
        for time_step in range(NUM_TIME_STEPS):

            print(f"Epoch {epoch+1}/{params['epochs']} (Loss: {last_loss:.4f} e: {epsilon:.2f}) | Time Step {time_step} -> Queue Size: {len(env.p_queue)} | Last Reward: {reward:.2f} | Executing: {len(env.executing)} | Completed: {len(env.completed)} |", end='\r')
            #print(f"Epoch {epoch} (Loss: {last_loss:.4f} e: {epsilon:.2f}) | Time Step {time_step} -> Queue Size: {len(env.p_queue)} | Last Reward: {reward:.2f} | Executing: {len(env.executing)} | Completed: {len(env.completed)} ")
            
            #.Choosing an action for the current state
            p = random.random(); q_vals=None
            if p < epsilon:
                action = random.randint(0, num_actions-1)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_vals = main_ddqn(tensor_state)
                    action_index = torch.argmax(q_vals, dim=1)

                action = action_index.item()
                #action = main_ddqn.select_action(tensor_state) #choosing an action for the current state
            
            #in case we are allocating more than available by accepting, force action to be reject
            if(action>0) and (len(env.p_queue) > action-1 ) and (check_resources(env,action-1)):
                action=0
            

            next_state, reward = env.step(action,time_step) #apply action
            episode_reward += reward

            #update replay buffers
            if(params['replay_type']=='priority'):
                with torch.no_grad():
                    tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                    y_pred = main_ddqn(tensor_state)[0][action]

                tensor_next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                y = get_ytrue(target_ddqn,main_ddqn,tensor_next_state,reward,params['gamma'])

                error = abs(y_pred - y).to('cpu')
                memory_replay.add(error,(state, next_state, action, reward)) #experience tuple for prioritized replay buffer
            else:
                memory_replay.add((state, next_state, action, reward)) #experience for other memory replays
   

            #.training with memory replay
            if memory_replay.size() > 128:

                learn_steps+=1

                #updating target ddqn after a set of steps
                if (learn_steps % params['update_steps']== 0):
                    target_ddqn.load_state_dict(main_ddqn.state_dict())

                #sampling batch from replay buffer
                if(params['replay_type']=='priority'):
                    batch, idxs, is_weights = memory_replay.sample(params['batch_size'])
                else:
                    batch = memory_replay.sample(params['batch_size']) #picking random samples from the replay buffer

                #training with batch
                last_loss, ypred, y = train_model(main_ddqn,target_ddqn,batch,optimizer,params,scheduler,is_weights)

                accum_loss +=last_loss
                counter+=1

                #updating buffer weights in case its prioritized replay buffer
                if(params['replay_type']=="priority"):
                    errors = torch.abs(y- ypred).data.to('cpu').numpy()

                    # update priority
                    for i in range(params['batch_size']):
                        idx = idxs[i]
                        memory_replay.update(idx, errors[i])

                writer.add_scalar('Batch_loss', last_loss, global_step=learn_steps)
                writer.add_scalar('Queue Size', len(env.p_queue), global_step=learn_steps)

                #InP Environment resource usage metrics
                for r in RESOURCES:
                    resource_consumption = env.total_inp_resources[r] - env.free_inp_resources[r]
                    writer.add_scalar(f'{r} Usage', resource_consumption, global_step=learn_steps)

                #decay exploration
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / params['explore']

            #moving to next state
            state = next_state

        if(counter==0):
            counter = 1

        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        writer.add_scalar('Average loss', accum_loss/counter, global_step=epoch)
        writer.add_scalar('Accepted Jobs', len(env.executing)+len(env.completed), global_step=epoch)
        writer.add_scalar('Timeouts', env.p_queue.num_timeouts, global_step=epoch)
        writer.add_scalar('Average Waiting Time', env.p_queue.get_average_waiting_time(), global_step=epoch)

        print("")
        #Experimental Rollback with Triple DQN
        if(params['rollback'] and learn_steps>1):
            #Rollback
            if (episode_reward>best_reward):
                print(" Better reward !")
                best_reward = episode_reward
                #scheduler.step()
                best_ddqn.load_state_dict(main_ddqn.state_dict())
                target_ddqn.load_state_dict(main_ddqn.state_dict())
            
            elif (best_reward > 0) and (1- (episode_reward/best_reward ) > 0.05 ): # if we are getting error higher than 5% we are out of track, rollback to best model
                main_ddqn.load_state_dict(best_ddqn.state_dict())
                target_ddqn.load_state_dict(best_ddqn.state_dict())
                scheduler.step()
                print(f" Error Higher than 5% (New LR: {scheduler.get_last_lr()})")

        #saving metrics
        reward_per_epoch[epoch] = episode_reward
        accepted_jobs[epoch] = len(env.executing)+len(env.completed)
        timeouts[epoch] = env.p_queue.num_timeouts
        average_waiting[epoch] = env.p_queue.get_average_waiting_time()

        for c,clazz in enumerate(REQUEST_CLASSES):
            acc_pct[clazz][epoch] = env.accepted_per_class[c]/env.total_num_requests[c]


        print(f' Episode Reward: {episode_reward:.2f} (Missed by: {(reward_to_beat - episode_reward):.2f})')
        print("-------------------------------------------")

        #checkpointing the model
        if(SAVE_MODEL) and (epoch % 10):
            torch.save(main_ddqn.state_dict(), 'ddqn-main.param')
        
    metrics_per_epoch = (reward_per_epoch, accepted_jobs, timeouts, average_waiting, acc_pct)
    
    return learn_steps, metrics_per_epoch

if(__name__=='__main__'):


    reward_to_beat = 0.0
    aux_name = DATASET_PATH.split("/")[-1][:-4]
    fname = '../output/FCFS_reward_'+aux_name+'.txt'
    name_aux = 'ddqn' if DUELING else 'dqn'
    model_name = 'cnn_'+name_aux if CNN else name_aux
    dset_name = DATASET_PATH.split('/')[-1][:-4]
    logname = f'../logs/{dset_name}/S3_{model_name}_0'
    if(os.path.exists(fname)):
        with open(fname,'r') as f:
            reward_to_beat = float(f.readline())

    if(os.path.exists(logname)):
        shutil.rmtree(logname)

    writer = SummaryWriter(logname)

    #setting up some hyperparams
    params = {}
    if(OPTIMAL):
        params = get_params(dset_name+'.pkl')
    else:
        params = get_params(dset_name+'.pkl',default=True)


    env,input_size,num_actions,main_ddqn,target_ddqn,best_ddqn = initialize(DATASET_PATH,params['height'],cnn=CNN,dueling=DUELING)

    optimizer = torch.optim.Adam(main_ddqn.parameters(), lr=params['lr'])
    models = (main_ddqn,target_ddqn,best_ddqn)


    if(params['replay_type']=='priority'):
        memory_replay = utils_replay_buffers.PrioritizedMemory(params['replay_memory']) 
    elif(params['replay_type']=='balance'):
        memory_replay = utils_replay_buffers.BalancedMemory(params['replay_memory'],num_actions)
    else:
        memory_replay = utils_replay_buffers.Memory(params['replay_memory'])

    learn_steps, _ = simulate_steps(env, memory_replay, models, optimizer,writer,num_actions,params,reward_to_beat)

    writer.close()


    #Resource Threshold (total # of resources)
    logname = f'../logs/reference_workaround'
    if(os.path.exists(logname)):
        shutil.rmtree(logname)
        
    writer2 = SummaryWriter(logname)
    for r in RESOURCES:
        for ls in range(learn_steps):
            writer2.add_scalar(f'{r} Usage', env.total_inp_resources[r], global_step=ls)
    writer2.close()









