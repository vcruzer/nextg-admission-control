
#.Hyperparameter Search with random search via Ray Tune

import torch
import torch.nn.functional as F
import utils_models, utils_replay_buffers, utils_nextg
from generate_dset import NUM_TIME_STEPS
from utils_parameters import INP_RESOURCES
from utils import check_resources
import random
import os, shutil
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

INITIAL_EPSILON = 0.99 
FINAL_EPSILON = 0.1 
NUM_TIME_STEPS = 1000
REMOVE = True
SAVE = False
COMMON = '../'

def initialize(config,dset_path):

    env = utils_nextg.ImageEnvironment(INP_RESOURCES,NUM_TIME_STEPS,height=config['height'],loadpath=dset_path, priorityq=False)#config['priorityq_enable'])
    input_size = env.state_size
    num_actions = env.state_size[1]

    main_ddqn = utils_models.CNN_DDQN(input_size).to(device)
    target_ddqn = utils_models.CNN_DDQN(input_size).to(device)     

    target_ddqn.load_state_dict(main_ddqn.state_dict()) #loads weights from main ddqn to target

    optimizer = torch.optim.Adam(main_ddqn.parameters(), lr=config['lr'])

    if(config['replay_type']=='priority'):
        memory_replay = utils_replay_buffers.PrioritizedMemory(config['replay_memory'])

    elif(config['replay_type']=='balance'):
        memory_replay = utils_replay_buffers.BalancedMemory(config['replay_memory'],num_actions)
    else:
        memory_replay = utils_replay_buffers.Memory(config['replay_memory'])

    return env,memory_replay,main_ddqn,target_ddqn,optimizer

def get_ytrue(target_ddqn,main_ddqn, batch_next_state,batch_reward,config):

    with torch.no_grad():

        #for Double DQN
        targetQ_next = target_ddqn(batch_next_state)

        mainQ_next = main_ddqn(batch_next_state)
        main_max_action = torch.argmax(mainQ_next, dim=1, keepdim=True) #picking action with max reward

        next_batch_actions = targetQ_next.gather(1, main_max_action.long()) #uses index from main network on the Q values from target network

        y = batch_reward + (config['gamma'] * next_batch_actions)

    return y

def train_model(main_ddqn,target_ddqn, batch, optimizer,config,scheduler=None,is_weights=None):

    batch_state, batch_next_state, batch_action, batch_reward = zip(*batch)

    batch_state = torch.FloatTensor(batch_state).to(device)
    batch_next_state = torch.FloatTensor(batch_next_state).to(device)
    batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)

    y = get_ytrue(target_ddqn,main_ddqn,batch_next_state,batch_reward,config)

    optimizer.zero_grad()

    #These are the actions which would've been taken for each batch state according to main_ddqn
    y_pred = main_ddqn(batch_state).gather(1, batch_action.long())

    if(config['replay_type']=="priority"):
        weights_tensor  = torch.FloatTensor(is_weights).unsqueeze(1).to(device)
        loss =  (weights_tensor * F.mse_loss(y_pred, y,reduction='none')).mean()
    else:
        loss = F.mse_loss(y_pred, y)
    
    loss.backward()

    if(config['max_norm'] != 0.0):
        torch.nn.utils.clip_grad_norm_(main_ddqn.parameters(), max_norm=config['max_norm'])

    optimizer.step()
    #scheduler.step()

    return loss.item(), y_pred, y

def simulate_steps(config,dset_path):

    env,memory_replay,main_ddqn, target_ddqn, optimizer = initialize(config,dset_path)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    num_actions = env.state_size[1]

    if(config['rollback']):
        best_ddqn = utils_models.CNN_DDQN(env.state_size).to(device)
        best_ddqn.load_state_dict(main_ddqn.state_dict())

    if (config['regret']):
        fcfs_results = np.load(f'{COMMON}output/FCFS_ts_reward_{dset.split("/")[-1][:-4]}.npy')
        fcfs_av_reward = np.mean(fcfs_results)

    epsilon = INITIAL_EPSILON
    learn_steps = 0; is_weights = None; best_reward =0

    for epoch in range(config['epochs']):
        state = env.reset(); episode_reward = 0; action = 0; reward =0; last_loss=0; accum_loss=0
        counter = 0
        for time_step in range(NUM_TIME_STEPS):

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
            
            if(action>0) and (len(env.p_queue) > action-1 ) and (check_resources(env,action-1)):
                action = 0

            next_state, reward = env.step(action,time_step) #apply action
            episode_reward += reward

            if(config['regret']):
                reward = reward - fcfs_av_reward

            if(config['replay_type']=='priority'):
                with torch.no_grad():
                    tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                    y_pred = main_ddqn(tensor_state)[0][action]

                tensor_next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                y = get_ytrue(target_ddqn,main_ddqn,tensor_next_state,reward,config)

                error = abs(y_pred - y).to('cpu')
                memory_replay.add(error,(state, next_state, action, reward))
            else:
                memory_replay.add((state, next_state, action, reward))

            #.training with memory replay
            if memory_replay.size() > 128:

                learn_steps+=1

                #updating target ddqn after a set of steps
                if (learn_steps % config['update_steps'] == 0):
                    target_ddqn.load_state_dict(main_ddqn.state_dict())

                if(config['replay_type']=='priority'):
                    batch, idxs, is_weights = memory_replay.sample(config['batch_size'])
                else:
                    batch = memory_replay.sample(config['batch_size']) #picking random samples from the replay buffer

                last_loss, ypred, y = train_model(main_ddqn,target_ddqn,batch,optimizer,config,scheduler,is_weights)

                accum_loss +=last_loss
                counter+=1

                if(config['replay_type']=="priority"):
                    errors = torch.abs(y- ypred).data.to('cpu').numpy()

                    # update priority
                    for i in range(config['batch_size']):
                        idx = idxs[i]
                        memory_replay.update(idx, errors[i])


                #decay exploration
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / config['explore']

            #moving to next state
            state = next_state

        if(config['rollback'] and learn_steps>1):
            #Rollback
            if (episode_reward>best_reward):
                best_reward = episode_reward
                best_ddqn.load_state_dict(main_ddqn.state_dict())
                target_ddqn.load_state_dict(main_ddqn.state_dict())
            elif(1- (episode_reward/best_reward ) > 0.05 ): # if we are getting error higher than 5% we out of track, get back to best model
                main_ddqn.load_state_dict(best_ddqn.state_dict())
                target_ddqn.load_state_dict(best_ddqn.state_dict())
                scheduler.step()

        #.reporting data to Ray Tune
        tune.report(loss=last_loss, score=episode_reward)

        if(SAVE) and (epoch % 10):
            torch.save(main_ddqn.state_dict(), 'ddqn-main.param')
    
    return learn_steps

if(__name__=='__main__'):

    config_df = pd.DataFrame()

    for dset in os.listdir(f'{COMMON}dataset'):
        print(dset)
        ray.init()

        dset_path = f'{COMMON}dataset/'+dset
        #parameter variations (note that we are not testing different model architectures)
        config = {
            "gamma": tune.grid_search([0.1,0.4,0.7,0.9]),
            "replay_type": tune.grid_search(['priority','normal','balance']),
            "replay_memory": tune.choice([5_000, 50_000, 500_000, 1_000_000]),
            "update_steps": tune.sample_from(lambda _: np.random.randint(1,100)),
            "explore": tune.sample_from(lambda _: np.random.randint(600, 10000)),
            "lr": tune.loguniform(1e-4, 1e-1),
            "max_norm": tune.choice([0.0,0.5,1.0,1.5,2.0,2.5]),
            "batch_size": tune.choice([16,32,64,128]),
            "epochs": tune.grid_search([30,60,100]),
            "regret": tune.choice([True,False]),
            "rollback": tune.choice([True,False]),
            "priorityq_enable": tune.choice([True,False]),
            "height": tune.grid_search([5,7,10])

        }

        scheduler = ASHAScheduler(
            metric='score',
            mode="max",
            #max_t=100,
            #grace_period=10,
            reduction_factor=2)

        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "score", "training_iteration"])
    

        result = tune.run(
            tune.with_parameters(simulate_steps, dset_path=dset_path),
            resources_per_trial={"cpu": 2},
            config=config,
            num_samples=50,
            scheduler=scheduler,
            progress_reporter=reporter,
            log_to_file=(f"{COMMON}tune_stdout.log", f"{COMMON}tune_stderr.log")
            )

        best_trial = result.get_best_trial("loss", "min", "last")
        print(f'Dataset: {dset}')
        print(" Best trial config: {}".format(best_trial.config))
        print(" Best trial final loss: {}".format(best_trial.last_result["loss"]))
        print(" Best trial final Score: {}".format(best_trial.last_result["score"]))
        
        #saving best config
        row = dict(best_trial.config); row['dataset'] = dset; row['reward'] = best_trial.last_result["score"]
        config_df = config_df.append(row, ignore_index=True)
        ray.shutdown()

    config_df.to_csv(f'{COMMON}params/config.csv',index=False)
   

    #for filename in os.listdir('/home/vcruz/ray_results/'):
    #    if('simulate_steps' in filename):
    #        shutil.move(f'/home/vcruz/ray_results/{filename}', f'{COMMON}ray_log/')






