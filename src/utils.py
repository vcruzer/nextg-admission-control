import utils_nextg
import pandas as pd

#normalize between 0,1
def normalize(data,min_val,max_val):
    return (data - min_val) / (max_val - min_val)

#normalize within range min_target, max_target
def normalize_range(data,min_val,max_val,min_target,max_target):

    return normalize(data,min_val,max_val) * (max_target-min_target) + min_target

#return True if resources will be overbooked
def check_resources(env,idx=0):

    if(len(env.p_queue)>0):
        request = env.p_queue[idx]
        env._consume_inp_resources(request)
        check = env.check_overbooking()
        env._release_inp_resources(request)
    else:
        check = env.check_overbooking()

    return 0 if check>0 else 1

#deep copy environment
def copy_env(env,inp_resources,priority_enable=True):
    cEnv = utils_nextg.Environment(inp_resources,-1,priorityq=priority_enable)

    cEnv.p_queue.queue = env.p_queue.queue.copy()
    cEnv.p_queue.num_timeouts = env.p_queue.num_timeouts
    cEnv.p_queue.time_step = env.p_queue.time_step

    cEnv.state = env.state.copy()
    cEnv.executing = env.executing.copy()
    cEnv.completed = env.completed.copy()
    cEnv.free_inp_resources = env.free_inp_resources.copy()
    cEnv.dset = env.dset.copy()

    return cEnv

#retrieve parameters for network
def get_params(dset,default=False):

    params = {}

    if(default):
        params['gamma'] = 0.4 #0.1 #importance of long-short term Reward ( Closer to 1.0 means full long-term)
        params['replay_type'] = 'normal' #replay type ( normal, priority or balance)
        params['replay_memory'] = 50_000
        params['update_steps'] = 100 #40
        params['explore'] = 50 #500 #5_000 #exploration decay
        params['lr'] = 0.001
        params['max_norm'] = 2.0 #norm clipping
        params['batch_size'] = 16
        params['epochs'] = 30
        params['regret'] = False #if regret enabled must compared with other baseline
        params['rollback'] = False
        params['height'] = 5
    
    else:
        df = pd.read_csv("config.csv")
        datasets = df.pop('dataset')
        idx = datasets[datasets==dset].index[0]
        not_int_cols = ['gamma','replay_type','lr','max_norm','reward']

        for key in df.columns:
            if not (key in not_int_cols):
                params[key] = int(df[key].loc[idx])
            else:
                params[key] = df[key].loc[idx]
    
    return params