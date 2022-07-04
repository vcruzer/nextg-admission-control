import operator
import random
import numpy as np
import pickle
from utils_parameters import MAX_VALS, MIN_VALS, RESOURCES, REQUEST_CLASSES
import utils_parameters as param
import utils


OVERBOOK = False

#A request and its attributes
class Request:
    def __init__(self,r_class,total_num_steps,request_id):

        #duration_ts = int(np.random.exponential(30))
        duration_ts = int(np.random.exponential(5))
        self.duration = duration_ts if duration_ts>1 else 2
        self.total_patience = int(self.duration/2)
        self.priority = random.randrange(0,3)
        self.clazz = r_class
        self.id = request_id

        self.profit_tunit = -1 #updated at the time of acceptance
        self.patience_end = -1 #updated when joining the queue
        self.job_end = -1 #updated when job starts executing
        self.arrival_ts = -1 #timestep of arrival

        #TODO Use reference values
        base_quantity = self._set_base_values(r_class)

        self.resource_quantity = {}
        for resource in RESOURCES:
            self.resource_quantity[resource] = base_quantity[resource] * random.uniform(0.8,1.2)
    
    def profit_per_tunit(self,total_price):
        self.profit_tunit = total_price/self.duration
    
    #update variables to indicate request has entered the waiting queue
    def queue_join(self,timestep):
        self.patience_end = timestep + self.total_patience

    #update variables to indicate the request was accepted and is executing
    def start_job(self,timestep):
        self.job_end = timestep + self.duration
    
    def print_info(self):
        print(f"Request {self.id}: {REQUEST_CLASSES[self.clazz]}")
        print(f" Priority: {self.priority}")
        print(f" Patience(end_time): {self.total_patience} ({self.patience_end})")
        print(f" Job Duration (end): {self.duration} ({self.job_end})")
        print(f" Resource Usage:")
        for i,r in enumerate(RESOURCES):
            print(f"  {RESOURCES[i]}: {self.resource_quantity[r]}")

    def _set_base_values(self,r_class):
        
        class_quantity = {}

        if(r_class == 0): #eMBB
            class_quantity['storage'] = param.EMBB_BASE_STORAGE
            class_quantity['computing'] = param.EMBB_BASE_COMPUTING
            class_quantity['bandwidth'] = param.EMBB_BASE_BAND
        elif(r_class==1): #mMTC
            class_quantity['storage'] = param.MMTC_BASE_STORAGE
            class_quantity['computing'] = param.MMTC_BASE_COMPUTING
            class_quantity['bandwidth'] = param.MMTC_BASE_BAND
        elif(r_class==2): #URLLC
            class_quantity['storage'] = param.URLLC_BASE_STORAGE
            class_quantity['computing'] = param.URLLC_BASE_COMPUTING
            class_quantity['bandwidth'] = param.URLLC_BASE_BAND
        else:
            print("Wrong Request Class ID")
            exit()
        
        return class_quantity

    def current_resource_usage(self,resource):
        return self.resource_quantity[resource] * max(0.2,min(np.random.normal(0.5,0.1),1.0))

#Sorted by Request Priority (Higher on top)
class PriorityQueue:
    def __init__(self):

        self.queue = []
        self.time_step = 0
        self.num_timeouts = 0
        self.sum_waiting_time = 0
        self.num_requests_out = 0 #number of requests that left the queue
    
    def pull(self,idx=0):
        self.num_requests_out +=1
        request = self.queue.pop(idx) if(len(self.queue)>idx) else None
        #print(f' Request {request.id} popped ({self.num_requests_out})')
        #return self.queue.pop(idx) if(len(self.queue)>0) else -1
        return request
    
    def insert(self,request):

        self.queue.append(request)
        self.queue.sort(key=operator.attrgetter('priority'),reverse=True) #highest priority on top of the queue
       
    #managing queue and dropping requests
    def update_time(self,current_time_step):
        size = len(self.queue); i=0; cid=0
        while(i<size):
            if(self.queue[i].patience_end < current_time_step):
                request = self.pull(i)
                #print(f' Request {request.id} timed out')
                self.update_average_waiting(current_time_step,request.arrival_ts)
                size-=1
                self.num_timeouts+=1
            else:
                i+=1
            
            cid+=1


        self.time_step=current_time_step
    
    def update_average_waiting(self,current_timestep, request_arrival_timestep):

        self.sum_waiting_time += current_timestep - request_arrival_timestep

        #print(f'Current ts: {current_timestep} | Arrival Ts: {request_arrival_timestep}')
        #print(f'Average: {self.sum_waiting_time/self.num_requests_out}')
    
    def get_average_waiting_time(self):

        return self.sum_waiting_time/self.num_requests_out #average waiting time from requests
    
    def reset(self):

        self.queue = []
        self.time_step = 0
        self.num_timeouts = 0
        self.sum_waiting_time = 0
        self.num_requests_out = 0

    def __getitem__(self, key):
        return self.queue[key]

    def __setitem__(self, key, value):
        self.queue[key] = value
    
    def __len__(self):
        return len(self.queue)

class FCFSQueue(PriorityQueue):  
    def insert(self,request):
        self.queue.append(request)

#Environment with Queue and Resources
class Environment():
    def __init__(self, inp_resources, total_time_steps, loadpath="", poisson_lambda=1, multi_arrival=True, priorityq=True):
        self.p_queue = PriorityQueue() if priorityq else FCFSQueue() #choosing Queue type
        
        self.state_size = 10
        self.empty_state = [0 for _ in range(self.state_size)] 

        self.state = self.empty_state.copy()
        self.executing = [] #jobs executing
        self.completed = [] #jobs completed

        self.total_inp_resources = inp_resources
        self.free_inp_resources = inp_resources.copy()

        self.unit_price = {RESOURCES[0]:0.01, RESOURCES[1]:1, RESOURCES[2]:0.05}
        self.max_profit = 100_000 #for normalization
        self.scaling = 1e-4
        self.accepted_per_class = [0 for _ in range(len(REQUEST_CLASSES))]
        self.total_num_requests = [0 for _ in range(len(REQUEST_CLASSES))]

        if(loadpath):
            with open(loadpath,'rb') as f: 
                self.dset = pickle.load(f)
        else:
            self.dset = intialize_dataset(total_time_steps,poisson_lambda,multi_arrival) 

        self.alpha = 1.2
        self.beta = 1.0
        self.cost = 0 
    
    #taking step in the environment based on $action
    def step(self,action,timestep):

        total_profit =0
        if(len(self.p_queue)>0) and (action<2): #if queue is not empty
            request = self.p_queue.pull() #if accepted or rejected it will be removed from the queue 
            self.p_queue.update_average_waiting(timestep,request.arrival_ts)
            
            if(action==1): #.if request accepted
                self.accepted_per_class[request.clazz]+=1

                request.start_job(timestep) #calculate timestep $t$ in which the job will finish
                self._consume_inp_resources(request) #remove resources from INP

                total_profit = self._calc_profit(request) #calculate total profit from accepting the request
                request.profit_per_tunit(total_profit) #updates the profit on the request

                self.executing.append(request) #job is now executing in the INP

                #print(f" Accepted {request.id} - Ends at Timestep: {request.job_end}")
                #request.print_info()
            else: #.rejected
                #print(f"Rejected {request.id}")
                del request
        
        #reward = self._calc_timestep_reward() * penalty #reward from the current running jobs
        reward = total_profit #reward for accepting current request
        self._update_executing(timestep) #update finished jobs
        self._update_queue(timestep+1)  #timeouts and new insertions for the next state

        next_state = self._update_state()
        self.state = next_state

        return next_state, reward
 
    def reset(self):

        self.executing = []
        self.completed = []
        self.total_num_requests = [0 for _ in range(len(REQUEST_CLASSES))]
        self.accepted_per_class = [0 for _ in range(len(REQUEST_CLASSES))]
        self.p_queue.reset()
        self.free_inp_resources = self.total_inp_resources.copy()

        self._update_queue(0) 
        self.state = self._update_state()

        return self.state

    #####################################
    #.moving into the next state
    def _update_state(self,timestep=0):

        request_requirements = []; inp_free = []; request_features = []; next_state = []
        for resource in RESOURCES:
            inp_free.append(self.free_inp_resources[resource])


        if(len(self.p_queue)>0):
            next_request = self.p_queue[0]
            for resource in RESOURCES:
                request_requirements.append(next_request.resource_quantity[resource])
                
            #request_features = [next_request.clazz]+request_requirements+[next_request.priority+1, next_request.duration]
            time_left = timestep - next_request.patience_end
            request_features =  [next_request.clazz+1]+request_requirements+[next_request.priority+1, next_request.duration, time_left]
        
        else: #if queue is empty
            request_features = [0 for _ in range(self.state_size-len(inp_free))]
        
        next_state = inp_free+request_features #input features with request and INP information

        return np.asarray(next_state)

    #calc total profit and update it inside the request
    def _calc_profit(self,request):
    
        base_price = 0; base_cost = 0
        #max_val_d = {'storage': param.EMBB_BASE_STORAGE*1.2, 'computing':param.EMBB_BASE_COMPUTING*1.2, 'bandwidth':param.EMBB_BASE_BAND*1.2} 

        #calculate base price of resources
        for r in RESOURCES:
            current_price= self.unit_price[r] * request.resource_quantity[r]
            current_cost = self.unit_price[r] * 0.1 * request.resource_quantity[r]

            supply = self.free_inp_resources[r]
            demand = 0
            #IDEA: Might want to get a demand window (same size as the image)
            if(len(self.p_queue)>0):
                for request in self.p_queue.queue:
                    demand += request.resource_quantity[r] #num of requests for this resource waiting inside the queue
                
                demand /= len(self.p_queue) #average demand

                ds_factor = max(0.8,min(5,pow(demand,self.alpha) / pow(supply,self.beta)))

                current_price*=ds_factor
      
            base_price +=current_price
            base_cost += current_cost

        priority = request.priority+1
        revenue = priority * base_price * request.duration
        self.cost = base_cost * request.duration

        total_profit = revenue - self.cost

        return self.scaling* total_profit #scaling the profit to help with stability

    
    #remove jobs and release resources if it finished executing
    def _update_executing(self,timestep):
  
        size = len(self.executing); i =0
        while(i<size):
            if(self.executing[i].job_end <= timestep): # if job finishes at the end of this timestep
                done_job = self.executing.pop(i) #remove from state
                self.completed.append(done_job) #send to complete list
                self._release_inp_resources(done_job)
                size=-1
            else:
                i+=1

    #add new arrivals and remove timed out requests
    def _update_queue(self,timestep):

        self.p_queue.update_time(timestep) #remove requests from the queue that have run out of patience

        #adds new arrivals
        if(self.dset[timestep]):
            for request in self.dset[timestep]:
                #print(f"\n({timestep})Adding Request {request.id}\n")
                request.arrival_ts = timestep
                self.p_queue.insert(request)
                self.total_num_requests[request.clazz] +=1

    def _consume_inp_resources(self,request):
        for resource in RESOURCES:
            self.free_inp_resources[resource] -= request.resource_quantity[resource]
            #if(self.free_inp_resources[resource] < 0):
            #    print(f"Warning {resource} at {self.free_inp_resources[resource]}")
    
    def _release_inp_resources(self,request):
        for resource in RESOURCES: 
            self.free_inp_resources[resource] += request.resource_quantity[resource]

    def check_overbooking(self):
        for resource in RESOURCES:
            if(self.free_inp_resources[resource] < 0):
                return -1
        return 1


#TODO Change state_size Order to match actual order (channels first)
class ImageEnvironment(Environment):
    def __init__(self, inp_resources, total_time_steps, height=5, loadpath="", poisson_lambda=1, multi_arrival=True, priorityq=True):
        super().__init__(inp_resources, total_time_steps, loadpath, poisson_lambda, multi_arrival, priorityq)
        
        self.state_size = (7,height,1) #width,height,channels
        self.empty_state = np.zeros(shape=(self.state_size[-1],self.state_size[1],self.state_size[0]))
        self.state = self.empty_state.copy()
        self.current_free = self.total_inp_resources.copy()

    def step(self,action,timestep):

        total_profit =0
        if(action>0) and (action<=len(self.p_queue)):

            request_idx = action-1 #the request from the queue we accepted
            request = self.p_queue.pull(request_idx) #pull chosen request

            self.p_queue.update_average_waiting(timestep,request.arrival_ts)
            self.accepted_per_class[request.clazz]+=1 #increment # of accepted requests from the class

            request.start_job(timestep) #calculate timestep that the job will finish
            self._consume_inp_resources(request) #remove resources from INP
            total_profit = self._calc_profit(request) #calculate total profit and updates it in request
            request.profit_per_tunit(total_profit)
            self.executing.append(request) #job is now executing
        #else:
        elif (action==0): #.rejected first request from the queue
            request = self.p_queue.pull()
            if(request):
                self.p_queue.update_average_waiting(timestep,request.arrival_ts)
                del request
    
        reward = total_profit
        self._update_executing(timestep) #update finished jobs
        self._update_queue(timestep+1)  #timeouts and new insertions for the next state


        next_state = self._update_state(timestep) #updating state based in the current action
        self.state = next_state

        return next_state, reward
 
    #####################################
    #.moving into the next state
    def _update_state(self,timestep=0):

        inp_free = [0 for _ in range(self.state_size[0])] 
        request_features = [0 for _ in range(self.state_size[1]-1)]

        #normalizing resources values to be similar to gray scale image 0-255
        for i,resource in enumerate(RESOURCES):
            inp_free[i] = utils.normalize_range(self.free_inp_resources[resource],0,self.total_inp_resources[resource],0,255)
            
        
        for i in range(0,self.state_size[1]-1,1):
            request_resources = []
            if(i < len(self.p_queue)):
                next_request = self.p_queue[i]
                for j,resource in enumerate(RESOURCES):
                    request_resources.append(utils.normalize_range(next_request.resource_quantity[resource],MIN_VALS[resource],MAX_VALS[resource],0,255))

                #S1
                time_left = timestep - next_request.patience_end
                current_features =  [next_request.clazz+1]+request_resources+[next_request.priority+1, next_request.duration, time_left]
                #current_features =  [next_request.clazz+1]+request_resources+[next_request.priority+1, next_request.duration, next_request.total_patience]
            else:
                current_features = [0 for _ in range(self.state_size[0])]
            
            request_features[i] = current_features


        image_state = [inp_free] + request_features #inp free resources info + request attributes info
        next_state = np.asarray([image_state],dtype=float)

        #S4
        #image_state = np.asarray([inp_free] + request_features)
        #next_state = np.zeros(shape=(self.state_size[-1],self.state_size[1],self.state_size[0]))
        #next_state[0] = self.state.copy()[1]
        #next_state[1] = image_state

        return next_state

    def _current_resources_consumption(self):
        current_free_resources = self.total_inp_resources.copy()
        for request in self.executing:
            for resource in RESOURCES:
                current_usage = request.current_resource_usage(resource)
                current_free_resources[resource] -= current_usage
        
        overbook = 0
        for key in current_free_resources:
            if(current_free_resources[key] < 0):
                overbook=1
                break
        
        
        return current_free_resources,overbook


#########################################################
#generates which requests will arrive at a given time t
def intialize_dataset(num_timesteps,plambda,multiarrival):
    arrivals = np.random.poisson(plambda,num_timesteps+1)
    dataset = [None for _ in range(num_timesteps+1)]

    def create_request(request_id,t):
        class_id  = random.randrange(0,3) #random class
        dataset[t].append(Request(class_id,num_timesteps,request_id)) #create request
        dataset[t][-1].queue_join(t) #update patience duration
        dataset[t][-1].print_info()


    rid = 0
    for t in range(num_timesteps+1):
        print(f"TimeStep {t}")
        if(arrivals[t] > 0):
            dataset[t] = [] 
            if(multiarrival): # if multiple requests can arrive at the same time
                for _ in range(arrivals[t]):
                    print(f" Adding Request {rid}")
                    create_request(rid,t)
                    rid+=1
            else:
                create_request(rid,t)
                rid+=1

    return dataset