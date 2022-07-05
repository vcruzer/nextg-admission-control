import torch.nn.functional as F
import torch

#Standard FC DDQN based on Van etal
class DDQN(torch.nn.Module):
    def __init__(self,state_size,num_actions):
        super(DDQN, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 12)

        self.fc_value = torch.nn.Linear(12, 6)
        self.fc_adv = torch.nn.Linear(12, 6)

        self.value = torch.nn.Linear(6, 1)
        self.adv = torch.nn.Linear(6, num_actions)
        
        '''
        One-INP
        self.fc1 = torch.nn.Linear(state_size, 64)

        self.fc_value = torch.nn.Linear(64, 128)
        self.fc_adv = torch.nn.Linear(64, 128)

        self.value = torch.nn.Linear(128, 1)
        self.adv = torch.nn.Linear(128, num_actions)
        '''
        self.dropout = torch.nn.Dropout(0.1)
        self.dropout_val = torch.nn.Dropout(0.4)
        self.dropout_adv = torch.nn.Dropout(0.4)
        
    def forward(self, state):
        y = F.relu(self.fc1(state))
        y = self.dropout(y)

        value = F.relu(self.fc_value(y))
        value = self.dropout_val(value)

        adv = F.relu(self.fc_adv(y))
        adv = self.dropout_adv(adv)

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1) #picking action with max reward
        return action_index.item()

#CNN Dueling DQN
class CNN_DDQN(torch.nn.Module):
    def __init__(self,state_size):
        super(CNN_DDQN, self).__init__()

        width, height, num_channels = state_size
        num_actions = height
        
        self.conv1 = torch.nn.Conv2d(num_channels, 16, kernel_size=3)
        #self.conv1 = torch.nn.Conv2d(num_channels, 6, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=2)
        #self.conv2 = torch.nn.Conv2d(6, 3, kernel_size=2)
        self.bn2 = torch.nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 1, padding=0):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(width,kernel_size=3),kernel_size=2)
        convh = conv2d_size_out(conv2d_size_out(height,kernel_size=3),kernel_size=2)
        linear_input_size = convw * convh * 32 #192

        self.fc_value = torch.nn.Linear(linear_input_size, 128)
        self.fc_adv = torch.nn.Linear(linear_input_size, 128)

        self.value = torch.nn.Linear(128, 1)
        self.adv = torch.nn.Linear(128, num_actions)

        self.dropout = torch.nn.Dropout(0.25)
        self.dropout_val = torch.nn.Dropout(0.4)
        self.dropout_adv = torch.nn.Dropout(0.4)
        
    def forward(self, state):

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.bn1(self.conv1(state)))
        #x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        
        value = F.relu(self.fc_value(x))
        value = self.dropout_val(value)

        adv = F.relu(self.fc_adv(x))
        adv = self.dropout_adv(adv)

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1) #picking action with max reward
        return action_index.item()

#CNN DQN
class CNN_DQN(torch.nn.Module):
    def __init__(self,state_size):
        super(CNN_DQN, self).__init__()

        width, height, num_channels = state_size
        num_actions = height
        
        self.conv1 = torch.nn.Conv2d(num_channels, 16, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=2)

        def conv2d_size_out(size, kernel_size = 5, stride = 1, padding=0):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(width,kernel_size=3),kernel_size=2)
        convh = conv2d_size_out(conv2d_size_out(height,kernel_size=3),kernel_size=2)
        linear_input_size = convw * convh * 32 #192

        self.fc1 = torch.nn.Linear(linear_input_size, 128)
        self.fc2 = torch.nn.Linear(128, num_actions)

        self.dropout = torch.nn.Dropout(0.4)
        
    def forward(self, state):

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        Q = self.fc2(x)

        return Q
