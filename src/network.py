import time
import torch
from torch import nn
from config import INPUT_DIM, OUTPUT_DIM

'''
Input: 
    max_gas min_gas mean_gas std_gas GO(#17)
Output:
    gastric cancer or not (binary classification)
'''
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(INPUT_DIM, 50),
            nn.PReLU(),
            nn.Linear(50, 100),
            nn.PReLU(),
            nn.Linear(100, 200),
            nn.PReLU(),
            nn.Linear(200, 300),
            nn.PReLU(),
            nn.Linear(300, OUTPUT_DIM),
            nn.Softmax()
        )
        #layer1 = nn.Linear(INPUT_DIM, 50)
        #torch.nn.init.kaiming_normal(layer1.weight, ....)
        #torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.xavier_normal_(conv)

    def forward(self, x):
        return self.conv(x)

        
if __name__ == '__main__':
    model = Network()
    inputs = torch.zeros((2, INPUT_DIM))
    model.train(True)
    a = time.time()
    output = model(inputs)
    b = time.time()
    print(output.size())
    print((b-a)*1000)
