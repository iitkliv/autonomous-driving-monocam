require 'nn'
local net = nn.Sequential()
net:add(nn.Linear(1024,100))
net:add(nn.Linear(100,10))
net:add(nn.View())
net:add(nn.LogSoftMax())

return net
