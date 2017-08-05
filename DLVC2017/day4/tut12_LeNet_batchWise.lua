require 'nn'
require 'cunn'
require 'torch'
require 'math'
require 'optim'

trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(classes[trainset.label[100]])

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double()

function trainset:size() 
    return self.data:size(1) 
end

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5))  -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())    -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                          -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                          -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                          -- non-linearity 
net:add(nn.Linear(84, 10))                  -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                    -- converts the output to a log-probability. Useful for classification problems

print('Lenet5\n' .. net:__tostring());

-- net = net:cuda()

criterion = nn.ClassNLLCriterion()
-- criterion = criterion:cuda()

--parameters, gradParameters = net:getParameters()

local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = net:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         net:backward(inputs, df_do)
--[[
         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end
--]]
         -- return f and df/dX
         return f, gradParameters
      end

batch = 60
state = {
        learningRate = 0.01,}
        
optimMethod = optim.sgd 

parameters, gradParameters = net:getParameters()

for epcoh = 1,5 do
    totalErr = 0	
    -- local shuffle = torch.randperm(trainData:size())
    for temp = 1, trainset:size()-batch, batch do
        inputs = trainset.data[{{temp,temp+batch},{},{},{}}]
        targets = trainset.label[{{temp,temp+batch}}]
        err = 0
        _, err = optimMethod(feval, parameters, state)
        totalErr = err[1] + totalErr
    end
    print('Error', totalErr/batch)
end


---- Test the network
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')
