require 'torch';
require 'optim';
require 'nn';
require 'gnuplot';


LR = 0.1
--LRDecay =0
--weightDecay=1e-4
--momentum=0.9

torch.setdefaulttensortype('torch.FloatTensor')

DataPath = 'dlvc/tut05/'

local optimState = {
    learningRate = LR,
    --momentum = momentum,
    --dampening = 0,
    --weightDecay = weightDecay,
    --learningRateDecay = LRDecay
}


---------------------------------
---------------------------------

trainset = torch.DiskFile(DataPath..'train-images.idx3-ubyte')
testset = torch.DiskFile(DataPath..'t10k-images.idx3-ubyte')

Tr_Set = torch.ByteTensor(60000, 784)
Test_Set = torch.ByteTensor(10000, 784)

trainset:readByte(16)
trainset:readByte(Tr_Set:storage())
Tr_Set = Tr_Set:float():div(255) 

testset:readByte(16)
testset:readByte(Test_Set:storage())
Test_Set = Test_Set:float():div(255) 

---------------------------------
---------------------------------

trainlabels = torch.DiskFile(DataPath..'train-labels.idx1-ubyte')
testlabels = torch.DiskFile(DataPath..'t10k-labels.idx1-ubyte')

Tr_labels = torch.ByteTensor(60000)
Test_labels = torch.ByteTensor(10000)

trainlabels:readByte(8)
trainlabels:readByte(Tr_labels:storage())

testlabels:readByte(8)
testlabels:readByte(Test_labels:storage())

TrainLabels = torch.Tensor(60000,10) 
TestLabels = torch.Tensor(10000,10)  

for i=1,60000 do
  x = torch.zeros(10)
  x[Tr_labels[i] + 1] = 1
  TrainLabels[i] = x
end


for i=1,10000 do
  x = torch.zeros(10)
  x[Test_labels[i] + 1] = 1
  TestLabels[i] = x
end


print('Dataset Loaded..')
---------------------------------
---------------------------------


function network(Data,Net,Out)
  Data = Data 
  Out = Out 
  function feval(Weights)
    Gradients:zero()
    y = Net:forward(Data)
    currLoss = loss:forward(y,Out)
    local dE_dy = loss:backward(y, Out)
    Net:backward(Data, dE_dy)
    return currLoss, Gradients
 end
 optim.sgd(feval, Weights, optimState)
 return currLoss
end




function Train_Net(Net,Data_Set,Data_out,sz,Data_Test,Data_Test_labels)
  
  local TrLoss = torch.Tensor(epoch)
  local vLoss = torch.Tensor(epoch)
  local epochNo = torch.Tensor(epoch)
  

  for Tepoch = 1,epoch do
    print('epoch ' .. Tepoch .. '/' .. epoch)
    Total_Loss=0
    TestError = 0
    for loopno=1,sz do
      input = Data_Set[loopno]
      output = Data_out[loopno]
      local LossTrain = network(input,Net,output)    
      Total_Loss = Total_Loss + LossTrain      
    end
    Total_Loss = Total_Loss/sz
    print('Training Loss = ' .. Total_Loss)
    TrLoss[Tepoch] = Total_Loss
    
    Total_Loss=0
    for i=1,10000 do
      output = Net:forward(Data_Test[i] )
      currLoss = loss:forward(output,Data_Test_labels[i] )
      Total_Loss = Total_Loss + currLoss
      output = output - torch.max(output)*torch.ones(10) 
      output = torch.floor(output) + torch.ones(10) 
      result = output - Data_Test_labels[i] 
      if torch.sum(torch.abs(result)) ~= 0 then
        TestError = TestError + 1
      end 
    end
    TestingLoss = Total_Loss/10000
    TestError = TestError/100
    print('Validation Loss : '.. TestingLoss)
    print('Test Error Rate (%) : '.. TestError .. ' %')
    vLoss[Tepoch] = TestingLoss
    epochNo[Tepoch] = Tepoch
    
  end
  return epochNo,TrLoss,vLoss

end


function Train_Nett(Net,Data_Set,Data_out,sz)

  for Tepoch = 1,epoch do
    print('epoch ' .. Tepoch .. '/' .. epoch)
    Total_Loss=0
    for loopno=1,sz do
      input = Data_Set[loopno]
      output = Data_out[loopno]
      local LossTrain = network(input,Net,output)    
      Total_Loss = Total_Loss + LossTrain      
    end
    Total_Loss = Total_Loss/sz
    print('Training Loss = ' .. Total_Loss)
  end


end




------------------------------------------------------------------
------------------------------------------------------------------
--      Training       -------------------------------------------
------------------------------------------------------------------
------------------------------------------------------------------

print('AutoEncoder Training...')

-- Training Autoencoder for Initialization 
epoch = 20

local net

--Encoder

net = nn.Sequential()
net:add(nn.Linear(784, 100)) 
net:add(nn.ReLU())  
net:add(nn.Linear(100, 50)) 
net:add(nn.ReLU()) 

--Decoder

net:add(nn.Linear(50, 100)) 
net:add(nn.ReLU())  
net:add(nn.Linear(100, 784))
net:add(nn.ReLU()) 
net = net 

loss = nn.MSECriterion()
loss = loss 

Weights,Gradients = net:getParameters()
Train_Nett(net,Tr_Set,Tr_Set,60000)


-- Creating Architecture for classification

epoch = 20


net:remove(8)
net:remove(7)
net:remove(6)
net:remove(5)
net:add(nn.Linear(50, 10))
net:add(nn.Sigmoid())
net = net 
print(net)

print('Classifier Training...')
Weights,Gradients = net:getParameters()
xepo,yTr,ytest = Train_Net(net,Tr_Set,TrainLabels,60000,Test_Set,TestLabels)
gnuplot.plot({'Training Error',xepo,yTr},{'Validation Error',xepo,ytest})

