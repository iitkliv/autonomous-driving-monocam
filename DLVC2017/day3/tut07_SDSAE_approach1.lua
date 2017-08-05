require 'torch';
require 'optim';
require 'nn';
require 'gnuplot';
require 'image';


LR = 0.1
epoch = 5
--LRDecay =0
--weightDecay=1e-4
--momentum=0.9

SNRdb = 2

DataPath = ''
torch.setdefaulttensortype('torch.FloatTensor')

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

Tr_SetEnergy = torch.mean(torch.pow(Tr_Set,2),2) 
SNR = 10^(SNRdb/10)
Noisemul = torch.sqrt(Tr_SetEnergy/SNR)
Noisemul = Noisemul:storage()
Noise = torch.FloatTensor(60000,784)
for i=1,60000 do
  Noise[i] = torch.rand(784)*Noisemul[i]
end
Tr_Set_Noisy = torch.add(Tr_Set,Noise)

Test_SetEnergy = torch.mean(torch.pow(Test_Set,2),2) 
SNR = 10^(SNRdb/10)
Noisemul = torch.sqrt(Test_SetEnergy/SNR)
Noisemul = Noisemul:storage()
Noise = torch.FloatTensor(10000,784)
for i=1,10000 do
  Noise[i] = torch.rand(784)*Noisemul[i]
end
Test_Set_Noisy = torch.add(Test_Set,Noise)


print('Dataset Loaded and Ready..')

---------------------------------
---------------------------------


function network(Data,Net,Out)

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




function Train_Net(Net,Data_Set,Data_out,sz)

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

function Train_Nett(Net,Data,sz)

  for Tepoch = 1,epoch do
    print('epoch ' .. Tepoch .. '/' .. epoch)
    Total_Loss=0
    for loopno=1,sz do
      input = Data[loopno]
      input = input[1]
      local LossTrain = network(input,Net,input)    
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

print('Training...')


local net

net = nn.Sequential()
net:add(nn.Linear(784, 500)) 
net:add(nn.ReLU())
net:add(nn.L1Penalty(1e-5))
net:add(nn.Linear(500, 784)) 
net:add(nn.ReLU()) 
net:add(nn.L1Penalty(1e-5))


loss = nn.MSECriterion()



-- STEP-1
print('At Step-1 ................................')
Weights,Gradients = net:getParameters()
Train_Net(net,Tr_Set_Noisy,Tr_Set,60000)

net:remove(6)
net:remove(5)
net:remove(4)
idata = {}
for loopno=1,60000 do
  input = Tr_Set_Noisy[loopno]
  input = input 
  out = net:forward(input)
  idata[loopno] = {out}
end



-- STEP-2
print('At Step-2 ................................')
local net1
net1 = nn.Sequential()
net1:add(nn.Linear(500, 400)) 
net1:add(nn.ReLU())  
net1:add(nn.L1Penalty(1e-5))
net1:add(nn.Linear(400, 500))  
net1:add(nn.ReLU())
net1:add(nn.L1Penalty(1e-5))
Weights,Gradients = net1:getParameters()
Train_Nett(net1,idata,60000)

net1:remove(6)
net1:remove(5)
net1:remove(4)
idata1 = {}
for loopno=1,60000 do
  input = idata[loopno]
  input = input[1]
  input = input 
  out = net1:forward(input)
  idata1[loopno] = {out}
end


-- STEP-3
print('At Step-3 ................................')
local net2
net2 = nn.Sequential()
net2:add(nn.Linear(400, 300)) 
net2:add(nn.ReLU())  
net2:add(nn.L1Penalty(1e-5))
net2:add(nn.Linear(300, 400))  
net2:add(nn.ReLU()) 
net2:add(nn.L1Penalty(1e-5)) 
Weights,Gradients = net2:getParameters()
Train_Nett(net2,idata1,60000)


--STEP-4
print('At Step-4 ................................')
net1:add(net2)
net1:add(nn.Linear(400, 500)) 
net1:add(nn.ReLU()) 
net1:add(nn.L1Penalty(1e-5))
Weights,Gradients = net1:getParameters()
Train_Nett(net1,idata,60000)




--STEP-5
print('At Step-5 ................................')
net:add(net1)
net:add(nn.Linear(500, 784)) 
net:add(nn.ReLU()) 
net:add(nn.L1Penalty(1e-5)) 
Weights,Gradients = net:getParameters()
Train_Net(net,Tr_Set_Noisy,Tr_Set,60000)


------------------------------------------------------------------
------------------------------------------------------------------
--      Validation    --------------------------------------------
------------------------------------------------------------------
------------------------------------------------------------------



Total_Loss = 0
for loopno=1,10000 do
  input = Test_Set_Noisy[loopno]
  output = Test_Set[loopno]
  out = net:forward(input)
  currLoss = loss:forward(out,output)
  Total_Loss = Total_Loss + currLoss
end

ValLoss = Total_Loss/10000
print('Total Testing/Validation Loss : '.. ValLoss)

input = Test_Set_Noisy[1]
inputimg = torch.reshape(input,28,28)
out = net:forward(input)
outputimg = torch.reshape(out,28,28)
output = Test_Set[1]
Noiselessimg = torch.reshape(output,28,28)

image.save('NoisyImage.png', inputimg)
image.save('DenoisedImage.png', outputimg)
image.save('ActualImage.png', Noiselessimg)

print(net)

