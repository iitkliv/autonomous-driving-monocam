require 'torch';
require 'optim';
require 'nn';
--require 'cudnn';
--require 'cunn';
require 'gnuplot';
require 'image';


torch.manualSeed(0) 
epoch = 5
Train_Error = torch.ones(epoch,3)
Valid_Error = torch.ones(epoch,3)




local optimState = {learningRate = 1e-4,}


---------------------------------
---------------------------------



trainset = torch.DiskFile('zip/train-images.idx3-ubyte')
testset = torch.DiskFile('zip/t10k-images.idx3-ubyte')

Tr_Set = torch.ByteTensor(60000, 784)
Test_Set = torch.ByteTensor(10000, 784)

trainset:readByte(16)
trainset:readByte(Tr_Set:storage())
--Tr_Set = Tr_Set:double():div(255) 
Tr_Set = Tr_Set[{{1,100},{}}]:double():div(255) 
testset:readByte(16)
testset:readByte(Test_Set:storage())
Test_Set = Test_Set:double():div(255) 


print('Dataset Loaded and Ready..')

---------------------------------
---------------------------------


function network(Data,Net,Out)
  Data = Data
  Out = Out
  function feval(Weights)
    Gradients:zero()
    y = Net:forward(Data)
    currLoss =  loss:forward(y,Out)
    local dE_dy = loss:backward(y, Out)
    Net:backward(Data, dE_dy)
    return currLoss, Gradients
 end
 optim.sgd(feval, Weights, optimState)
 return currLoss
end




function Train_Net(Net,Data_Set,Data_out)

  for Tepoch = 1,epoch do
    print('epoch ' .. Tepoch .. '/' .. epoch)
    Total_Loss=0
    for loopno=1,Data_out:size(1) do
      input = Data_Set[loopno]
      output = Data_out[loopno]
      local LossTrain = network(input,Net,output)    
      Total_Loss = Total_Loss + LossTrain
      loopno = loopno + 1
    end
    Total_Loss = Total_Loss/Data_out:size(1)
    print('Training Loss = ' .. Total_Loss)
    Train_Error[{{Tepoch},{1}}] = Total_Loss
    Val_Loss = 0
    for loopno=1,10000 do
        input = Test_Set[loopno]
        input = input
        currLoss = nn.MSECriterion():forward(Net:forward(input),input)
        Val_Loss = Val_Loss + currLoss
     end
    ValLoss = Val_Loss/10000
    print('Total Testing/Validation Loss : '.. ValLoss)
    Valid_Error[{{Tepoch},{1}}] = Total_Loss
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
net:add(nn.Linear(500, 784)) 
net:add(nn.ReLU()) 
--net:add(nn.L1Penalty(1e-5))
--net = net
loss = nn.AbsCriterion()
--loss = loss


-- STEP-1
print('At Ladder-1 with ABS loss................................')
Weights,Gradients = net:getParameters()
Train_Net(net,Tr_Set,Tr_Set)


--[[
------------------------------------------------------------------
------------------------------------------------------------------
--      Validation    --------------------------------------------
------------------------------------------------------------------
------------------------------------------------------------------

Total_Loss = 0
for loopno=1,10000 do
  input = Test_Set[loopno]
  input = input
  currLoss = nn.MSECriterion():forward(net:forward(input),input)
  Total_Loss = Total_Loss + currLoss
end

ValLoss = Total_Loss/10000
print('Total Testing/Validation Loss : '.. ValLoss)
]]--
input = Test_Set[1]
inputimg = torch.reshape(input,28,28)
input = input
out = net:forward(input)
outputimg = torch.reshape(out,28,28)

image.save('reconsImage_abs.png', outputimg)
image.save('ActualImage.png', inputimg)

--print(net)

