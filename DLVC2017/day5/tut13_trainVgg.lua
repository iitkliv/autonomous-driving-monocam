require 'nn'
--require 'cunn'
--require 'cudnn'
--require 'nngraph'
require 'gnuplot'
require 'image'
require 'loadcaffe'
require 'optim'

batchSize = 5
torch.manualSeed(1) 
-- create data loader
dataNames={}
dataLabels={}
dataIter=0
for i=1,10 do
  filename = 'Filenames/'..i..'.txt'
  print(filename)
  local file = io.open(filename)
  for l in io.lines(filename) do
    temp = image.load(l)
    if temp:size(1)==3 then
      dataIter=dataIter+1
      dataNames[dataIter] = l
      dataLabels[dataIter]=i
    end
  end
end

------- Load pretrained VGGNet
model = loadcaffe.load('VGG.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn')
model:remove(46)
model:remove(45)

model:add(nn.Linear(4096,10))
model:add(nn.LogSoftMax())

--------- Loss function
loss = nn.ClassNLLCriterion()

-- model = cudnn.convert(model,cudnn)
-- model:cuda() 
-- loss:cuda()
model:training()
 
it = 0
ep = 0 
pl ={}     

function feval(weights)
   gradients:zero()
   local loss_val = 0
   local inputs = inputs:double()
   local labels = labels:double()
	
   --local inputs = inputs:cuda()
   --local labels = labels:cuda()
   local y = model:forward(inputs)
   loss_val = loss:forward(y, labels)
   local df_dw = loss:backward(y, labels)
   model:backward(inputs, df_dw)
   it = it+1
   
   pl[it] = loss_val
   gnuplot.plot(torch.Tensor(pl))

   --print ('Running epoch ' .. ep .. ' loss value = ' .. loss_val .. 'at batch number ' .. it .. ' with batch size of ' .. opt.batchSize)
   return loss_val, gradients
end

sgdState = {
		learningRate = 1e-2,
		}

weights, gradients = model:getParameters()

for epoch = 1,30 do
  totalErr = 0
  idx = torch.randperm(dataIter)
  for i = 1, torch.floor(dataIter/batchSize) do
    inputs = torch.Tensor(batchSize,3,224,224);
    labels = torch.Tensor(batchSize);
    for j=1,batchSize do
      inputs[j] = image.scale(image.load(dataNames[idx[(i-1)*batchSize + j]]),224,224);
      labels[j] = dataLabels[idx[(i-1)*batchSize + j]]
    end
    _, err = optim.sgd(feval, weights, sgdState)
    totalErr = err[1] + totalErr
  end

--gnuplot.plot(torch.Tensor(pl))

end
--]]
