require 'nn'
require 'gnuplot'


-- Random initialization of training, validation and testing data
train_data=torch.rand(10,1024)
train_label=torch.Tensor(10):fill(1)

val_data=torch.rand(10,1024)
val_label=torch.Tensor(10):fill(1)

test_data=torch.rand(10,1024)
test_label=torch.Tensor(10):fill(1)


model = require 'myNet.lua'
--model = cudnn.convert(model, cudnn)
--model:cuda()
criterion = nn.ClassNLLCriterion()
--criterion = criterion:cuda()


LR = 0.01 --Learning rate
epochs = 5  --Number of epochs

-----------Train model----------------------------
data_sz = train_data:size(1)

local train_err = torch.zeros(1,epochs)
local val_err = torch.zeros(1,epochs)

for n = 1,epochs do	
	print('Epoch ' .. n)
	err = 0
	for i = 1,data_sz do	
	  -- feed data to the neural network and the criterion 
	  data = train_data[{{i},{}}]	  
	  target = train_label[i]  
	  output = model:forward(data)
	  criterion:forward(output,target)	  
	  -- train over this example in 3 steps
	  -- (1) zero the accumulation of the gradients
	  model:zeroGradParameters()
	  -- (2) accumulate gradients
    
	  model:backward(data, criterion:backward(output, target))
	  -- (3) update parameters with a 0.01 learning rate
	  model:updateParameters(LR)	  
	  err_train = criterion:forward(output,target)	 
	  err = err + err_train
	 -- print(err)
	end	
	train_err[{{1},{n}}] = err/data_sz
  
	----------------Finding validation error----------------
	val_output = model:forward(val_data)	
    val_err[{{1},{n}}] = criterion:forward(val_output,val_label)
end
------------Plot performance------------------------------------------
gnuplot.pngfigure('ErrorvsEpochs.png')

gnuplot.figure()
gnuplot.plot({'Training error',train_err[1]},{'Validation error',val_err[1]})
gnuplot.xlabel('Epoch')
gnuplot.ylabel('Training error')
gnuplot.grid(true)
gnuplot.title('Plot of error vs. epochs')
gnuplot.plotflush()
-------------Test model------------------------------------------------
sample_test = test_data[{{5,10},{},{},{}}]
sample_label = test_label[{{5,10}}]
pred = torch.exp(model:forward(sample_test))
pred_val,pred_class = torch.max(pred,1)
-------------Printing results---------------------------------------
print('Predicted class...')
print(pred_class) -- predicted class
print('Ground truth label...')
print(sample_label) -- goundtruth class
