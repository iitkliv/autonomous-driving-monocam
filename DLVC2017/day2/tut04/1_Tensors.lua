--ref: https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb

require 'torch'
print('Ex 1')
a = 'hello'
print(a)

print('-------------------------------------------------')

print('Ex 2')
b = {}
b[1] = a
print(b)

print('-------------------------------------------------')

print('Ex 3')
b[2] = 30
for i=1,#b do -- the # operator is the length operator in Lua
    print(b[i]) 
end

print('-------------------------------------------------')

print('Ex 4')
a = torch.Tensor(5,3) -- construct a 5x3 matrix, uninitialized
a = torch.rand(5,3)
print(a)

print('-------------------------------------------------')

print('Ex 5')
b=torch.rand(3,4)
-- matrix-matrix multiplication: syntax 1
c = a*b
print(c)

-- matrix-matrix multiplication: syntax 2
torch.mm(a,b)

-- matrix-matrix multiplication: syntax 3
c=torch.Tensor(5,4)
c:mm(a,b) -- store the result of a*b in c

function addTensors(a,b)
    return a -- FIX ME
end

print('-------------------------------------------------')

print('Ex 6')
a = torch.ones(5,2)
b = torch.Tensor(2,5):fill(4)
print(addTensors(a,b))



