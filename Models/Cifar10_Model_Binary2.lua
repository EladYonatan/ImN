
require 'cunn'
require 'cudnn'
require './cudnnBinarySpatialConvolution'
require './BinaryLinear'
require './BinarizedNeurons'
require 'nnx'
local model = nn.Sequential()

-- Convolution Layers

model:add(cudnnBinarySpatialConvolution(3, 256, 5, 5 ))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.SpatialBatchNormalization(256))
model:add(nn.HardTanh())
model:add(BinarizedNeurons(opt.stcNeurons))

model:add(cudnnBinarySpatialConvolution(256, 512, 3, 3))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.SpatialBatchNormalization(512))
model:add(nn.HardTanh())
model:add(BinarizedNeurons(opt.stcNeurons))

model:add(cudnnBinarySpatialConvolution(512, 512, 3, 3))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.SpatialBatchNormalization(512))
model:add(nn.HardTanh())
model:add(BinarizedNeurons(opt.stcNeurons))


model:add(cudnnBinarySpatialConvolution(512, 512, 2,2))
model:add(nn.SpatialBatchNormalization(512))
model:add(nn.HardTanh())
model:add(BinarizedNeurons(opt.stcNeurons))
model:add(nn.View(512))

model:add(BinaryLinear(512,10))
model:add(nn.BatchNormalization(10))
--model:add(nn.LogSoftMax())
local dE, param = model:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)
local clipvector = torch.Tensor(weight_size):fill(1)
local counter = 0
for i, layer in ipairs(model.modules) do
   if layer.__typename == 'BinaryLinear' then
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
      GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
    elseif layer.__typename == 'nn.BatchNormalization' then
      local weight_size = layer.weight:size(1)
      learningRates[{{counter+1, counter+weight_size}}]:fill(1)
      clipvector[{{counter+1, counter+weight_size}}]:fill(0)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(1)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
    elseif layer.__typename == 'nn.SpatialBatchNormalization' then
        local weight_size = layer.weight:size(1)
        local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]))
        learningRates[{{counter+1, counter+weight_size}}]:fill(1)
        clipvector[{{counter+1, counter+weight_size}}]:fill(0)
        counter = counter+weight_size
        local bias_size = layer.bias:size(1)
        learningRates[{{counter+1, counter+bias_size}}]:fill(1)
        clipvector[{{counter+1, counter+bias_size}}]:fill(0)
        counter = counter+bias_size
    elseif layer.__typename == 'cudnnBinarySpatialConvolution' then
      local size_w=layer.weight:size();
      local weight_size = size_w[1]*size_w[2]*size_w[3]*size_w[4]

      local filter_size=size_w[3]*size_w[4]
      GLR=1/torch.sqrt(1.5/(size_w[1]*filter_size+size_w[2]*filter_size))
      GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size

  end
end
-- clip all parameter
--clipvector:fill(1)
--
print(learningRates:eq(0):sum())
print(learningRates:ne(0):sum())
print(clipvector:ne(0):sum())
print(counter)
return {
     model = model,
     lrs = learningRates,
     clipV =clipvector,
  }
