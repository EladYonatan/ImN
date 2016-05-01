
require 'cudnn'
require 'cunn'
require './BinarizedNeurons'
require './cudnnBinarySpatialConvolution'
require './BinaryLinear'
require './BatchNormalizationBC'
require './NormTransform'

local SpatialConvolution = cudnnBinarySpatialConvolution--lib[1]
local SpatialMaxPooling = cudnn.SpatialMaxPooling--lib[2
print('------------------')
print(opt.stcNeurons)
-- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
-- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
local features = nn.Sequential()
features:add(SpatialConvolution(3,256,11,11,4,4,2,2))       -- 224 -> 55
features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
--features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(256))
features:add(nn.HardTanh())
features:add(BinarizedNeurons(opt.stcNeurons))

features:add(SpatialConvolution(256,384,5,5,1,1,2,2))       --  27 -> 27
features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
--features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(384))
features:add(nn.HardTanh())
features:add(BinarizedNeurons(opt.stcNeurons))

--features:add(nn.SpatialBatchNormalization(192))
features:add(SpatialConvolution(384,512,3,3,1,1,1,1))      --  13 ->  13
--features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(512))
features:add(nn.HardTanh())
features:add(BinarizedNeurons(opt.stcNeurons))

features:add(SpatialConvolution(512,256,3,3,1,1,1,1))      --  13 ->  13
--features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(256))
features:add(nn.HardTanh())
features:add(BinarizedNeurons(opt.stcNeurons))

features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
--features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(256))
features:add(nn.HardTanh())
features:add(BinarizedNeurons(opt.stcNeurons))



local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
--classifier:add(nn.Dropout(0.5))
classifier:add(BinaryLinear(256*6*6, 4096))
--classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(4096))
classifier:add(nn.HardTanh())
classifier:add(BinarizedNeurons(opt.stcNeurons))
--classifier:add(nn.Dropout(0.5))
classifier:add(BinaryLinear(4096, 4096))
--classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(4096))
classifier:add(nn.HardTanh())
classifier:add(BinarizedNeurons(opt.stcNeurons))
classifier:add(BinaryLinear(4096, 1000))
--classifier:add(nn.Linear(4096, 1000))
--classifier:add(nn.ReLU())
--classifier:add(nn.BatchNormalization(1000))
classifier:add(BatchNormalizationBC(1000,nil,nil,false))
--classifier:add(nn.CMul(1000))
--classifier:add(NormTransform(0.001))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()

function fillBias(m)
for i=1, #m.modules do
    if m:get(i).bias then
        m:get(i).bias:fill(0.1)
    end
end
end

--fillBias(features)
--fillBias(classifier)
model:add(features):add(classifier)

local dE, param = model:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)
local clipvector = torch.Tensor(weight_size):fill(0)
local counter = 0
for j, layer_out in ipairs(model.modules) do
  for i, layer in ipairs(layer_out.modules) do
   if layer.__typename == 'BinaryLinear' then
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
      print(GLR)
      --GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
   elseif layer.__typename == 'nn.Linear' then
         local weight_size = layer.weight:size(1)*layer.weight:size(2)
         local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
         print(GLR)
         --GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
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
    --elseif layer.__typename == 'BatchNormalizationBC' then
    --    local weight_size = layer.weight:size(1)
    --    learningRates[{{counter+1, counter+weight_size}}]:fill(1)
    --    clipvector[{{counter+1, counter+weight_size}}]:fill(0)
    --    counter = counter+weight_size
    --    local bias_size = layer.bias:size(1)
    --    learningRates[{{counter+1, counter+bias_size}}]:fill(1)
    --    clipvector[{{counter+1, counter+bias_size}}]:fill(0)
    --    counter = counter+bias_size
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
      print(GLR)
    --  GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      clipvector[{{counter+1, counter+bias_size}}]:fill(0)
      counter = counter+bias_size
   end
  end
end
learningRates:fill(1)
--clipvector:fill(1)
print(param:size())
print(learningRates:eq(0):sum())
print(learningRates:ne(0):sum())
print(clipvector:ne(0):sum())
print(counter)
return {
     model = model,
     lrs = learningRates,
     clipV =clipvector,
  }
