require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require './BinarizedNeurons'
require './cudnnBinarySpatialConvolution'
require './BinaryLinear'
require 'nnx'
local DimConcat = 2

local SpatialConvolution = cudnnBinarySpatialConvolution
local SpatialMaxPooling = cudnn.SpatialMaxPooling
local SpatialAveragePooling = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU

local BNInception = true

---------------------------------------Inception Modules-------------------------------------------------
local Inception = function(nInput, n1x1, n3x3r, n3x3, dn3x3r, dn3x3, nPoolProj, type_pool,stride)
    local stride = stride or 1
    local InceptionModule = nn.Concat(DimConcat)

    if n1x1>0 then
        InceptionModule:add(nn.Sequential():add(SpatialConvolution(nInput,n1x1,1,1,stride,stride)))
        --InceptionModule:add(nn.SpatialBatchNormalization(n1x1,nil,nil,false))
        --InceptionModule:add(nn.HardTanh())
        --InceptionModule:add(BinarizedNeurons(opt.stcNeurons))
    end

    if n3x3>0 and n3x3r>0 then
        local Module_3x3 = nn.Sequential()
        Module_3x3:add(SpatialConvolution(nInput,n3x3r,1,1)) --:add(ReLU(true))

        Module_3x3:add(nn.SpatialBatchNormalization(n3x3r))
        Module_3x3:add(nn.HardTanh())
        Module_3x3:add(BinarizedNeurons(opt.stcNeurons))
        --if BNInception then
        --    Module_3x3:add(nn.SpatialBatchNormalization(n3x3r,nil,nil,false))
        --end
        Module_3x3:add(SpatialConvolution(n3x3r,n3x3,3,3,stride,stride,1,1))
        --Module_3x3:add(nn.SpatialBatchNormalization(n3x3,nil,nil,false))
        --Module_3x3:add(nn.HardTanh())
        --Module_3x3:add(BinarizedNeurons(opt.stcNeurons))
        InceptionModule:add(Module_3x3)
    end

    if dn3x3>0 and dn3x3r>0 then
        local Module_d3x3 = nn.Sequential()
        Module_d3x3:add(SpatialConvolution(nInput,dn3x3r,1,1)) --:add(ReLU(true))

        Module_d3x3:add(nn.SpatialBatchNormalization(dn3x3r))
        Module_d3x3:add(nn.HardTanh())
        Module_d3x3:add(BinarizedNeurons(opt.stcNeurons))

        --if BNInception then
        --    Module_d3x3:add(nn.SpatialBatchNormalization(dn3x3r,nil,nil,false))
        --end

        Module_d3x3:add(SpatialConvolution(dn3x3r,dn3x3r,3,3,1,1,1,1)) --:add(ReLU(true))

        Module_d3x3:add(nn.SpatialBatchNormalization(dn3x3r))
        Module_d3x3:add(nn.HardTanh())
        Module_d3x3:add(BinarizedNeurons(opt.stcNeurons))

        --if BNInception then
        --    Module_d3x3:add(nn.SpatialBatchNormalization(dn3x3r,nil,nil,false))
        --end

        Module_d3x3:add(SpatialConvolution(dn3x3r,dn3x3,3,3,stride,stride,1,1))
        InceptionModule:add(Module_d3x3)
    end

    local PoolProj = nn.Sequential()
    if type_pool == 'avg' then
        PoolProj:add(SpatialMaxPooling(3,3,stride,stride,1,1))
    elseif type_pool == 'max' then
        PoolProj:add(SpatialMaxPooling(3,3,stride,stride,1,1))
    end
    if nPoolProj > 0 then
        PoolProj:add(SpatialConvolution(nInput, nPoolProj, 1, 1))
    end


    InceptionModule:add(PoolProj)
    return InceptionModule
end

-----------------------------------------------------------------------------------------------------------


local part1 = nn.Sequential()
part1:add(SpatialConvolution(3,64,7,7,2,2,3,3)) --3x224x224 -> 64x112x112
part1:add(SpatialMaxPooling(3,3,2,2):ceil()) -- 64x112x112 -> 64x56x56

part1:add(nn.SpatialBatchNormalization(64))
--part1:add(ReLU(true))
part1:add(nn.HardTanh())
part1:add(BinarizedNeurons(opt.stcNeurons))

part1:add(SpatialConvolution(64,192,3,3,1,1,1,1)) -- 64x56x56 -> 192x56x56
part1:add(SpatialMaxPooling(3,3,2,2):ceil()) -- 192x56x56 -> 192x28x28

part1:add(nn.SpatialBatchNormalization(192))
--part1:add(ReLU(true))
part1:add(nn.HardTanh())
part1:add(BinarizedNeurons(opt.stcNeurons))


--Inception(nInput, n1x1, n3x3r, n3x3, dn3x3r, dn3x3, nPoolProj, type_pool=['avg','max',nil])
part1:add(Inception(192,64,64,64,64,96,32,'avg')) --(3a) 192x28x28 -> 256x28x28

part1:add(nn.SpatialBatchNormalization(256))
--part1:add(ReLU(true))
part1:add(nn.HardTanh())
part1:add(BinarizedNeurons(opt.stcNeurons))

part1:add(Inception(256,64,64,96,64,96,64,'avg'))  --(3b) 256x28x28 -> 320x28x28

part1:add(nn.SpatialBatchNormalization(320))
--part1:add(ReLU(true))
part1:add(nn.HardTanh())
part1:add(BinarizedNeurons(opt.stcNeurons))

part1:add(Inception(320,0,128,160,64,96,0,'max',2)) --(3c) 320x28x28 -> 576x14x14

part1:add(nn.SpatialBatchNormalization(576))
--part1:add(ReLU(true))
part1:add(nn.HardTanh())
part1:add(BinarizedNeurons(opt.stcNeurons))


local part2 = nn.Sequential()
part2:add(Inception(576,224,64,96,96,128,128,'avg'))  --(4a) 576x14x14 -> 576x14x14

part2:add(nn.SpatialBatchNormalization(576))
--part2:add(ReLU(true))
part2:add(nn.HardTanh())
part2:add(BinarizedNeurons(opt.stcNeurons))

part2:add(Inception(576,192,96,128,96,128,128,'avg'))  --(4b) 576x14x14 -> 576x14x14

part2:add(nn.SpatialBatchNormalization(576))
part2:add(nn.HardTanh())
part2:add(BinarizedNeurons(opt.stcNeurons))
--part2:add(ReLU(true))
part2:add(Inception(576,160,128,160,128,160,96,'avg'))  --(4c) 576x14x14 -> 576x14x14

part2:add(nn.SpatialBatchNormalization(576))
--part2:add(ReLU(true))
part2:add(nn.HardTanh())
part2:add(BinarizedNeurons(opt.stcNeurons))

local part3 = nn.Sequential()
part3:add(Inception(576,96,128,192,160,192,96,'avg'))  --(4d) 576x14x14 -> 576x14x14

part3:add(nn.SpatialBatchNormalization(576))
--part3:add(ReLU(true))
part3:add(nn.HardTanh())
part3:add(BinarizedNeurons(opt.stcNeurons))

part3:add(Inception(576,0,128,192,192,256,0,'max',2))  --(4e) 576x14x14 -> 1024x7x7

part3:add(nn.SpatialBatchNormalization(1024))
--part3:add(ReLU(true))
part3:add(nn.HardTanh())
part3:add(BinarizedNeurons(opt.stcNeurons))

part3:add(Inception(1024,352,192,320,160,224,128,'avg'))  --(5a) 1024x7x7 -> 1024x7x7

part3:add(nn.SpatialBatchNormalization(1024))
--part3:add(ReLU(true))
part3:add(nn.HardTanh())
part3:add(BinarizedNeurons(opt.stcNeurons))
part3:add(Inception(1024,352,192,320,192,224,128,'max'))  --(5b) 1024x7x7 -> 1024x7x7

part3:add(nn.SpatialBatchNormalization(1024))
--part3:add(ReLU(true))
part3:add(nn.HardTanh())
part3:add(BinarizedNeurons(opt.stcNeurons))

--Classifier
local mainClassifier = nn.Sequential()
--mainClassifier:add(nn.Dropout(0.5))

mainClassifier:add(SpatialMaxPooling(7,7))
mainClassifier:add(nn.View(1024):setNumInputDims(3))
---

mainClassifier:add(nn.BatchNormalization(1024))

mainClassifier:add(nn.HardTanh())
mainClassifier:add(BinarizedNeurons(opt.stcNeurons))
---
mainClassifier:add(BinaryLinear(1024,1000))
mainClassifier:add(nn.LogSoftMax())


local auxClassifier1 = nn.Sequential()

auxClassifier1:add(SpatialMaxPooling(5,5,3,3):ceil())
auxClassifier1:add(SpatialConvolution(576,128,1,1))
--auxClassifier1:add(cudnn.ReLU(true))

auxClassifier1:add(nn.SpatialBatchNormalization(128))
auxClassifier1:add(nn.HardTanh())
auxClassifier1:add(BinarizedNeurons(opt.stcNeurons))
--
auxClassifier1:add(nn.View(128*4*4):setNumInputDims(3))
auxClassifier1:add(BinaryLinear(128*4*4,768))
--auxClassifier1:add(cudnn.ReLU(true))

auxClassifier1:add(nn.BatchNormalization(768))
auxClassifier1:add(nn.HardTanh())
auxClassifier1:add(BinarizedNeurons(opt.stcNeurons))
--
--auxClassifier1:add(nn.Dropout(0.5))
auxClassifier1:add(BinaryLinear(768,1000))
auxClassifier1:add(nn.LogSoftMax())

local auxClassifier2 = auxClassifier1:clone()

local input = nn.Identity()()
local output1 = part1(input)
local branch1 = auxClassifier1(output1)
local output2 = part2(output1)
local branch2 = auxClassifier2(output2)
local mainBranch = mainClassifier(part3(output2))
local model = nn.gModule({input},{mainBranch,branch1,branch2})


local NLL = nn.ClassNLLCriterion()
local loss = nn.ParallelCriterion(true):add(NLL):add(NLL,0.3):add(NLL,0.3)


local dE, param = model:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)
local clipvector = torch.Tensor(weight_size):fill(0)
local counter = 0
for j, layer_out in ipairs(model.modules) do
  if j>1 then
  for i, layer in ipairs(layer_out.modules) do
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
end
end
learningRates:fill(1)
clipvector:fill(1)
print(learningRates:eq(0):sum())
print(learningRates:ne(0):sum())
print(clipvector:ne(0):sum())
print(counter)
return {
     model = model,
     loss = loss,
     lrs = learningRates,
     clipV =clipvector,
  }

--return {model = model, loss = loss}
