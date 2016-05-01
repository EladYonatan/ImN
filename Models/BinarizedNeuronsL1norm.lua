local BinarizedNeurons,parent = torch.class('BinarizedNeurons', 'nn.Module')


function BinarizedNeurons:__init(stcFlag)
   parent.__init(self)
   self.stcFlag = stcFlag
   self.randmat=torch.Tensor();
   self.outputR=torch.Tensor();
   self.convmodel=nn.Sequential()
   self.convmodel:add(nn.SpatialConvolution(1,1,3,3,1,1,1,1))
   self.convmodel.modules[1].weight:fill(1/9)
   self.convmodel.modules[1].bias:fill(0)
  --print(self.convmodel.modules[1].weight)
 end
function BinarizedNeurons:updateOutput(input)
    self.randmat:resizeAs(input);
    self.outputR:resizeAs(input);
    self.output:resizeAs(input);
    self.output:copy(input)

    if input:size():size()>2 then
      --print(input:size())
       l1c=torch.norm(input,1,2)
       --print(l1c:size())
       l1 = self.convmodel:forward(l1c);
       --print(l1:size())
    else
       l1=torch.norm(input,1,2)
    --   l1:fill(1)
     end
    l1:div(input:size(2))
    l1=l1:expandAs(input)
   self.output=self.output:sign():cmul(l1)
   return self.output
end

function BinarizedNeurons:updateGradInput(input, gradOutput)
        self.gradInput:resizeAs(gradOutput)
        self.gradInput:copy(gradOutput) --:mul(0.5)
        self.gradInput[input:ge(1)]=0
        self.gradInput[input:le(-1)]=0
   return self.gradInput
end
