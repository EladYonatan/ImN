local NormTransform, parent = torch.class('NormTransform', 'nn.Module')

function NormTransform:__init(constant_scalar,ip)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar

  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function NormTransform:updateOutput(input)
--  print(input:norm(1),input:norm(2),input:max(),input:min())
  --input:add(-input:min())
  self.constant_scalar=1/4096 --input:max();
  if self.inplace then
    input:mul(self.constant_scalar)
    self.output:set(input)
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    self.output:mul(self.constant_scalar)
    print(self.output:norm(1),self.output:norm(2),self.output:max(),self.output:min(),self.output:mean())
  end
  return self.output
end

function NormTransform:updateGradInput(input, gradOutput)
  if self.gradInput then
    if self.inplace then
      gradOutput:mul(self.constant_scalar)
      self.gradInput:set(gradOutput)
      -- restore previous input value
      input:div(self.constant_scalar)
    else
      self.gradInput:resizeAs(gradOutput)
      self.gradInput:copy(gradOutput)
      self.gradInput:mul(self.constant_scalar)
    end
    return self.gradInput
  end
end
