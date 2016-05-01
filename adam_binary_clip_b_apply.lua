--[[ An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.beta1'             : first moment coefficient
- 'config.beta2'             : second moment coefficient
- 'config.epsilon'           : for numerical stability
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function adam_binary_clip_b(opfunc, x, config, state)
    -- (0) get/update state

    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001
    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8
    local f=function(model)
      if torch.type(model)=='cudnnBinarySpatialConvolution' or torch.type(model)=='BinaryLinear' then
        model.weight:clamp(-1,1)
      end
    end
    local g=function(model)
      if  torch.type(model)=='BinaryLinear' then
        local weight_size = model.weight:size(1)*model.weight:size(2)
        local size_w=model.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
        model.weight:mul(GLR)
        model.bias:mul(GLR)
      elseif  torch.type(model)=='BinaryLinear' then
        local size_w=model.weight:size();
        local weight_size = size_w[1]*size_w[2]*size_w[3]*size_w[4]
        local filter_size=size_w[3]*size_w[4]
        GLR=1/torch.sqrt(1.5/(size_w[1]*filter_size+size_w[2]*filter_size))
        model.weight:mul(GLR)
        model.bias:mul(GLR)
      end
    end
    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    --print(lr,dfdx:size())
    -- Initialization
    --print('OK')
    state.dfdx1 = state.dfdx1 or  x.new(dfdx:size()):zero()
    state.t = state.t or 0
  --  state.tmp = state.tmp or torch.zeros(x:size()):cuda()
    -- Exponential moving average of gradient values
    state.m = state.m or x.new(dfdx:size()):zero()
    -- Exponential moving average of squared gradient values
    state.v = state.v or x.new(dfdx:size()):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.denom = state.denom or x.new(dfdx:size()):zero()

    state.t = state.t + 1

    --state.clipVnot = state.clipVnot or torch.zeros(clipV:size()):cuda():fill(1):add(-1,clipV)
    --print('---Norm--')
    --print(dfdx:norm())
    --dfdx:sign():mul(0.1)
    --local norm = state.dfdx1:copy(dfdx):cmul(state.clipVnot):norm()

    ----print(dfdx:max(),dfdx:min())
    --if norm > 10 then
    --  print('Grad Renorm')
    --  print(norm)
    --  local shrink = 10/ norm
    --  state.dfdx1:mul(shrink)
 -- --    dfdx:mul(shrink)
    --end
    ----state.dfdx1:copy(dfdx):cmul(clipV)
    ----print('------!!--------')
    ----print(state.dfdx1:max(),dfdx:min(),state.clipVnot:sum())
    --dfdx:cmul(clipV) --:add(state.dfdx1)
    ----print('------------:)-------')
    ----print(state.dfdx1:max(),state.dfdx1:min(),dfdx:max(),dfdx:min())
    --dfdx:add(state.dfdx1)


    state.x1 = state.x1 or  x.new(dfdx:size()):zero()
    -- Decay the first and second moment running average coefficient
    state.m:mul(beta1):add(1-beta1, dfdx)
    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

    state.denom:copy(state.v):sqrt():add(epsilon)

    local biasCorrection1 = 1 - beta1^state.t
    local biasCorrection2 = 1 - beta2^state.t
    local stepSize = lr * math.sqrt(biasCorrection2)/biasCorrection1
    -- (2) update x

--    state.tmp:fill(0)
    x:addcdiv(-stepSize, state.m, state.denom)
    --x=x:clamp(-1,1)
    --local clipVnot=torch.zeros(clipV:size()):cuda():fill(1):add(-1,clipV)
    --model:apply(g)
    model:apply(f)
    --state.x1:copy(x):cmul(clipV):clamp(-1,1)
    --x:cmul(state.clipVnot):add(state.x1)
    --x[clipV:eq(1)]=x[clipV:eq(1)]:clamp(-1,1)

    --print(x:min(),x:mean())
    -- return x*, f(x) before optimization
    return x, {fx}
end
