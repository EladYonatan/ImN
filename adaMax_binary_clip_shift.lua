--[[ An implementation of Shift based AdaMax based on  http://arxiv.org/pdf/1412.6980.pdf as described the paper:
   "Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio'

Note that this function perform the weight cliping as well

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

function adaMax_binary_clip_shift(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.002
    local GLRvec = config.GLRvec or 1
    local clipV = config.clipV or 0

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 2^-27

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    -- Initialization
    state.t = state.t or 0
    -- Exponential moving average of gradient values
    state.m = state.m or x.new(dfdx:size()):zero()
    -- Exponential moving average of squared gradient values
    state.v = state.v or x.new(dfdx:size()):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.denom = state.denom or x.new(dfdx:size()):zero()

    state.t = state.t + 1

    -- Decay the first and second moment running average coefficient
    state.m:mul(beta1):add(1-beta1, dfdx)
    state.v:copy( torch.cmax(state.v:mul(beta2),dfdx:abs()) )

    state.tmp = state.tmp or torch.zeros(x:size()):cuda()
    state.clipVnot = state.clipVnot or torch.zeros(clipV:size()):cuda():fill(1):add(-1,clipV)
    state.x1 = state.x1 or  x.new(dfdx:size()):zero()

    local biasCorrection1 = 1 - beta1^state.t

    local stepSize = lr/biasCorrection1 --math.sqrt(biasCorrection2)/biasCorrection1

    stepSize=math.pow(2,torch.round(math.log(stepSize)/(math.log(2))))
    -- (2) update x
    --local tmp=torch.zeros(x:size())
    --if opt.type == 'cuda' then
    --  tmp=tmp:cuda()
    --end
    print(model)

    --state.v:copy(torch.pow(2,torch.round(torch.log(state.v):div(math.log(2)))))
    state.v:add(epsilon)
    state.tmp:fill(0)
    state.tmp:addcdiv(1, state.m, state.v)
    -- Multiply by Glorot learning rate vector
    x:addcmul(-stepSize, state.tmp, GLRvec)
    -- Clip to [-1,1]
    state.x1:copy(x):cmul(clipV):clamp(-1,1)
    x=x:cmul(state.clipVnot):add(state.x1)
    --x[clipV:eq(1)]=x[clipV:eq(1)]:clamp(-1,1)
    -- return x*, f(x) before optimization
    return x, {fx}
end
