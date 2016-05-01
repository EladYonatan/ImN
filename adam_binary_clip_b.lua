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
    local GLRvec = config.GLRvec or 1
    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    --print(lr,dfdx:size())
    -- Initialization
    --print('OK')
    state.dfdx1 = state.dfdx1 or  x.new(dfdx:size()):zero()
    state.t = state.t or 0
    state.tmp = state.tmp or torch.zeros(x:size()):cuda()
    -- Exponential moving average of gradient values
    state.m = state.m or x.new(dfdx:size()):zero()
    -- Exponential moving average of squared gradient values
    state.v = state.v or x.new(dfdx:size()):zero()
    -- A tmp tensor to hold the sqrt(v) + epsilon
    state.denom = state.denom or x.new(dfdx:size()):zero()

    state.t = state.t + 1

    state.clipVnot = state.clipVnot or torch.zeros(clipV:size()):cuda():fill(1):add(-1,clipV)
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

    state.tmp:fill(0)
    state.tmp:addcdiv(1, state.m, state.denom)
    x:addcmul(-stepSize, state.tmp, GLRvec)
    --x=x:clamp(-1,1)
    --local clipVnot=torch.zeros(clipV:size()):cuda():fill(1):add(-1,clipV)
    state.x1:copy(x):cmul(clipV):clamp(-1,1)
    x:cmul(state.clipVnot):add(state.x1)
    --x[clipV:eq(1)]=x[clipV:eq(1)]:clamp(-1,1)

    --print(x:min(),x:mean())
    -- return x*, f(x) before optimization
    return x, {fx}
end
