--[[ A plain implementation of SGD

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

(Clement Farabet, 2012)
]]
function sgd_clip_b(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   --local lrs = GLRvec
   local lrs = config.GLRvec or 1
   local clipV = config.clipV or 0
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")


   state.clipVnot = state.clipVnot or torch.zeros(clipV:size()):cuda():fill(1):add(-1,clipV)
   state.tmp = state.tmp or torch.zeros(x:size()):cuda()

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)
   --print(dfdx:max(),dfdx:min(),dfdx:mean())
   state.x1 = state.x1 or  x.new(dfdx:size()):zero()
   state.dfdx1 = state.dfdx1 or  x.new(dfdx:size()):zero()
   -- (2) weight decay with single or individual parameters
   if wd ~= 0 then
      dfdx:add(wd, x)
   elseif wds then
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   --local norm = state.dfdx1:copy(dfdx):cmul(state.clipVnot):norm()
   --print(dfdx:max(),dfdx:min())
   --if norm > 10 then
  --   print('Grad Renorm')
  --   print(norm)
  --   local shrink = 10/ norm
  --   state.dfdx1:mul(shrink)
-- --    dfdx:mul(shrink)
   --end
   --state.dfdx1:copy(dfdx):cmul(clipV)
   --print('------!!--------')
   --print(state.dfdx1:max(),dfdx:min(),state.clipVnot:sum())
   --dfdx:cmul(clipV) --:add(state.dfdx1)
   --print('------------:)-------')
   --print(state.dfdx1:max(),state.dfdx1:min(),dfdx:max(),dfdx:min())
   --dfdx:add(state.dfdx1)
   -- (5) parameter update with single or individual learning rates
   --print(lrs:max(),lrs:min())
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      --print(clr,lrs:max())
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
    --  print(state.deltaParameters:max(),state.deltaParameters:min(),state.deltaParameters:mean(),clr)
   else
      x:add(-clr, dfdx)
   end
   --print(dfdx:max(),dfdx:min(),dfdx:mean())
   state.x1:copy(x):cmul(clipV):clamp(-1,1)
   x:cmul(state.clipVnot):add(state.x1)
   --print('---------------')
   --print(state.x1:max(),state.x1:min(),x:max(),x:min())
  --x:clamp(-1,1)
   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
