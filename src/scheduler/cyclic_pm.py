import param as pm
import typing
import holoviews as hv

class TLR(pm.Parameterized):
    min_lr = pm.Number(default=1e-6, doc='Lowerbound of lr')
    max_lr = pm.Number(default=1e-5, doc='Upperbound of lr')
    stepsize = pm.Integer(default=500, doc='stepsize in number of iterations')
    
    def __init__(self, min_lr, max_lr, stepsize, **params):
        super().__init__(**params)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.stepsize = stepsize
        
        self.slope = (self.min_lr - self.max_lr)/self.stepsize
        self.n_called = 0
        
        self.print_info()
    
    @pm.depends('min_lr', 'max_lr', 'stepsize', watch=True)
    def _update_slope(self):
        self.slope = (self.min_lr - self.max_lr)/self.stepsize
        
    def __call__(self, x:int):
        """
        x (int): iteration index 
        """
        self.n_called += 1
        x = x%(2*self.stepsize)
        return self.slope * abs(x - self.stepsize) + self.max_lr
    
    @pm.depends('stepsize', watch=True)
    def _hvplot(self, width=400, n_iters=None):
        """
        GUI version
        """
        if n_iters is None:
            n_iters=2*self.stepsize
        xs = range(n_iters)
        ys = [self(x) for x in xs]
        return hv.Curve((xs, ys)).opts(width=width)
    
    @pm.depends('stepsize', watch=True)
    def view(self):
        return self._hvplot()#hv.DynamicMap(self._hvplot())
    
    def print_info(self):
        print(f'min_lr: {self.min_lr}, max_lr: {self.max_lr}, stepsize: {self.stepsize}, '
              f'\nslope: {self.slope:9.9f}, n_called: {self.n_called}')
        