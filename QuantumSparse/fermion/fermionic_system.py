# "fermionic_system" class
from .fermionic_operators import fermionic_operators
from ..system.system import system


class fermionic_system(fermionic_operators,system):
    
   
    def __init__(self,N=1,deg_names=["spin"],deg_keys=[["up","dw"]],opts=None,**kwargs):
        
        super(fermionic_operators, self).__init__(N,deg_names,deg_keys,opts)
        super(system, self).__init__(0)
        
        # https://realpython.com/python-super/
        super().__init__(**kwargs)
        
        return
    
   
    def nsites(self):
        N = len(self.index)
        return N
    
    #
    # def multi_index_template(self,group_by=None):
    #     if group_by is None:
    #         out = self.operators_template
    #     elif group_by
        
