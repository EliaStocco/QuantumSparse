# "fermionic_operators" class
from ..operator.operator import operators
from ..tools.functions import prepare_opts
import numpy as np
import pandas as pd
from scipy import sparse


class fermionic_operators(operators):
    
   
    def __init__(self,N,quantum_numbers,opts=None,**kwargs):
        
        print("\n\tconstructor of \"fermionic_operators\" class")   
        opts = prepare_opts(opts)
        
        self._quantum_numbers = quantum_numbers
        self._index = pd.MultiIndex.from_product(list(self._quantum_numbers.values()),\
                                                names=self._quantum_numbers.keys())
        self._template = pd.Series(index=self._index,dtype=object)
        self._constructor_operators = self._template.copy()
        self._destructor_operators  = self._template.copy()
        self._number_operators = None
                
        c_dag = fermionic_operators.constructor_fermionic_operator()
        c     = fermionic_operators.destructor_fermionic_operator()
        iden  = fermionic_operators.identity(2)

        Iden = np.full(len(self._index),iden)
        C    = Iden.copy()
        Cdag = Iden.copy()
        
        
        N = len(self._index)
        for n,i in enumerate(self._index): #quantum numbers
            print("\t\tcomputing the constructor and destructor operators",n+1,"/",N,end="\r")
            C[n] = c
            Cdag [n] = c_dag
            
            for In,Out in zip([Cdag,C],[self._constructor_operators,self._destructor_operators]):
                Out.at[i] = In[0]
                for m in range(1,len(In)):
                    Out.at[i] = sparse.kron( Out.at[i] , In[m] )
            
            # restore original values
            C[n] = iden
            Cdag[n] = iden
        print("\t\tcomputed the constructor and destructor operators          ")
            
        self._constructor_operators.astype(object)
        self._destructor_operators.astype(object)
        
        # https://realpython.com/python-super/
        super().__init__(**kwargs)
        
        return        
        
   
    def n_quantum_numbers(self):
        N = len(self._quantum_numbers)
        return N
    
   
    # def get_index(self,dtype="MultiIndex"):
    #     if dtype == "MultiIndex":
    #         return self._index
    #     elif dtype == "numpy.array":
    #         return np.asarray(list(self._index),dtype=object)
        
   
    def get_template(self,dtype=object):
        out = self._template.copy().astype(dtype)
        return out
    
   
    def sub_index(self,quantum_numbers):
        todrop = list(set(self._quantum_numbers) - set(quantum_numbers)) # not ordered
        new_indexes = self._index.droplevel(todrop).unique()
        return new_indexes
    
   
    def masks(self,quantum_numbers=None):
        if quantum_numbers is None:
            quantum_numbers = self._quantum_numbers
            
        # todrop = list(set(self._quantum_numbers) - set(quantum_numbers)) # not ordered
        # todrop_num = [list(self._quantum_numbers.keys()).index(i) for i in todrop]
        # tokeep_num = [list(self._quantum_numbers.keys()).index(i) for i in quantum_numbers]
        # todrop_num.sort()
        # tokeep_num.sort()
        # new_indexes = self._index.droplevel(todrop_num).unique()
        # index_num = self.get_index(dtype="numpy.array")
        
        # out = pd.Series(index=new_indexes,dtype=object)
        # for i in new_indexes:
        #     out.at[i] = [tuple(index_num[j,tokeep_num]) == tuple([i]) for j in range(len(index_num))]
        #     out.at[i] = list(self._index[out.at[i]])
            
        # return out
        
        tokeep_num = [list(self._quantum_numbers.keys()).index(i) for i in quantum_numbers]
        tokeep_num.sort()
        if len(tokeep_num) == 1 :
            tokeep_num = tokeep_num[0]
        
        new_indexes = self.sub_index(quantum_numbers)
        out = pd.Series(index=new_indexes,dtype=object)
        old_index = pd.Series(index=self._index,data=list(self._index))
        
        for i in new_indexes:
            a = old_index.xs(i,level=tokeep_num)
            out.at[i] = list(a)
            
        return out
    
   
    # def index(self,key=None):
    #     if key is None:
    #         out = self._index
    #     elif key not in self._index.keys():
    #         print("\n\terror in \"index\" method: key \"",key,"\" does not exists")
    #         raise()
    #     else :
    #         out = self._index[key]
    #     return out
    
   
    def quantum_numbers(self,only_keys=False,dtype=list):
        if only_keys:
            out =  self._quantum_numbers.keys()
            if dtype is not None:
                out = dtype(out)
        else :
            out = self._quantum_numbers
        return out
    
   
    def number_operators(self,quantum_numbers=None):
        if self._number_operators is None:
            self._number_operators = self._template.copy()
            for i in self._index:
                self._number_operators.at[i] = \
                    self._constructor_operators.at[i] @ \
                    self._destructor_operators.at[i]
                    
        if quantum_numbers == "all" or quantum_numbers == self.quantum_numbers(only_keys=True) :
            out = self._number_operators.copy()
        elif quantum_numbers is None or quantum_numbers == []:
            out = self._number_operators.sum()
        else :            
            masks = self.masks(quantum_numbers)
            out = pd.Series(index=self.sub_index(quantum_numbers),\
                            dtype=object)
            for i in out.index:
                out.at[i] = self._number_operators[masks[i]].sum()
            
        return out
    
   
    @staticmethod
    def constructor_fermionic_operator():
        c_dag = sparse.diags([1],offsets= 1)#.todense()
        return c_dag
            
   
    @staticmethod
    def destructor_fermionic_operator():
        c = sparse.diags([1],offsets=-1)#.todense()
        return c       