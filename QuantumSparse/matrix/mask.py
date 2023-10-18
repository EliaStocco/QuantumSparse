from scipy import sparse
import numpy

def diags(*argc,**argv):
    """diagonal matrix"""
    return sparse.diags(*argc,**argv)

def kron(*argc,**argv):
    """kronecker product"""
    return sparse.kron(*argc,**argv)

def identity(*argc,**argv):
    """identity operator"""
    return sparse.identity(*argc,**argv)