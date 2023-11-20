use_jit=True
def jit(*args, **kwargs):
    def decorator(func):
        if use_jit:
            import numba
            return numba.jit(*args, **kwargs)(func)
        else:
            return func
    return decorator