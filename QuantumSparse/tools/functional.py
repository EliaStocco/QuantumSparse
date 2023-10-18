
def output(cls):
    def decorator(func):
        def func_wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            _out = list(out)
            for n in range(len(_out)):
                _out[n] = cls(_out[n])
            return type(out)(_out)
        return func_wrapper
    return decorator