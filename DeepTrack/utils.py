def hasfunction(o:any, function_name:str):
    return hasattr(object, function_name) and callable(o.function_name) 
    
def isiterable(o:any):
    return hasattr(o, "__iter__") or hasattr(o, "__getitem__")