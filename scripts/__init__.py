# __init__.py

#check python version/feature compatibility
# python2 does not support * operator on lists
try:
    eval("*first, second = [1,2,3,4,5]")
except SyntaxError:
    raise ImportError("requires '*' operator on list splitting")

    #import modules
    