
def f_x(x):
    ret=x**4 -3*x**3+2
    return ret

def bruto_force():
    x_start=-10.0
    x_end=10.0
    max_iter=1000
#    learning_rate=0.01
#    learning_rate=0.0001
    learning_rate=0.1
    iter=0
    min_x=x_start
    min_f_x=f_x(min_x)
    x=x_start

    while x<x_end:
        new_f_x=f_x(x)
        if new_f_x<min_f_x:
            min_x=x
            min_f_x=new_f_x
        x=x+learning_rate

    print('Local Minimum at', min_x)
    print('Function Value ', min_f_x)

bruto_force()