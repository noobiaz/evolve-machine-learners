
def f_x(x):
    ret=x**4 -3*x**3+2
    return ret

def f_prime_x(x):
    ret=4*x**3-9*x**2
    return ret

def gradient_descent():
    x_old=0.0
    x_new=6.0
    precission=0.00001
    max_iter=1000
 #   learning_rate=0.01
    learning_rate=0.0001
#    learning_rate=0.1
    iter=0

    while abs(x_new-x_old)>precission and iter<max_iter:
        x_old=x_new
        x_new=x_old-learning_rate*f_prime_x(x_old)
        iter=iter+1
        print('iter',iter,'new_x', x_new)

    print('Local Minimum at', x_new)
    print('Function Value ', f_x(x_new))

gradient_descent()