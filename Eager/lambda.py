# list1 = [3 , 5 , -1, 0 , -2, -6]
# list1 = sorted(list1, key = lambda x: abs(x))
# print(list1)


# def get_y(a,b):
#     return lambda x: a*x+b 
# y1 =get_y(100,1)
# print(y1(1))

 
def get_y(a,b):
    def func(x):
        return a*x+b
    return func
y1 =get_y(100,1)
print(y1(1))