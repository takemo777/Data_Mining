import numpy as np

a=np.array([1, 1, 0.8])
b=np.array([0.45, 0.4, 0.6])

a_2=np.array([0.5, 0.4, 0.6])
b_2=np.array([-1.0, -0.79, -1.3])

a_3=np.array([-1.5, 1.4, 3.6])
b_3=np.array([-4.5, 2.9, 0.6])

def get_cos_ruijido(x, y):
    
    norm_x = np.linalg.norm(x, ord=2)
    norm_y = np.linalg.norm(y, ord=2)
    
    return x @ y / (norm_x * norm_y)

print(get_cos_ruijido(a, b))
print(get_cos_ruijido(a_2, b_2))
print(get_cos_ruijido(a_3, b_3))
