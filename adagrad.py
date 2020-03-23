import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_cl

def beale(x1, x2):
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2

def dbeale_dx(x1, x2):
    dfdx1 = 2*(1.5 - x1 + x1*x2)*(x2-1) + 2*(2.25 - x1 + x1*x2**2)*(x2**2-1) + 2*(2.625 - x1 + x1*x2**3)*(x2**3-1)
    dfdx2 = 2*(1.5 - x1 + x1*x2)*x1 + 2*(2.25 - x1 + x1*x2**2)*(2*x1*x2) + 2*(2.625 - x1 + x1*x2**3)*(3*x1*x2**2)
    return dfdx1, dfdx2

step_x1, step_x2 = 0.1, 0.1
X1, X2 = np.meshgrid(np.arange(-4, 4+step_x1, step_x1), np.arange(-4, 4+step_x2, step_x2))
Y = beale(X1,X2)

def gd_adagrad(df_dx, x0, conf_para=None):
    if conf_para is None:
        conf_para = {}

    conf_para.setdefault('n_iter', 1000)  # number of iterations
    conf_para.setdefault('learning_rate', 0.001)  # learning rate
    conf_para.setdefault('epsilon', 1e-7)

    x_traj = []
    x_traj.append(x0)
    r = np.zeros_like(x0)

    for iter in range(1, conf_para['n_iter']+1):
        dfdx = np.array(df_dx(x_traj[-1][0], x_traj[-1][1]))
        r += dfdx**2
        x_traj.append(x_traj[-1] - conf_para['learning_rate'] / (np.sqrt(r) + conf_para['epsilon']) * dfdx)

    return x_traj

x0 = np.array([1.0, 1.5])
conf_para_adag = {'n_iter':100,'learning_rate':1}
x_traj_adag = gd_adagrad(dbeale_dx, x0, conf_para_adag)
print("The final solution is (x_1,x_2) = (",x_traj_adag[-1][0],",",x_traj_adag[-1][1],")")

plt.rcParams['figure.figsize'] = [8, 8]
plt.contour(X1, X2, Y, levels=np.logspace(0, 5, 35), norm=plt_cl.LogNorm(), cmap=plt.cm.jet)
plt.title('2D Contour Plot of Beale function')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.plot(3,0.5,'k*',markersize=10)
x_traj_adag = np.array(x_traj_adag)
plt.plot(x_traj_adag[:,0],x_traj_adag[:,1],'k-')
plt.show()



















