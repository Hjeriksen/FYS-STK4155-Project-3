import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,7)
plt.rcParams.update({'font.size': 25})
plt.rcParams['axes.linewidth'] = 1.5


def diffusion1d_analytical(x, t):

    """
    Calculates the analytical function value for the one-dimensional diffusion equation. This function assumes the diffusion coefficient to be 1 and the spatial domain to be [0,1].

    Arguments:
        x (float): The spatial coordinate.
        t (float): The time coordinate.

    Returns:
        float: The function value.
    """

    if x < 0 or x > 1:
        raise ValueError('Spatial coordinate out of bounds.')

    if t < 0:
        raise ValueError('Time coordinate negative')

    return np.sin(np.pi * x) * np.exp(- t * np.pi ** 2)

def diffusion1d_numerical(x, t, dx, dt):

    """
    Approximates the analytical function value for the one-dimensional diffusion equation using the forward Euler scheme. This function assumes the diffusion coefficient to be 1 and the spatial domain to be [0,1].

    Arguments:
        x (float): The spatial coordinate.
        t (float): The time coordinate.
        dx (float): The spatial step size.
        dt (float): The time step size.

    Returns:
        float: The function value.
    """

    # calculate number of steps
    nx = int(1 / dx)
    nt = int(t / dt)

    # initialize solution
    u = np.sin(np.pi * np.linspace(0, 1, nx + 1))
    # boundary conditions
    u[0] = u[-1] = 0.0

    u_new = u.copy()
    for n in range(nt):
        for i in range(1, nx, 1):
            u_new[i] = u[i] + dt * (u[i+1] - 2 * u[i] + u[i-1]) / dx ** 2
        u = u_new.copy()

    return u

def plot_u(ts):

    # choosing step sizes
    dx = 1 / 100
    dt = 0.1 * dx ** 2

    for t in ts:
        x = np.linspace(0, 1, 1001)
        u_anal = [diffusion1d_analytical(_, t) for _ in x]
        plt.plot(x, u_anal, c='black')

        u_num = diffusion1d_numerical(0, t, dx, dt)
        x = np.linspace(0, 1, len(u_num))
        plt.plot(x, u_num, marker='o', linestyle='', linewidth=2, markersize=6, label='t='+str(t))


    ## choosing step sizes
    #dx = 1 / 100
    #dt = 0.1 * dx ** 2

    #u_num = diffusion1d_numerical(0, t, dx, dt)
    #x = np.linspace(0, 1, len(u_num))
    #plt.plot(x, u_num, marker='x', linestyle='dashed', linewidth=2, markersize=12)


    #u_anal = [diffusion1d_analytical(_, t) for _ in x]

    #plt.plot(x, u_anal)
    plt.legend(loc='best')

    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()
def plot_u_t_anal(ts):

    x = np.linspace(0, 1, 1001)

    for t in ts:
        u_anal = [diffusion1d_analytical(_, t) for _ in x]
        plt.plot(x, u_anal, label='t='+str(t), linewidth=2)

    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(loc='best')
    plt.show()


plot_u_t_anal([0, 0.1, 0.3])

plot_u([0.1,0.3])

