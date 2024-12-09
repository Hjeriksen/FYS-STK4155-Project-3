import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pde import *

plt.rcParams["figure.figsize"] = (10,7)
plt.rcParams.update({'font.size': 25})
plt.rcParams['axes.linewidth'] = 1.5

class NN(nn.Module):
    def __init__(self, n_layers, n_hidden, activation_function):

        super(NN, self).__init__()

        hidden = []
        for i in range(n_layers):
            hidden.append(nn.Linear(n_hidden, n_hidden))
            hidden.append(activation_function())

        self.layers = nn.Sequential(
            nn.Linear(2, n_hidden),
            activation_function(),
            *hidden,
            #nn.Linear(n_hidden, n_hidden),
            #nn.Tanh(),
            #nn.Linear(n_hidden, n_hidden),
            #nn.Tanh(),
            #nn.Linear(n_hidden, n_hidden),
            #nn.Tanh(),
            #nn.Linear(n_hidden, n_hidden),
            #nn.Tanh(),
            #nn.Linear(n_hidden, n_hidden),
            #nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

        #self.input_layer = nn.Linear(2, 1)

    def forward(self, x, t):
        xt = torch.stack([x, t]).T
        return self.layers(xt)

def initial_loss(model, xt):

    x = xt[:, 0]
    t = xt[:, 1]
    u_pred = model(x, t)
    u_init = torch.sin(np.pi * x)

    return torch.mean((u_pred - u_init)**2)

def boundary_loss(model, xt):

    x = xt[:, 0]
    t = xt[:, 1]
    u_pred = model(x, t)

    return torch.mean((u_pred)**2)

def physics_loss(model, xt):

    xt.requires_grad_(True)
    x = xt[:, 0]
    t = xt[:, 1]

    # Prediction u(x, t)
    u = model(x, t)

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Heat equation residual
    residual = u_t - u_xx

    return torch.mean(residual**2)

def total_loss(model, xt_collocation, xt_boundary, xt_initial):

    return physics_loss(model, xt_collocation) + boundary_loss(model, xt_boundary) + initial_loss(model, xt_initial)

def run(n_layers, n_hidden, activation_function, epochs=10000, learning_rate=0.01):

    model = NN(n_layers=n_layers, n_hidden=n_hidden, activation_function=activation_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        optimizer.zero_grad()

        n_collocation = 1000
        n_boundary = 2000
        n_initial = 1000

        # sample random collocation points
        x_collocation = torch.rand(n_collocation)
        t_collocation = torch.rand(n_collocation)
        xt_collocation = torch.stack([x_collocation, t_collocation]).T

        # sample random boundary points
        x_boundary = torch.randint(high=2, size=(n_boundary,))
        t_boundary = torch.rand(n_boundary)
        xt_boundary = torch.stack([x_boundary, t_boundary]).T

        # sample random initial points
        x_initial = torch.rand(n_initial)
        t_initial = torch.zeros(n_initial)
        xt_initial = torch.stack([x_initial, t_initial]).T

        # compute the total loss
        loss = total_loss(model, xt_collocation, xt_boundary, xt_initial)

        # backpropagation
        loss.backward()
        optimizer.step()

        # printing
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model

layers = [1,2,3,4,5,6,7]
hiddens = [10, 20, 30, 40]
activation_functions = [nn.Tanh, nn.Sigmoid, nn.ReLU]

for layer in layers:
    for hidden in hiddens:
        for activation_function in activation_functions:

            file_name = 'pinn-nl_' + str(layer) + '-nh_' + str(hidden) + '-act_' + str(activation_function.__name__) + '.png'

            model = run(layer, hidden, activation_function)

            # test
            x = torch.linspace(0, 1, 1001)
            t = torch.ones(1001) * 0.1
            print(model(torch.tensor([0.0]), torch.tensor([0.1])))
            u = model(x,t).detach().numpy()

            plt.plot(x, u, linewidth=2, label='t='+str(0.1))

            x = torch.linspace(0, 1, 1001)
            t = torch.ones(1001) * 0.3
            print(model(torch.tensor([0.0]), torch.tensor([0.0])))
            u = model(x,t).detach().numpy()

            plt.plot(x, u, linewidth=2, label='t='+str(0.3))

            x = np.linspace(0, 1, 1001)
            u_anal = [diffusion1d_analytical(_, 0.1) for _ in x]
            plt.plot(x, u_anal, c='black')
            u_anal = [diffusion1d_analytical(_, 0.3) for _ in x]
            plt.plot(x, u_anal, c='black')


            plt.xlabel('x')
            plt.ylabel('u')
            plt.legend(loc='best')
            plt.savefig(file_name)
            plt.clf()








