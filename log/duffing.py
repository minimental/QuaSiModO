##
# D U F F I N G . P Y
#
# This code implements the Duffing equation
#
#    dx1 = x2
#    dx2 = -delta * x2 - alpha * x1 - beta * x1³ + u,
#
# just to get an impression how it (autonomously) behaves.
#
# The equation is taken from the Jupyter notebook "Standalone_Duffing_EDMD" of
# Sebastian Peitz and Katharina Bieker, department of mathematics at the Paderborn University
#
# GitHub:
# https://github.com/SebastianPeitz/QuaSiModO
# commit: e04c060 on January 20 2023
#
# Author: Herrmann, Max
# Date: Fri JUN 30 2023
# Revision:  0   Fri JUN 30 2023   First version / Herrmann, Max

from numpy import linspace
from numpy import array
from numpy import interp
from numpy import nonzero
from numpy import min as smallest

from scipy.integrate import odeint

import matplotlib.pyplot as plotter

# parameters
alpha = -1
beta = 1
delta = -0.1

def rhs(x, t, u, time):

    # assign states
    x1 = x[0]
    x2 = x[1]

    # determine input
    u_current = interp(t, time, u, left=0, right=0)

    # compute derivatives
    dx1 = x2
    dx2 = -delta * x2 - alpha * x1 - beta * x1**3 + u_current

    return [dx1, dx2]

# time vector
time_resolution = 128
time_start = 0
time_end = 8
time = linspace(time_start, time_end, num=time_resolution)

# inputs
u0 = array([0]*time_resolution)
u_impulse = array([0]*time_resolution)
u_impulse[nonzero(time < 2)] = 1
index_of_u_turned_off = smallest(nonzero(time > 2))

# initial conditions
x0 = [0, 1]

# integrate
sol_autonomous = odeint(rhs, x0, time, args=(u0, time))
sol_forced = odeint(rhs, [0, 0], time, args=(u_impulse, time))

# plot
figure = plotter.figure()
axis = figure.add_subplot(title='Duffing equation\nalpha=-1, beta=1, delta=-0.1', xlabel='$x_1$', ylabel='$x_2$')

axis.plot(sol_autonomous[:, 0], sol_autonomous[:, 1], label='autonomous')
axis.plot(sol_forced[:, 0], sol_forced[:, 1], label='forced')
axis.plot(x0[0], x0[1], marker='x', linestyle='', label='$x_0$', color='green')
axis.plot(0, 0, marker='x', color='green')
axis.plot(sol_forced[index_of_u_turned_off, 0], sol_forced[index_of_u_turned_off, 1], marker='v', label='input turned off')
axis.legend()
plotter.show()
