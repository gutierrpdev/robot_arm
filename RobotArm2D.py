from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from MotionPlanner2D import MotionPlanner2D as mp2D


class RobotArm2D:

    def __init__(self,
                 init_state=np.zeros(2, float),
                 origin=(0, 0)):
        self.init_state = np.copy(np.array(init_state))
        self.origin = origin
        self.time_elapsed = 0

        self.state = self.init_state * np.pi / 180.

    def position(self):
        """compute the current x,y positions of the robot arms"""
        x = np.cumsum(np.concatenate((np.array([self.origin[0]], float), cos(self.state)), axis=None))
        y = np.cumsum(np.concatenate((np.array([self.origin[1]], float), sin(self.state)), axis=None))
        return x, y

    def step(self, angles):
        """execute one time step of length dt and update state"""
        self.state = angles
        self.time_elapsed += dt


# ------------------------------------------------------------
# set up initial state and global variables
start = [np.pi/4, 0, np.pi/2, np.pi]
end = [np.pi, np.pi, np.pi, 0]
robotArm = RobotArm2D(np.array(start))
dt = 1. / 30  # 30 fps
mp = mp2D(len(start))
path = mp.apply_algorithm(len(start) - 1, start, end, 300)
path = np.array(path)

# ------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-4, 4), ylim=(-4, 4))
ax.grid()

line, = ax.plot([], [], 'o-', lw=3)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    """perform animation step"""
    global robotArm, dt
    robotArm.step(np.array(path)[i])

    line.set_data(*robotArm.position())
    time_text.set_text('time = %.1f' % robotArm.time_elapsed)
    return line, time_text


# choose the interval based on dt and the time to animate one step
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
