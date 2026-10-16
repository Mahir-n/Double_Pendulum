import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle
plt.style.use('Solarize_Light2')

# Define the class
class DoublePendulum:
    def __init__(self, m1, m2, L1, L2, g = 9.81):

        # Initialize the parameters
        self.mass_1 = m1
        self.mass_2 = m2
        self.length_1 = L1
        self.length_2 = L2
        self.gravity = g
        self.l = L1 + L2

    
    # Initialize the conditions at t = 0
    def set_initial_conditions(self, theta1, theta2, velocity1, velocity2):
        # Set the initial conditions
        self.theta1_0 = theta1
        self.theta2_0 = theta2
        self.velocity1_0 = velocity1
        self.velocity2_0 = velocity2

        return None

    # set the time array
    def set_time_array(self,total_time, time_step):

        # Define the time array
        self.time = np.arange(0, total_time, time_step)

        # Initialize the arrays to store the results
        self.theta1 = np.zeros(len(self.time))
        self.theta2 = np.zeros(len(self.time))
        self.omega1 = np.zeros(len(self.time))
        self.omega2 = np.zeros(len(self.time))
    

        # Set the initial conditions
        self.theta1[0] = self.theta1_0 * np.pi/180
        self.theta2[0] = self.theta2_0 * np.pi/180
        self.omega1[0] = self.velocity1_0 / self.length_1
        self.omega2[0] = self.velocity2_0 / self.length_2


        return None
    
    # Solve the differential equations
    def solve(self,damping_coefficient = lambda t: 0):
        
        # Define the parameters in short form
        m1 = self.mass_1
        m2 = self.mass_2
        l1 = self.length_1
        l2 = self.length_2
        g = self.gravity
        b = damping_coefficient
        t = self.time
        th1 = self.theta1
        th2 = self.theta2
        w1 = self.omega1
        w2 = self.omega2
        h = t[1] - t[0]

        # Define the differential equations
        th1_d = lambda time, th1, th2, w1, w2: w1
        th2_d = lambda time, th1, th2, w1, w2: w2
        w1_d = lambda time, th1, th2, w1, w2: (m2 * l1**2 * w1**2 * np.cos(th1 - th2) * np.sin(th1 - th2) - m2 * l1 * g * np.cos(th1 - th2) * np.sin(th2) -
                                            b(time) * l1 * l2 * w2 * np.cos(th1 - th2) - b(time) * l1**2 * w1 * np.cos(th1 - th2)**2 + m2 * l1 * l2 * w2**2 * np.sin(th1 - th2) +
                                            (m1 + m2) * g * l1 * np.sin(th1) + 2 * b(time) * l1**2 * w1 + b(time) * l1 * l2 * w2 * np.cos(th1 - th2)) / (m2 * l1**2 * np.cos(th1 - th2)**2 - (m1 + m2) * l1**2)
        w2_d = lambda time, th1, th2, w1, w2: (m2 * l1 * l2 * w2**2 * np.sin(th1 - th2) + (m1 + m2) * g * l1 * np.sin(th1) + 2 * b(time) * l1**2 * w1 + b(time) * l1 * l2 * w2 * np.cos(th1 - th2) +
                                            (m1 + m2) * l1**2 * w1_d(time, th1, th2, w1, w2)) / (-m2 * l1 * l2 * np.cos(th1 - th2))


        # Numerical integration using Runge-Kutta method
        for i in range(len(t) - 1):
            th1_f1 = th1_d(t[i], th1[i], th2[i], w1[i], w2[i])
            th2_f1 = th2_d(t[i], th1[i], th2[i], w1[i], w2[i])
            w1_f1 = w1_d(t[i], th1[i], th2[i], w1[i], w2[i])
            w2_f1 = w2_d(t[i], th1[i], th2[i], w1[i], w2[i])

            th1_f2 = th1_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f1, th2[i] + (h / 2) * th2_f1, w1[i] + (h / 2) * w1_f1, w2[i] + (h / 2) * w2_f1)
            th2_f2 = th2_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f1, th2[i] + (h / 2) * th2_f1, w1[i] + (h / 2) * w1_f1, w2[i] + (h / 2) * w2_f1)
            w1_f2 = w1_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f1, th2[i] + (h / 2) * th2_f1, w1[i] + (h / 2) * w1_f1, w2[i] + (h / 2) * w2_f1)
            w2_f2 = w2_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f1, th2[i] + (h / 2) * th2_f1, w1[i] + (h / 2) * w1_f1, w2[i] + (h / 2) * w2_f1)

            th1_f3 = th1_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f2, th2[i] + (h / 2) * th2_f2, w1[i] + (h / 2) * w1_f2, w2[i] + (h / 2) * w2_f2)
            th2_f3 = th2_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f2, th2[i] + (h / 2) * th2_f2, w1[i] + (h / 2) * w1_f2, w2[i] + (h / 2) * w2_f2)
            w1_f3 = w1_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f2, th2[i] + (h / 2) * th2_f2, w1[i] + (h / 2) * w1_f2, w2[i] + (h / 2) * w2_f2)
            w2_f3 = w2_d(t[i] + (h / 2), th1[i] + (h / 2) * th1_f2, th2[i] + (h / 2) * th2_f2, w1[i] + (h / 2) * w1_f2, w2[i] + (h / 2) * w2_f2)

            th1_f4 = th1_d(t[i] + h, th1[i] + h * th1_f3, th2[i] + h * th2_f3, w1[i] + h * w1_f3, w2[i] + h * w2_f3)
            th2_f4 = th2_d(t[i] + h, th1[i] + h * th1_f3, th2[i] + h * th2_f3, w1[i] + h * w1_f3, w2[i] + h * w2_f3)
            w1_f4 = w1_d(t[i] + h, th1[i] + h * th1_f3, th2[i] + h * th2_f3, w1[i] + h * w1_f3, w2[i] + h * w2_f3)
            w2_f4 = w2_d(t[i] + h, th1[i] + h * th1_f3, th2[i] + h * th2_f3, w1[i] + h * w1_f3, w2[i] + h * w2_f3)

            th1[i + 1] = th1[i] + (h / 6) * (th1_f1 + 2 * th1_f2 + 2 * th1_f3 + th1_f4)
            th2[i + 1] = th2[i] + (h / 6) * (th2_f1 + 2 * th2_f2 + 2 * th2_f3 + th2_f4)
            w1[i + 1] = w1[i] + (h / 6) * (w1_f1 + 2 * w1_f2 + 2 * w1_f3 + w1_f4)
            w2[i + 1] = w2[i] + (h / 6) * (w2_f1 + 2 * w2_f2 + 2 * w2_f3 + w2_f4)


        self.theta1 = th1
        self.theta2 = th2
        self.omega1 = w1
        self.omega2 = w2
        


    # Set the coordinates of the pendulum
    def set_coordinates(self):

        # Define the coordinates of the pendulum
        self.x1 = self.length_1 * np.sin(self.theta1)
        self.y1 = -self.length_1 * np.cos(self.theta1)
        self.x2 = self.x1 + self.length_2 * np.sin(self.theta2)
        self.y2 = self.y1 - self.length_2 * np.cos(self.theta2)


    # animate the pendulum
    def animate(self, skip_frame_number=50, trace__length=2000, repettion="no"):
        self.set_coordinates()

        # Layout: pendulum on the left, info panel on the right
        fig, (ax, info) = plt.subplots(1, 2, figsize=(12, 7),
                                       gridspec_kw={'width_ratios': [3, 1.3]})
        fig.suptitle('Double Pendulum Simulation', fontsize=20,
                     fontweight='bold', color='#2F3645')

        c1 = '#800000'   # colour of bob 1
        c2 = '#1F4E79'   # colour of bob 2

        # ---------- Pendulum plot ----------
        ax.set_aspect('equal')
        ax.set_xlim(-1.2 * self.l, 1.2 * self.l)
        ax.set_ylim(-1.2 * self.l, 1.2 * self.l)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('x (m)', fontsize=13)
        ax.set_ylabel('y (m)', fontsize=13)

        # pivot point
        ax.plot(0, 0, marker='s', color='#2F3645', markersize=8, zorder=4)

        String_1, = ax.plot([], [], lw=2, color='#DCA47C')
        String_2, = ax.plot([], [], lw=2, color='#DCA47C')
        trajectory_1, = ax.plot([], [], lw=0.6, color=c1, alpha=0.6)
        trajectory_2, = ax.plot([], [], lw=0.6, color=c2, alpha=0.6)

        # bob radius proportional to cube root of mass
        r1 = 0.04 * self.l * self.mass_1 ** (1/3)
        r2 = 0.04 * self.l * self.mass_2 ** (1/3)
        mass1 = Circle((self.x1[0], self.y1[0]), r1, color=c1, zorder=3)
        mass2 = Circle((self.x2[0], self.y2[0]), r2, color=c2, zorder=3)
        ax.add_patch(mass1)
        ax.add_patch(mass2)

        # labels that follow the bobs
        label1 = ax.text(0, 0, 'm1', fontsize=11, fontweight='bold', color=c1, zorder=5)
        label2 = ax.text(0, 0, 'm2', fontsize=11, fontweight='bold', color=c2, zorder=5)

        # ---------- Info panel ----------
        info.axis('off')
        info.set_xlim(0, 1)
        info.set_ylim(0, 1)
        box = dict(boxstyle='round,pad=0.8', facecolor='#FDF6E3', edgecolor='#93A1A1')

        info.text(0.02, 0.97, 'System Parameters', fontsize=14,
                  fontweight='bold', color='#2F3645', va='top')
        params = (f"Mass 1  (m1) : {self.mass_1:g} kg\n"
                  f"Mass 2  (m2) : {self.mass_2:g} kg\n"
                  f"Rod 1   (L1) : {self.length_1:g} m\n"
                  f"Rod 2   (L2) : {self.length_2:g} m\n"
                  f"Gravity (g)  : {self.gravity:g} m/s²\n"
                  f"\n"
                  f"Initial θ1   : {self.theta1_0:g}°\n"
                  f"Initial θ2   : {self.theta2_0:g}°")
        info.text(0.02, 0.89, params, fontsize=11, family='monospace',
                  va='top', bbox=box)

        info.text(0.02, 0.40, 'Live', fontsize=14,
                  fontweight='bold', color='#2F3645', va='top')
        live_text = info.text(0.02, 0.32, '', fontsize=11, family='monospace',
                              va='top', bbox=box)

        # keep angles between -180° and 180° for display
        wrap = lambda a: (np.degrees(a) + 180) % 360 - 180

        def update_frame(frame):
            x1, y1 = self.x1[frame], self.y1[frame]
            x2, y2 = self.x2[frame], self.y2[frame]

            mass1.set_center((x1, y1))
            mass2.set_center((x2, y2))
            String_1.set_data([0, x1], [0, y1])
            String_2.set_data([x1, x2], [y1, y2])

            start = max(0, frame - trace__length)
            trajectory_1.set_data(self.x1[start:frame], self.y1[start:frame])
            trajectory_2.set_data(self.x2[start:frame], self.y2[start:frame])

            label1.set_position((x1 + r1 * 1.2, y1 + r1 * 1.2))
            label2.set_position((x2 + r2 * 1.2, y2 + r2 * 1.2))

            live_text.set_text(f"Time : {self.time[frame]:6.2f} s\n"
                               f"θ1   : {wrap(self.theta1[frame]):7.1f}°\n"
                               f"θ2   : {wrap(self.theta2[frame]):7.1f}°")

            return (mass1, mass2, String_1, String_2,
                    trajectory_1, trajectory_2, label1, label2, live_text)

        skip_frames = skip_frame_number
        frame_indices = np.arange(0, len(self.time), skip_frames)

        dt = self.time[1] - self.time[0]
        real_interval = skip_frames * dt * 1000   # ms of real time per frame

        fig.tight_layout()
        ani = animation.FuncAnimation(fig, update_frame, frames=frame_indices,
                                      blit=True, interval=real_interval,
                                      repeat=False if repettion == "no" else True)
        plt.show()

    def energy(self):

        # Calculate the kinetic energy
        self.KE = 0.5 * (self.mass_1 + self.mass_2) * self.length_1**2 * self.omega1**2 + 0.5 * self.mass_2 * self.length_2**2 * self.omega2**2 + self.mass_2 * self.length_1 * self.length_2 * self.omega1 * self.omega2 * np.cos(self.theta1 - self.theta2)

        # Calculate the potential energy
        self.PE = -(self.mass_1 + self.mass_2) * self.gravity * self.length_1 * np.cos(self.theta1) - self.mass_2 * self.gravity * self.length_2 * np.cos(self.theta2)

        # Calculate the total energy
        self.total_energy = self.KE + self.PE
        


    
    def plot_energy(self,skip_frame_number = 50):
            
            self.energy()
            
            fig, ax = plt.subplots()
            ax.set_xlim(0, self.time[-1])
            ax.set_ylim(0.95 * min(min(self.KE), min(self.PE), min(self.total_energy)), 1.05 * max(max(self.KE), max(self.PE), max(self.total_energy)))
            ax.set_title('Energy of the Double Pendulum', fontsize=20, fontweight='bold', color='#2F3645')
            ax.set_xlabel('Time', fontsize=15)
            ax.set_ylabel('Energy', fontsize=15)
            
            ke, = ax.plot([], [], lw=2, color='#FFAD60', label='Kinetic Energy')
            pe, = ax.plot([], [], lw=2, color='#FF8C9E', label='Potential Energy')
            te, = ax.plot([], [], lw=2, color='#D95F59', label='Total Energy')

            ke_edge, = ax.plot([], [], 'o', color='#FFAD60', markersize=5)
            pe_edge, = ax.plot([], [], 'o', color='#FF8C9E', markersize=5)
            te_edge, = ax.plot([], [], 'o', color='#D95F59', markersize=5)

            def update_frame(frame):
                    
                    ke.set_data(self.time[:frame], self.KE[:frame])
                    pe.set_data(self.time[:frame], self.PE[:frame])
                    te.set_data(self.time[:frame], self.total_energy[:frame])
                    ke_edge.set_data([self.time[frame]], [self.KE[frame]])
                    pe_edge.set_data([self.time[frame]], [self.PE[frame]])
                    te_edge.set_data([self.time[frame]], [self.total_energy[frame]])
    
                    return ke, pe, te, ke_edge, pe_edge, te_edge
            
            # number of frames to skip
            skip_frames = skip_frame_number
            frame_indices = np.arange(0, len(self.time), skip_frames)
            
            ani = animation.FuncAnimation(fig, update_frame, frames=frame_indices, blit=True, interval=1,repeat = False)

            plt.legend()
            plt.show()


            
        
    

    
