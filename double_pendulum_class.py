import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
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
    def animate(self,skip_frame_number = 50,trace__length = 2000,repettion = "no"):

        self.set_coordinates()

        fig,ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-self.l, self.l)
        ax.set_ylim(-self.l, self.l)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.set_title('Double Pendulum', fontsize=20, fontweight='bold', color='#2F3645')
        ax.set_xlabel('x', fontsize=15)
        ax.set_ylabel('y', fontsize=15)

        String_1, = ax.plot([], [], lw=2, color='#DCA47C')
        String_2, = ax.plot([], [], lw=2, color='#DCA47C')
        mass1, = ax.plot([], [], 'o', color='#800000', markersize=self.mass_1 /(self.mass_1 + self.mass_2) * 20)
        mass2, = ax.plot([], [], 'o', color='#800000', markersize=self.mass_2 /(self.mass_1 + self.mass_2) * 20)
        trajectory_1, = ax.plot([], [], lw=0.5, color='#800000')
        trajectory_2, = ax.plot([], [], lw=0.5, color='#800000')


        def update_frame(frame):
                
            mass1.set_data([self.x1[frame]], [self.y1[frame]])
            mass2.set_data([self.x2[frame]], [self.y2[frame]])
            String_1.set_data([0, self.x1[frame]], [0, self.y1[frame]])
            String_2.set_data([self.x1[frame], self.x2[frame]], [self.y1[frame], self.y2[frame]])
            trajectory_1.set_data(self.x1[max(0, frame - trace__length):frame], self.y1[max(0, frame - trace__length):frame])
            trajectory_2.set_data(self.x2[max(0, frame - trace__length):frame], self.y2[max(0, frame - trace__length):frame])
        
            return mass1, mass2, String_1, String_2, trajectory_1, trajectory_2
        
        # number of frames to skip
        skip_frames = skip_frame_number
        frame_indices = np.arange(0, len(self.time), skip_frames)

        ani = animation.FuncAnimation(fig, update_frame, frames=frame_indices, blit=True, interval=1,repeat = False if repettion == "no" else True)

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


            
        
    

    
