from double_pendulum_class import DoublePendulum

# masses (kg) and rod lengths (m)
p = DoublePendulum(m1=5, m2=1, L1=1, L2=1)

# angles in degrees, starting velocities in m/s
p.set_initial_conditions(theta1=120, theta2=-10, velocity1=0, velocity2=0)

# simulate 20 s with a 0.001 s time step
p.set_time_array(total_time=20, time_step=0.001)

print("Solving...")
p.solve()

p.animate(skip_frame_number=20)        # pendulum animation
p.plot_energy()    # energy plot opens after you close the animation