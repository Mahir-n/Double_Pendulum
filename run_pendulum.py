"""
Run the double pendulum simulation.

Edit the values in the PARAMETERS section below, then run:
    python run_pendulum.py
"""

from double_pendulum_class import DoublePendulum

# ============================ PARAMETERS ============================
# Masses (kg) and rod lengths (m)
M1, M2 = 1.0, 2.0
L1, L2 = 1.0, 1.0
G = 9.81                    # gravitational acceleration (m/s^2)

# Starting angles (degrees, measured from the downward vertical)
THETA1_0, THETA2_0 = 120, -10

# Starting speeds of the bobs (m/s)
V1_0, V2_0 = 0.0, 0.0

# Simulation length and integration step (s)
TOTAL_TIME = 20
TIME_STEP = 0.001

# Animation settings
SKIP_FRAMES = 20            # draw every Nth step (20 x 1 ms = 20 ms per frame)
TRACE_LENGTH = 2000         # number of steps shown in the trail
SHOW_ENERGY_PLOT = True     # show the energy plot after the animation
# ====================================================================


def main():
    p = DoublePendulum(m1=M1, m2=M2, L1=L1, L2=L2, g=G)
    p.set_initial_conditions(theta1=THETA1_0, theta2=THETA2_0,
                             velocity1=V1_0, velocity2=V2_0)
    p.set_time_array(total_time=TOTAL_TIME, time_step=TIME_STEP)

    print("Solving equations of motion...")
    p.solve()
    print("Done.")

    p.animate(skip_frame_number=SKIP_FRAMES, trace__length=TRACE_LENGTH)

    if SHOW_ENERGY_PLOT:
        p.plot_energy(skip_frame_number=SKIP_FRAMES)


if __name__ == "__main__":
    main()
