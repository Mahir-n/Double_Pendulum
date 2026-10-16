# Double Pendulum Simulation

A Python simulation of a double pendulum (two masses on rigid, massless rods). The equations of motion are solved with a hand-written 4th-order Runge–Kutta integrator and animated in real time with Matplotlib.

The double pendulum is one of the simplest physical systems that shows **chaos**: changing a starting angle by a fraction of a degree gives a completely different motion after a few seconds.

<!-- To show a preview: take a screenshot of the animation, save it as media/demo.png, and remove these comment markers from the line below. -->
<!-- ![Double pendulum animation](media/demo.png) -->

## Features

- Equations of motion derived from Lagrangian mechanics and integrated with RK4
- Real-time playback, so animation speed matches simulated time
- Bob size scales with the cube root of mass (bobs treated as solid spheres of the same material, so volume ∝ mass)
- Trail showing the recent path of each bob
- Side panel with the system parameters and a live readout of time and angles
- Energy plot (kinetic, potential, total) to check energy conservation
- Optional time-dependent damping

## Project structure

```
Double_Pendulum/
├── double_pendulum_class.py   # DoublePendulum class: physics, solver, animation, energy plot
├── run_pendulum.py            # Entry point: set the parameters here and run
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files Git should not track (venv, caches)
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10 or newer (tested on Python 3.14)
- NumPy and Matplotlib (installed in the setup below)

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/Mahir-n/Double_Pendulum.git
cd Double_Pendulum
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate it**

| System | Command |
|---|---|
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| Windows (Command Prompt) | `venv\Scripts\activate.bat` |
| Linux / macOS | `source venv/bin/activate` |

> **Windows PowerShell:** if activation fails with *"running scripts is disabled on this system"*, run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**4. Install the dependencies**

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_pendulum.py
```

The solver runs first and takes a few seconds. Then the animation window opens. Close it to see the energy plot.

### Changing the parameters

All settings are in the **PARAMETERS** section at the top of `run_pendulum.py`:

| Variable | Meaning | Unit |
|---|---|---|
| `M1`, `M2` | Masses of the upper and lower bob | kg |
| `L1`, `L2` | Lengths of the upper and lower rod | m |
| `G` | Gravitational acceleration | m/s² |
| `THETA1_0`, `THETA2_0` | Starting angles, measured from the downward vertical | degrees |
| `V1_0`, `V2_0` | Starting speeds of the bobs | m/s |
| `TOTAL_TIME` | Length of the simulation | s |
| `TIME_STEP` | Integration step (smaller is more accurate but slower) | s |
| `SKIP_FRAMES` | Draw every N-th step (frame time = N × `TIME_STEP`) | — |
| `TRACE_LENGTH` | Number of steps shown in the trail | — |
| `SHOW_ENERGY_PLOT` | Show the energy plot after the animation | — |

### Using the class directly

```python
from double_pendulum_class import DoublePendulum

p = DoublePendulum(m1=1, m2=2, L1=1, L2=1)
p.set_initial_conditions(theta1=120, theta2=-10, velocity1=0, velocity2=0)
p.set_time_array(total_time=20, time_step=0.001)

p.solve()                                  # no damping
# p.solve(damping_coefficient=lambda t: 0.1)   # with damping (can depend on time)

p.animate(skip_frame_number=20)
p.plot_energy()
```

## Physics

The angles $\theta_1$ and $\theta_2$ are measured from the downward vertical. The bob positions are

$$x_1 = L_1\sin\theta_1,\qquad y_1 = -L_1\cos\theta_1$$

$$x_2 = x_1 + L_2\sin\theta_2,\qquad y_2 = y_1 - L_2\cos\theta_2$$

The kinetic and potential energies are

$$T = \tfrac{1}{2}(m_1+m_2)L_1^2\dot\theta_1^2 + \tfrac{1}{2}m_2L_2^2\dot\theta_2^2 + m_2L_1L_2\,\dot\theta_1\dot\theta_2\cos(\theta_1-\theta_2)$$

$$V = -(m_1+m_2)\,g\,L_1\cos\theta_1 - m_2\,g\,L_2\cos\theta_2$$

Applying the Euler–Lagrange equations to $\mathcal{L} = T - V$ gives two coupled second-order differential equations. These are rewritten as four first-order equations in $(\theta_1, \theta_2, \omega_1, \omega_2)$ and integrated with the classic 4th-order Runge–Kutta method.

### Checking accuracy

Without damping, the total energy $E = T + V$ should stay constant. The energy plot is a quick check of the numerical accuracy: if the total energy drifts noticeably, reduce `TIME_STEP`.

## License

This project is released under the [MIT License](LICENSE).
