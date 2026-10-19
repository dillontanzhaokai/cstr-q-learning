"""
Tabular Q-learning on the pc-gym CSTR
=====================================
GOAL
    Learn which cooling-jacket temperature Tc to apply, in every reactor state,
    so that BOTH the concentration C_A and the temperature T track their setpoints.

    Note what is being learned: not one "best Tc", but a POLICY, i.e. a rule
    "if the reactor looks like THIS, apply THAT Tc". This is the same as the
    gridworld, where you learn the best move for every cell, not one move overall.

THE BIG PICTURE (the reinforcement-learning loop)
    Every 0.25 min the agent:
        1. observes the state       (where is the reactor relative to the setpoint?)
        2. picks an action          (a cooling-water temperature Tc)
        3. the environment responds (the reactor physics move C_A and T)
        4. receives a reward        (minus how far C_A and T are from setpoint)
    Q-learning uses lots of these (state, action, reward, next state) samples to
    learn Q(state, action) = "how good is it, long-term, to apply this Tc here?"
    The best Tc in any state is then simply the one with the highest Q.

THE GRIDWORLD ANALOGY
    gridworld                          CSTR
    ---------                          ----
    cell (row, col)               ->   (C_A-error bin, T-error bin)
    goal cell                     ->   the setpoint (centre cell of the error grid)
    moves {up, down, left, right} ->   discrete Tc levels {295, 295.5, ..., 302} K
    "take one step"               ->   integrate the reactor ODEs for dt = 0.25 min
    reward -1 per move            ->   -(weighted squared error in C_A and T)
    episode ends at goal          ->   no terminal state; fixed 100-step horizon
    Q[row, col, move]             ->   Q[setpoint, ca_bin, T_bin, Tc_level]

HOW THE FILE IS ORGANISED
    1. The environment  - the reactor physics, setpoints and reward
    2. Discretisation   - chopping the continuous reactor into a grid
    3. Q-learning       - the learning algorithm itself (same as gridworld)
    4. Testing          - running the learned policy without exploration
    5. Main script      - train, test, compare to a baseline, plot

The ODE model and parameters are copied from pc-gym's `cstr` class
(src/pcgym/model_classes.py), so the dynamics are the ones you will meet there.
Only numpy and matplotlib are needed.  Run:  python cstr_qlearning.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# A seeded random number generator makes the run reproducible: the random
# exploration and random start states are the same every time you run it,
# so you get the same Q-table and the same plots.
rng = np.random.default_rng(0)


# ==========================================================================
# 1. THE ENVIRONMENT: the CSTR
# ==========================================================================
# In the gridworld the "environment" is the grid plus the rule "a move takes
# you to the neighbouring cell". Here the environment is the reactor and its
# physics: the mass and energy balances decide where you end up after
# applying a given Tc.
#
# KEY CONCEPT - Q-learning is MODEL-FREE:
# The agent never sees these equations. It only sees the state it lands in
# and the reward it gets. The equations are here purely to SIMULATE the
# plant. You could replace step_reactor() with a real reactor (or with
# pc-gym's env.step) and the learning code in section 3 would not change.
# --------------------------------------------------------------------------

# Physical parameters (identical to pc-gym's defaults)
q, V, rho, Cp = 100.0, 100.0, 1000.0, 0.239      # flow L/min, volume L, density g/L, heat capacity J/(g K)
dHr, EA_R, k0, UA = -5e4, 8750.0, 7.2e10, 5e4    # heat of reaction J/mol (negative = exothermic),
                                                  # activation energy / R (K), pre-exponential 1/min,
                                                  # jacket heat-transfer coefficient x area J/(min K)
Ti, Caf = 350.0, 1.0                              # feed temperature K, feed concentration mol/L

# Time discretisation. The agent makes one decision every DT minutes and
# holds that Tc constant until the next decision (like a digital controller).
DT = 25.0 / 100          # pc-gym default: tsim = 25 min split into N = 100 steps -> 0.25 min each
N_STEPS = 100            # decisions per episode (one episode = one 25-minute run of the reactor)
SUBSTEPS = 5             # RK4 sub-steps per control interval. Fine in the normal
                         # operating range; if you explore conditions where the reactor
                         # runs away (e.g. Ti >= 355 K with Tc near 301 K, T > 450 K),
                         # switch to a stiff solver such as scipy's LSODA.


def f(ca, T, Tc):
    """The reactor's physics: how fast C_A and T are changing right now.

    Identical to pc-gym's cstr.__call__. Reaction: A -> B, first order in A.
    """
    # Reaction rate. The Arrhenius term exp(-EA/RT) makes the rate very
    # sensitive to temperature. This is why the CSTR is NONLINEAR, and why
    # Tc can steer C_A at all: Tc changes T, T changes the rate, and the
    # rate changes C_A.
    rA = k0 * math.exp(-EA_R / T) * ca

    # Mass balance on A:  accumulation = (in - out by flow) - (consumed by reaction)
    dca = q / V * (Caf - ca) - rA

    # Energy balance:  accumulation = (heat in/out with the flow)
    #                                + (heat released by the exothermic reaction)
    #                                + (heat exchanged with the cooling jacket at Tc)
    # The last term is the ONLY place the action Tc enters the physics.
    # It acts on T directly, and on C_A only indirectly through rA.
    dT = q / V * (Ti - T) + (-dHr) * rA / (rho * Cp) + UA * (Tc - T) / (rho * Cp * V)
    return dca, dT


def step_reactor(ca, T, Tc):
    """ONE ENVIRONMENT STEP: hold Tc for DT minutes and see where the reactor goes.

    This is the CSTR version of "move one cell" in the gridworld, and is what
    pc-gym's env.step(action) does internally. It uses the classic 4th-order
    Runge-Kutta method to integrate the ODEs accurately over the interval.
    """
    h = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        # RK4: sample the slope at the start, twice in the middle, and at the
        # end of the sub-step, then take a weighted average of those slopes.
        k1 = f(ca, T, Tc)
        k2 = f(ca + h / 2 * k1[0], T + h / 2 * k1[1], Tc)
        k3 = f(ca + h / 2 * k2[0], T + h / 2 * k2[1], Tc)
        k4 = f(ca + h * k3[0], T + h * k3[1], Tc)
        ca += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        T += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return ca, T


def steady_state(ca_sp):
    """For a C_A setpoint, the T and Tc that would hold the reactor there forever.

    The AGENT DOES NOT USE THIS. We use it for two things only:
      (a) choosing a T setpoint that is physically compatible with the C_A setpoint;
      (b) building a model-based baseline to compare Q-learning against.

    KEY CONCEPT - one input cannot hit two arbitrary targets:
    There is one action (Tc) but two outputs (C_A, T). At steady state the
    mass balance alone fixes T once you pick C_A. So if you asked for, say,
    C_A = 0.85 AND T = 330 K, no controller could deliver both. The setpoints
    have to be consistent with each other.
    """
    # Set dC_A/dt = 0 in the mass balance and solve for the rate constant k(T)...
    k = (Caf / ca_sp - 1.0) * q / V
    # ...then invert Arrhenius, k = k0 exp(-EA/RT), to get the T that gives that k.
    T = -EA_R / math.log(k / k0)
    # Set dT/dt = 0 in the energy balance and solve for the jacket temperature Tc.
    Tc = T - (q / V * (Ti - T) + (-dHr) * k * ca_sp / (rho * Cp)) * rho * Cp * V / UA
    return T, Tc


# --------------------------------------------------------------------------
# Setpoints (the "goal cell")
# --------------------------------------------------------------------------
# Same C_A schedule as the pc-gym CSTR example (0.85 for the first half of the
# episode, then 0.90), with the matching T setpoint calculated from steady state.
# Because the setpoint CHANGES during the episode, the goal moves. This is like
# a gridworld where the goal cell jumps halfway through.
CA_SPS = [0.85, 0.90]
T_SPS = [steady_state(c)[0] for c in CA_SPS]     # ~327.3 K and ~321.7 K
TC_SS = [steady_state(c)[1] for c in CA_SPS]     # ~301.4 K and ~298.2 K (only used by the baseline)

# --------------------------------------------------------------------------
# Reward (what "good" means to the agent)
# --------------------------------------------------------------------------
# The gridworld gives -1 per move, so the agent learns to reach the goal fast.
# Here we give minus the squared distance from setpoint at EVERY step. So the
# agent is rewarded for getting close quickly AND for staying there.
#
# Maximising the total reward over an episode is the same as minimising the
# summed squared error: the integral-squared-error (ISE) criterion from
# classical control. RL just phrases it as a reward.
#
# C_A and T have very different units and magnitudes (0.01 mol/L vs 1 K), so
# each error is divided by a scale. With these values, 0.01 mol/L of C_A error
# "costs" the same as 1 K of T error. Changing them changes which output the
# agent cares about more: that is your design choice as the engineer.
# (In pc-gym the same thing is r_scale = {'Ca': 1e4, 'T': 1}.)
SIGMA_CA, SIGMA_T = 0.01, 1.0


def reward(ca, T, sp):
    """Reward for being at (ca, T) when the active setpoint is number `sp`.

    Always <= 0, and exactly 0 only when both errors are zero.
    """
    return -(((ca - CA_SPS[sp]) / SIGMA_CA) ** 2 + ((T - T_SPS[sp]) / SIGMA_T) ** 2)


# ==========================================================================
# 2. DISCRETISATION: turning the continuous reactor into a grid
# ==========================================================================
# KEY CONCEPT - tabular Q-learning needs a FINITE set of states and actions,
# because Q is literally a table with one entry per (state, action) pair.
# The gridworld already has 25 cells and 4 moves. The reactor has infinitely
# many (C_A, T) values and infinitely many Tc values, so we have to draw the
# grid ourselves.
#
# Choice 1 - measure the state as ERROR from setpoint (C_A - C_A,sp, T - T_sp).
#   Then the goal is always the middle cell of the grid, wherever the setpoint is.
#
# Choice 2 - make the bins NARROW near the setpoint and WIDE far away.
#   Near the goal we need precision to hold the reactor still. Far away we
#   only need to know roughly which direction to push.
#
# Trade-off: more bins means a finer controller, but more cells to visit and
# learn, so training is slower (the "curse of dimensionality"). Fewer bins
# means faster learning, but different reactor states get lumped into one cell
# and treated identically (called "state aliasing"). This limitation is what
# function approximation (DQN, Gaussian-process Q-learning) is designed to remove.
# --------------------------------------------------------------------------
# Bin EDGES. 10 edges create 11 bins: below the first edge, between each
# pair of edges, and above the last edge.
CA_EDGES = np.array([-0.06, -0.03, -0.015, -0.007, -0.002, 0.002, 0.007, 0.015, 0.03, 0.06])
T_EDGES = np.array([-6.0, -3.0, -1.5, -0.7, -0.2, 0.2, 0.7, 1.5, 3.0, 6.0])
N_CA, N_T = len(CA_EDGES) + 1, len(T_EDGES) + 1  # 11 x 11 grid per setpoint (5 bins each would be the 5x5 grid)

# Actions = a menu of cooling-water temperatures (pc-gym bounds are 295-302 K).
# This plays the role of {up, down, left, right}. Note there is no exact
# "stay put" move: holding the reactor at setpoint needs Tc = 301.44 K, which
# is not on the menu. So the agent flips between 301.0 and 301.5 near the
# goal, which shows up as small "chatter" in the plots.
TC_LEVELS = np.linspace(295.0, 302.0, 15)         # 15 levels, 0.5 K apart
N_A = len(TC_LEVELS)


def to_cell(ca, T, sp):
    """Map a continuous reactor state to its grid cell (like reading off (row, col)).

    Returns (sp, i, j):
      sp - which setpoint is active. It is part of the state because the right
           action depends on WHERE the goal is (pc-gym also puts the setpoint
           in the observation).
      i  - C_A-error bin (the "row")
      j  - T-error bin   (the "column")
    np.searchsorted finds which pair of edges the error falls between.
    """
    i = int(np.searchsorted(CA_EDGES, ca - CA_SPS[sp]))
    j = int(np.searchsorted(T_EDGES, T - T_SPS[sp]))
    return sp, i, j



# ==========================================================================
# 3. Q-LEARNING: exactly the gridworld algorithm
# ==========================================================================
# Q[s][a] estimates: "if I am in cell s, apply Tc level a now, and then act
# as well as I can afterwards, what total (discounted) reward will I collect?"
# The update rule moves each estimate toward a better guess:
#
#   Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a')  -  Q(s,a) ]
#                                 \_____ TD target ________/
#                               \__________ TD error _____________________/
#
# Read the TD target as: "the reward I actually got, plus the value of the
# best thing I can do from where I landed". If that is better than I
# expected, Q(s,a) goes up; if worse, it goes down.
# --------------------------------------------------------------------------

# alpha (learning rate): how far each update moves Q toward the TD target.
# Here it shrinks with how often (cell, action) has been tried: 1/n^0.6.
# Bins lump together slightly different reactor states, so the same (cell,
# action) can give slightly different outcomes. A shrinking step size
# AVERAGES those outcomes instead of chasing the latest one. (A constant 0.1
# also works, just noisier.) ALPHA_MIN stops it shrinking all the way to 0.
ALPHA_MIN = 0.01

# gamma (discount factor): how much future reward counts relative to now.
# Rule of thumb: the agent "looks ahead" about 1/(1 - gamma) = 20 steps
# (= 5 minutes). That needs to be longer than the reactor takes to respond
# (~2-3 min), or the agent cannot see the benefit of an action it takes now.
GAMMA = 0.95

EPISODES = 4000          # number of 25-minute reactor runs to learn from

# epsilon (exploration rate) for the epsilon-greedy policy:
#   with probability epsilon -> try a RANDOM Tc   (explore: find out what happens)
#   otherwise                -> use the best Tc so far (exploit: use what you know)
# Starts at 1.0 (pure exploration) and falls linearly to 0.02 over the first
# 80% of training, so the agent explores widely early on and refines later.
EPS_START, EPS_END = 1.0, 0.02

# The Q-table: one number per (setpoint, C_A bin, T bin, Tc level)
#   = 2 x 11 x 11 x 15 = 3,630 entries (vs 25 x 4 = 100 in the 5x5 gridworld).
# Starting at 0 is OPTIMISTIC, because every real reward is negative. Any
# action that has not been tried yet looks better than ones that have, which
# nudges the agent to try everything at least once.
Q = np.zeros((len(CA_SPS), N_CA, N_T, N_A))
VISITS = np.zeros_like(Q)       # how often each (cell, action) has been tried (used for alpha and for plotting)


def setpoint_schedule(first):
    """Which setpoint is active at each step: switches halfway, as in pc-gym."""
    return [first] * (N_STEPS // 2) + [1 - first] * (N_STEPS // 2)


def run_training():
    """Learn the Q-table by trial and error over many simulated episodes."""
    returns = []                                  # total reward per episode, to draw the learning curve
    for ep in range(EPISODES):
        # Exploration schedule: decay epsilon as training progresses.
        eps = max(EPS_END, EPS_START * (1 - ep / (0.8 * EPISODES)))

        # Start each episode from a RANDOM reactor state ("exploring starts").
        # If we always started from the same point, the agent would only
        # learn about the cells on its usual path and know nothing elsewhere.
        ca = rng.uniform(0.75, 0.95)
        T = rng.uniform(315.0, 335.0)
        # Randomly choose whether the setpoint goes 0.85 -> 0.90 or 0.90 -> 0.85,
        # so both goals (and both kinds of jump) get practised.
        sched = setpoint_schedule(int(rng.integers(2)))

        # 1. OBSERVE: which grid cell are we in?
        s = to_cell(ca, T, sched[0])
        G = 0.0                                   # running total reward this episode
        for t in range(N_STEPS):
            # 2. CHOOSE an action (epsilon-greedy). Q[s] is the row of 15
            #    action values for this cell, the way a gridworld cell has 4
            #    arrow values. argmax picks the best one.
            a = int(rng.integers(N_A)) if rng.random() < eps else int(np.argmax(Q[s]))

            # 3. ACT: apply that Tc to the reactor for one interval.
            ca, T = step_reactor(ca, T, TC_LEVELS[a])

            # 4. REWARD: how close is the reactor to the setpoint we were
            #    aiming for during this step?
            r = reward(ca, T, sched[t])

            # 5. OBSERVE THE NEXT STATE: which cell did we land in?
            #    (It uses the NEXT step's setpoint, because that is the goal
            #    the agent will be chasing from there.)
            sp_next = sched[min(t + 1, N_STEPS - 1)]
            s_next = to_cell(ca, T, sp_next)

            # 6. LEARN: the Q-learning update.
            #    TD target = reward now + discounted value of the best action
            #    from the next cell. Unlike the gridworld there is no terminal
            #    "goal reached" state where the future is worth 0. The episode
            #    only stops because we ran out of time (a time limit, not a
            #    real ending), so we ALWAYS include the future term.
            td_target = r + GAMMA * np.max(Q[s_next])
            VISITS[s][a] += 1
            alpha = max(ALPHA_MIN, VISITS[s][a] ** -0.6)
            Q[s][a] += alpha * (td_target - Q[s][a])   # nudge Q toward the target by the TD error

            # Move on: the next cell becomes the current cell.
            s = s_next
            G += r
        returns.append(G)
        if (ep + 1) % 500 == 0:
            print(f"episode {ep + 1:5d}  eps={eps:.2f}  mean return (last 500) = {np.mean(returns[-500:]):9.1f}")
    return np.array(returns)


# ==========================================================================
# 4. TESTING: run a policy on the reactor WITHOUT learning or exploring
# ==========================================================================
# After training we "freeze" the Q-table and see how the controller behaves.
# A policy here is just a function: (C_A, T, setpoint) -> Tc.
# --------------------------------------------------------------------------
def rollout(policy, ca0=0.8, T0=330.0, first_sp=0, Ti_plant=350.0):
    """Run one 25-minute closed-loop test from pc-gym's default start x0 = [0.8, 330].

    Ti_plant lets the REAL feed temperature differ from the 350 K the agent was
    trained on. This is an unmeasured disturbance (plant-model mismatch): the
    agent cannot see it in its state. It is a simple test of ROBUSTNESS, which
    is the theme of your project.
    """
    # Temporarily swap the feed temperature used by f() for this test only,
    # then restore it afterwards so training conditions are unchanged.
    global Ti
    Ti_nominal, Ti = Ti, Ti_plant
    ca, T = ca0, T0
    sched = setpoint_schedule(first_sp)
    log = {"t": [0.0], "ca": [ca], "T": [T], "Tc": [], "r": []}
    for t in range(N_STEPS):
        Tc = policy(ca, T, sched[t])              # controller decides
        ca, T = step_reactor(ca, T, Tc)           # plant responds
        log["t"].append((t + 1) * DT)
        log["ca"].append(ca)
        log["T"].append(T)
        log["Tc"].append(Tc)
        log["r"].append(reward(ca, T, sched[t]))
    Ti = Ti_nominal
    log["sched"] = sched
    return {k: np.array(v) for k, v in log.items()}


def greedy_policy(ca, T, sp):
    """THE LEARNED CONTROLLER: find the current cell and apply its best Tc.

    This is the "follow the arrows" step in the gridworld. It is FEEDBACK
    control: the action depends on the measured state, so if a disturbance
    pushes the reactor off target, the controller sees it and reacts.
    """
    return TC_LEVELS[int(np.argmax(Q[to_cell(ca, T, sp)]))]


def steady_state_policy(ca, T, sp):
    """BASELINE: ignore the measurements and apply the textbook steady-state Tc.

    This is OPEN-LOOP (no feedback). With a perfect model and no disturbances
    it eventually settles exactly on target. But it is slow to get there, and
    if the real plant differs from the model it settles in the wrong place
    and never corrects.
    """
    return TC_SS[sp]


def bin_labels(edges, fmt):
    """Axis labels for the policy-grid plot: the centre value of each bin."""
    mids = [fmt.format((a + b) / 2) for a, b in zip(edges[:-1], edges[1:])]
    return ["<" + fmt.format(edges[0])] + mids + [">" + fmt.format(edges[-1])]


# ==========================================================================
# 5. MAIN SCRIPT: train -> test -> compare -> plot
# ==========================================================================
if __name__ == "__main__":
    # Show the consistent setpoint pairs (see steady_state() for why they matter).
    for c, Tsp, Tc in zip(CA_SPS, T_SPS, TC_SS):
        print(f"C_A sp = {c:.2f} mol/L  ->  consistent T sp = {Tsp:.2f} K, steady-state Tc = {Tc:.2f} K")

    # ---- TRAIN: fill in the Q-table by trial and error ----
    returns = run_training()

    # ---- TEST: compare the learned feedback policy against the open-loop baseline ----
    # Test 1: the plant exactly matches what the agent trained on.
    # Test 2: the real feed is 2 K warmer than the model (and agent) assumed.
    #         The agent was NEVER trained on this, so this tests robustness.
    tests = {
        "nominal plant": dict(),
        "feed 2 K warmer than modelled (Ti = 352 K)": dict(Ti_plant=352.0),
    }
    logs = {}
    print()
    for name, kw in tests.items():
        ql = rollout(greedy_policy, **kw)
        ss = rollout(steady_state_policy, **kw)
        logs[name] = (ql, ss)
        # Episode return = total reward = minus the summed squared error. Closer to 0 is better.
        print(f"[{name}]  episode return  Q-learning {ql['r'].sum():8.1f}   |   constant steady-state Tc {ss['r'].sum():8.1f}")
        # Average error once things have settled (minutes 7.5-12.5, before the setpoint
        # changes). This shows the long-run offset separately from the speed of response.
        for lab, lg in [("Q-learning", ql), ("steady-state Tc", ss)]:
            e_ca = np.abs(lg["ca"][31:51] - CA_SPS[0]).mean()
            e_T = np.abs(lg["T"][31:51] - T_SPS[0]).mean()
            print(f"    {lab:16s} mean |error|, minutes 7.5-12.5:  C_A {e_ca:.4f} mol/L   T {e_T:.2f} K")

    # Test 3: one starting point can be lucky or unlucky, so average over
    # several different start states and both setpoint orders.
    starts = [(0.8, 330, 0), (0.8, 330, 1), (0.9, 320, 0), (0.75, 335, 1), (0.95, 318, 0)]
    q_avg = np.mean([rollout(greedy_policy, c, T, f)["r"].sum() for c, T, f in starts])
    s_avg = np.mean([rollout(steady_state_policy, c, T, f)["r"].sum() for c, T, f in starts])
    print(f"\nMean return over {len(starts)} start states  Q-learning {q_avg:8.1f}   |   constant steady-state Tc {s_avg:8.1f}")




    # ------------------------------------------------------------------
    # Figure 1: closed-loop responses (left = nominal plant, right = disturbed plant)
    # What to look for:
    #   - Q-learning reaches the setpoint faster (it uses aggressive Tc early on).
    #   - Near the setpoint it chatters slightly (the exact Tc it needs isn't on the menu).
    #   - With the disturbance, the open-loop baseline settles far from target,
    #     while the feedback policy stays much closer.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for col, (name, (ql, ss)) in enumerate(logs.items()):
        tt = ql["t"][1:]
        sp_ca = [CA_SPS[k] for k in ql["sched"]]
        sp_T = [T_SPS[k] for k in ql["sched"]]
        ax[0, col].plot(ql["t"], ql["ca"], label="Q-learning policy")
        ax[0, col].plot(ss["t"], ss["ca"], "--", alpha=0.8, label="constant steady-state Tc (from model)")
        ax[0, col].step(tt, sp_ca, "k:", where="pre", label="setpoint")
        ax[1, col].plot(ql["t"], ql["T"])
        ax[1, col].plot(ss["t"], ss["T"], "--", alpha=0.8)
        ax[1, col].step(tt, sp_T, "k:", where="pre")
        ax[2, col].step(tt, ql["Tc"], where="pre")
        ax[2, col].step(tt, ss["Tc"], "--", where="pre", alpha=0.8)
        ax[0, col].set_title(name)
        ax[2, col].set_xlabel("time (min)")
    ax[0, 0].set_ylabel("C_A (mol/L)")
    ax[1, 0].set_ylabel("T (K)")
    ax[2, 0].set_ylabel("Tc = action (K)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Tabular Q-learning on the CSTR: one input (Tc) tracking two outputs (C_A and T)")
    fig.tight_layout()
    fig.savefig("cstr_qlearning_response.png", dpi=130)

    # ------------------------------------------------------------------
    # Figure 2: the learned policy drawn as a grid (the gridworld "arrow map")
    # Each cell shows the Tc the agent would choose there (argmax of Q).
    # What to look for:
    #   - Reactor too COLD (left side) -> high Tc (red): warm it up.
    #   - Reactor too HOT (right side) -> low Tc (blue): cool it down.
    #   - Grey cells were rarely/never visited. C_A and T are physically
    #     coupled, so many error combinations simply can't happen
    #     (the CSTR's equivalent of walls).
    #   - The learning curve rises as the agent improves, then flattens.
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ca_lab = bin_labels(CA_EDGES, "{:.3f}")
    T_lab = bin_labels(T_EDGES, "{:.1f}")
    for sp in range(2):
        pol = TC_LEVELS[np.argmax(Q[sp], axis=-1)]           # best Tc in every cell
        explored = VISITS[sp].sum(axis=-1) >= 30              # only trust cells visited enough
        im = axs[sp].imshow(np.where(explored, pol, np.nan), cmap="coolwarm", origin="lower", vmin=295, vmax=302)
        axs[sp].imshow(np.where(explored, np.nan, 1.0), cmap="Greys", vmin=0, vmax=4, origin="lower")
        for i in range(N_CA):
            for j in range(N_T):
                if explored[i, j]:
                    axs[sp].text(j, i, f"{pol[i, j]:.1f}", ha="center", va="center", fontsize=7)
        mid = len(CA_EDGES) // 2                              # index of the centre (goal) cell
        axs[sp].add_patch(plt.Rectangle((mid - 0.5, mid - 0.5), 1, 1, fill=False, lw=2.5, ec="k"))
        axs[sp].set_xticks(range(N_T), T_lab, fontsize=7, rotation=45)
        axs[sp].set_yticks(range(N_CA), ca_lab, fontsize=7)
        axs[sp].set_xlabel("T - T_sp (K)")
        axs[sp].set_ylabel("C_A - C_A,sp (mol/L)")
        axs[sp].set_title(f"Greedy Tc in each cell, C_A sp = {CA_SPS[sp]}\n(goal cell boxed, grey = rarely/never reached)", fontsize=10)
    fig.colorbar(im, ax=axs[:2], label="Tc chosen (K)", shrink=0.8)
    w = 100                                                   # smooth the noisy per-episode returns
    axs[2].plot(np.convolve(returns, np.ones(w) / w, mode="valid"))
    axs[2].set_xlabel("training episode")
    axs[2].set_ylabel(f"episode return ({w}-episode moving average)")
    axs[2].set_title("Learning curve", fontsize=10)
    fig.savefig("cstr_qlearning_policy_grid.png", dpi=130, bbox_inches="tight")
    print("\nSaved cstr_qlearning_response.png and cstr_qlearning_policy_grid.png")
