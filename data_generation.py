import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
from tqdm import tqdm

net = pn.case14()
baseline_p_mw = net.load.p_mw.copy()
baseline_q_mvar = net.load.q_mvar.copy()


def generate_data(n_samples=1000):
    results = []

    for i in tqdm(range(n_samples)):
        # 2. Randomize the loads (Physics: Variation in Demand)
        # We multiply the baseline loads by a random factor (0.8 to 1.2)
        random_factor = np.random.uniform(0.8, 1.2, size=len(net.load))
        net.load.p_mw = baseline_p_mw * random_factor
        net.load.q_mvar = baseline_q_mvar * random_factor

        # 3. Run a Power Flow
        try:
            pp.runpp(net, numba=True)

            # 4. Extract what your PINN needs
            # Input X: Load P and Q
            inputs = net.load[["p_mw", "q_mvar"]].values.flatten()

            # Target Y: Voltage Mag (pu) and Angle (degrees)
            v_mag = net.res_bus.vm_pu.values
            v_ang = net.res_bus.va_degree.values

            results.append(np.concatenate([inputs, v_mag, v_ang]))
        except:
            # Power flow might fail to converge if randomization is too wild
            continue

    load_ids = net.load.index.tolist()
    bus_ids = net.bus.index.tolist()
    columns = (
        [f"load_{i}_p_mw" for i in load_ids]
        + [f"load_{i}_q_mvar" for i in load_ids]
        + [f"bus_{i}_vm_pu" for i in bus_ids]
        + [f"bus_{i}_va_degree" for i in bus_ids]
    )
    return pd.DataFrame(results, columns=columns)


df = generate_data(20000)
df.to_csv("data/ieee_14.csv", index=False)
