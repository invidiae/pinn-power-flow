import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import scipy.sparse
from tqdm import tqdm

np.random.seed(42)


net = pn.case14()

baseline_load_p_mw = net.load.p_mw.copy()
baseline_load_q_mvar = net.load.q_mvar.copy()
net.gen.p_mw = np.random.uniform(10, 40, size=len(net.gen))
baseline_gen_p_mw = net.gen.p_mw.copy()


def generate_data(n_samples=1000):
    results = []

    for i in tqdm(range(n_samples)):
        # Randomize loads
        load_factor = np.random.uniform(0, 5, size=len(net.load))
        net.load.p_mw = baseline_load_p_mw * load_factor
        net.load.q_mvar = baseline_load_q_mvar * load_factor

        # Randomize generators
        gen_factor = np.random.uniform(0, 3, size=len(net.gen))
        net.gen.p_mw = baseline_gen_p_mw * gen_factor

        try:
            pp.runpp(net, numba=True)

            # Input X: Load P and Q, Generator P
            inputs = np.concatenate([
                net.load[["p_mw", "q_mvar"]].values.flatten(),
                net.gen["p_mw"].values.flatten(),
            ])

            # Target Y: Voltage Mag (pu) and Angle (degrees)
            v_mag = net.res_bus.vm_pu.values
            v_ang = net.res_bus.va_degree.values

            # Net power injections per bus (MW/Mvar), positive = generation
            p_bus = net.res_bus.p_mw.values
            q_bus = net.res_bus.q_mvar.values

            results.append(np.concatenate([inputs, v_mag, v_ang, p_bus, q_bus]))
        except:
            # Power flow might fail to converge if randomization is too wild
            print("no convergence for sample")
            continue

    load_ids = net.load.index.tolist()
    gen_ids = net.gen.index.tolist()
    bus_ids = net.bus.index.tolist()
    columns = (
        [f"load_{i}_p_mw" for i in load_ids]
        + [f"load_{i}_q_mvar" for i in load_ids]
        + [f"gen_{i}_p_mw" for i in gen_ids]
        + [f"bus_{i}_vm_pu" for i in bus_ids]
        + [f"bus_{i}_va_degree" for i in bus_ids]
        + [f"bus_{i}_p_mw" for i in bus_ids]
        + [f"bus_{i}_q_mvar" for i in bus_ids]
    )
    return pd.DataFrame(results, columns=columns)


## Run once to extract Ybus
pp.runpp(net, numba=True)
Ybus = net._ppc["internal"]["Ybus"]
scipy.sparse.save_npz("data/Ybus.npz", Ybus)

df = generate_data(200)
df.to_csv("data/ieee_14.csv", index=False)