import json

import matplotlib.pyplot as plt
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
            inputs = net.load[['p_mw', 'q_mvar']].values.flatten()
            
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
        [f"load_{i}_p_mw" for i in load_ids] +
        [f"load_{i}_q_mvar" for i in load_ids] +
        [f"bus_{i}_vm_pu" for i in bus_ids] +
        [f"bus_{i}_va_degree" for i in bus_ids]
    )
    return pd.DataFrame(results, columns=columns)

df = generate_data(1000)
df.to_csv("data/ieee_14.csv", index=False)

# Visualization of the network
# Run one power flow for visualization
pp.runpp(net, numba=True)
coords = net.bus["geo"].apply(lambda g: json.loads(g)["coordinates"])
geo = pd.DataFrame(coords.tolist(), index=net.bus.index, columns=["x", "y"])

fig, ax = plt.subplots(figsize=(10, 7))

# Draw lines
for _, line in net.line.iterrows():
    x = [geo.loc[line.from_bus, "x"], geo.loc[line.to_bus, "x"]]
    y = [geo.loc[line.from_bus, "y"], geo.loc[line.to_bus, "y"]]
    ax.plot(x, y, "k-", linewidth=1, zorder=1)

# Draw transformers
for _, trafo in net.trafo.iterrows():
    x = [geo.loc[trafo.hv_bus, "x"], geo.loc[trafo.lv_bus, "x"]]
    y = [geo.loc[trafo.hv_bus, "y"], geo.loc[trafo.lv_bus, "y"]]
    ax.plot(x, y, "b--", linewidth=1.5, zorder=1)

# Draw buses coloured by voltage magnitude
vm = net.res_bus.vm_pu
sc = ax.scatter(geo["x"], geo["y"], c=vm, cmap="RdYlGn", vmin=0.95, vmax=1.05,
                s=200, zorder=2)
plt.colorbar(sc, ax=ax, label="Voltage magnitude (pu)")

for bus_id, row in geo.iterrows():
    ax.annotate(str(bus_id), (row["x"], row["y"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)

ax.set_title("IEEE 14-Bus Network")
ax.axis("off")
plt.tight_layout()
plt.savefig("data/ieee_14_bus.png", dpi=150)
plt.show()