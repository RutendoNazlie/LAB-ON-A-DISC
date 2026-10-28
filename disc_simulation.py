"""
MALARIA LAB-ON-A-DISC: FLUID-MOVEMENT SIMULATION
================================================
Semi-quantitative optical analyser for malaria test intensity
(centrifugal stratification + acridine orange fluorescence + radial scan)
 
What this program does
----------------------
1. Calculates the design numbers from engineering formulas:
   - burst speed of each capillary burst valve
   - relative centrifugal force (RCF) in the separation chamber
   - where the plasma, buffy coat and red-cell layers settle
2. Animates one complete test run in the ROTATING FRAME of the disc
   (as if a camera were spinning with the disc), showing:
   - blood moving from chamber to chamber as each valve opens
   - shake-mode mixing with the dried acridine orange stain
   - cells separating into layers by density
   - the optical head scanning the chamber and building the
     fluorescence profile, then reporting a density level
3. Saves:
   - disc_layout.png          labelled design drawing
   - spin_protocol.png        rotation speed for each step
   - disc_simulation.gif      the animation
   - design_calculations.txt  all calculated values
 
IMPORTANT: this is a simplified engineering model for teaching and
presentation. Fluid motion uses lumped (chamber-by-chamber) volumes and
Stokes-law particle settling, not a full CFD solution. The fluorescence
signal is modelled, not measured. It does not diagnose malaria.
 
All lengths are in millimetres unless stated otherwise.
Change the numbers in the SETTINGS section and run again.
"""
 
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # works in GitHub Codespaces
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
 
# ======================================================================
# SETTINGS  (safe to change)
# ======================================================================
 
DISC_RADIUS = 70.0            # mm
CENTRE_HOLE_DIAMETER = 6.0    # mm
CHAMBER_DEPTH = 1.0           # mm (laser-cut PMMA middle layer)
VALVE_DEPTH = 0.05            # mm (thin adhesive layer that forms valve channels)
 
SAMPLE_LOADED_UL = 25.0       # blood placed in the loading well (µL)
METERED_UL = 20.0             # metering chamber volume (µL), set by geometry
HAEMATOCRIT = 0.40            # red-cell volume fraction of the patient's blood
BUFFY_FRACTION = 0.01         # leukocyte + platelet volume fraction
WBC_PER_UL = 7000             # leukocytes per µL of blood
PARASITE_DENSITY = 20000      # parasites per µL (try 500, 5000, 20000, 150000)
 
# Blood properties used in the valve and settling calculations
BLOOD_DENSITY = 1060.0        # kg/m^3
SURFACE_TENSION = 0.055       # N/m (blood)
VALVE_CONTACT_ANGLE = 110.0   # degrees, hydrophobic patch on the valve
PLASMA_VISCOSITY = 1.2e-3     # Pa·s
 
SEPARATION_RPM = 5000
FPS = 10
GIF_NAME = "disc_simulation.gif"
 
# ======================================================================
# DISC GEOMETRY  (rotating frame: x = across the disc, y = radius)
# One test unit sits on the top spoke of the disc (angle 90 degrees).
# Larger y means further from the centre.
# ======================================================================
 
# rectangles: (x_left, x_right, y_inner, y_outer)
CH = {
    "Loading well":        dict(kind="circle", cx=0.0, cy=16.0, r=3.0),
    "Metering chamber":    dict(kind="rect", box=(-2.0, 2.0, 22.0, 27.0)),
    "Overflow (waste)":    dict(kind="rect", box=(7.5, 13.0, 22.0, 29.0)),
    "Mixing chamber":      dict(kind="rect", box=(-7.0, 7.0, 30.0, 35.0)),
    "Separation chamber (wide)":   dict(kind="rect", box=(-1.5, 1.5, 46.0, 53.0)),
    "Separation chamber (narrow)": dict(kind="rect", box=(-0.5, 0.5, 53.0, 64.0)),
    "Blank chamber":       dict(kind="rect", box=(-12.0, -9.0, 53.0, 64.0)),
    "Standard chamber":    dict(kind="rect", box=(9.0, 12.0, 53.0, 64.0)),
}
 
COLOURS = {
    "Loading well": "#F4A261", "Metering chamber": "#E9C46A",
    "Overflow (waste)": "#C9C9C9", "Mixing chamber": "#2A9D8F",
    "Separation chamber (wide)": "#457B9D", "Separation chamber (narrow)": "#457B9D",
    "Blank chamber": "#A8DADC", "Standard chamber": "#B794F4",
}
 
# Valve channels: width (mm) and the liquid column pushing on them
VALVES = {
    "V1": dict(label="V1 capillary burst valve\n(metering outlet)",
               width=0.40, y=27.0, column_rmean=24.5, column_dr=5.0),
    "V2": dict(label="V2 capillary burst valve\n(mixing outlet)",
               width=0.15, y=35.0, column_rmean=None, column_dr=None),
}
# V2 liquid column is calculated from the volume sitting in the mixing chamber.
 
_trapz = getattr(np, "trapezoid", None) or np.trapz   # works with old and new NumPy
 
BLOOD_RED = "#B3202A"
PLASMA = "#F2D16B"
STAINED = "#D9822B"
 
# ======================================================================
# ENGINEERING CALCULATIONS
# ======================================================================
 
def rpm_to_omega(rpm):
    return rpm * 2 * np.pi / 60.0
 
 
def rcf(radius_mm, rpm):
    """Relative centrifugal force: RCF = 1.118e-5 * r(cm) * rpm^2"""
    return 1.118e-5 * (radius_mm / 10.0) * rpm ** 2
 
 
def capillary_barrier_pa(width_mm, depth_mm, theta_deg=VALVE_CONTACT_ANGLE,
                         gamma=SURFACE_TENSION):
    """Pressure a hydrophobic rectangular channel can hold back:
       P = -2*gamma*cos(theta)*(1/w + 1/d)"""
    w = width_mm / 1000.0
    d = depth_mm / 1000.0
    return -2 * gamma * np.cos(np.deg2rad(theta_deg)) * (1 / w + 1 / d)
 
 
def burst_rpm(width_mm, rmean_mm, dr_mm):
    """Speed at which centrifugal pressure rho*w^2*rmean*dr exceeds the barrier."""
    p_cap = capillary_barrier_pa(width_mm, VALVE_DEPTH)
    omega2 = p_cap / (BLOOD_DENSITY * (rmean_mm / 1000.0) * (dr_mm / 1000.0))
    return np.sqrt(omega2) * 60 / (2 * np.pi), p_cap
 
 
def rect_width(name):
    x0, x1, _, _ = CH[name]["box"]
    return x1 - x0
 
 
# --- V2 liquid column: metered blood lying against the outer wall of the mixing chamber
mix_x0, mix_x1, mix_y0, mix_y1 = CH["Mixing chamber"]["box"]
mix_height = METERED_UL / ((mix_x1 - mix_x0) * CHAMBER_DEPTH)       # mm of liquid
VALVES["V2"]["column_dr"] = mix_height
VALVES["V2"]["column_rmean"] = mix_y1 - mix_height / 2
 
for v in VALVES.values():
    v["burst_rpm"], v["p_cap"] = burst_rpm(v["width"], v["column_rmean"], v["column_dr"])
 
V1_BURST = VALVES["V1"]["burst_rpm"]
V2_BURST = VALVES["V2"]["burst_rpm"]
 
# Spin protocol speeds chosen from the burst speeds
METERING_RPM = int(round(0.5 * V1_BURST, -1))
TRANSFER1_RPM = int(round(0.5 * (V1_BURST + V2_BURST), -1))
SHAKE_LOW, SHAKE_HIGH = int(round(0.3 * V1_BURST, -1)), int(round(0.85 * V2_BURST, -1))
TRANSFER2_RPM = int(round(1.5 * V2_BURST, -1))
 
 
# --- Where the layers settle in the separation chamber (fill from the rim inward)
def separation_area_per_mm(y):
    """Cross-sectional area (mm^2) of the separation chamber at radius y."""
    if 53.0 <= y <= 64.0:
        return rect_width("Separation chamber (narrow)") * CHAMBER_DEPTH
    if 46.0 <= y < 53.0:
        return rect_width("Separation chamber (wide)") * CHAMBER_DEPTH
    return 0.0
 
 
def radius_holding_volume(vol_ul, y_outer=64.0):
    """Walk inward from the rim until vol_ul has been accommodated."""
    y, filled, step = y_outer, 0.0, 0.001
    while filled < vol_ul and y > 46.0:
        filled += separation_area_per_mm(y) * step
        y -= step
    return y
 
 
def layer_positions(hct=HAEMATOCRIT):
    rbc = METERED_UL * hct
    buffy = METERED_UL * BUFFY_FRACTION
    y_rbc_top = radius_holding_volume(rbc)
    y_buffy_top = radius_holding_volume(rbc + buffy)
    y_liquid_top = radius_holding_volume(METERED_UL)
    return y_liquid_top, y_buffy_top, y_rbc_top
 
 
Y_TOP, Y_BUFFY, Y_RBC = layer_positions()
 
# Separation force and Stokes settling speeds at the middle of the narrow section
R_SEP = 58.0
RCF_SEP = rcf(R_SEP, SEPARATION_RPM)
_acc = rpm_to_omega(SEPARATION_RPM) ** 2 * (R_SEP / 1000.0)
 
 
def stokes_mm_s(diam_um, cell_density):
    a = diam_um / 2 * 1e-6
    v = 2 * a ** 2 * (cell_density - 1025.0) * _acc / (9 * PLASMA_VISCOSITY)
    return v * 1000.0
 
 
V_RBC = stokes_mm_s(5.5, 1095.0)     # volume-equivalent sphere of a red cell
V_WBC = stokes_mm_s(8.0, 1070.0)
 
 
def write_calculations():
    lines = []
    lines.append("MALARIA LAB-ON-A-DISC - DESIGN CALCULATIONS (simplified model)\n")
    lines.append(f"Metered blood volume: {METERED_UL:.0f} uL, haematocrit {HAEMATOCRIT:.2f}\n")
    lines.append("CAPILLARY BURST VALVES  P_cap = -2*gamma*cos(theta)*(1/w + 1/d)")
    lines.append("                        burst when rho*omega^2*r_mean*dr > P_cap")
    for k, v in VALVES.items():
        lines.append(f"  {k}: width {v['width']:.2f} mm x depth {VALVE_DEPTH:.2f} mm, "
                     f"barrier {v['p_cap']:.0f} Pa, column r_mean {v['column_rmean']:.1f} mm "
                     f"dr {v['column_dr']:.2f} mm  ->  burst ~ {v['burst_rpm']:.0f} rpm")
    lines.append("")
    lines.append("SPIN PROTOCOL (derived from the burst speeds)")
    lines.append(f"  Metering            {METERING_RPM} rpm  (below V1, so V1 holds)")
    lines.append(f"  Transfer 1          {TRANSFER1_RPM} rpm  (above V1, below V2)")
    lines.append(f"  Shake-mode mixing   {SHAKE_LOW}-{SHAKE_HIGH} rpm  (below V2)")
    lines.append(f"  Transfer 2          {TRANSFER2_RPM} rpm  (above V2)")
    lines.append(f"  Separation          {SEPARATION_RPM} rpm")
    lines.append("")
    lines.append(f"SEPARATION  RCF = 1.118e-5 * r(cm) * rpm^2 = {RCF_SEP:.0f} g at r = {R_SEP:.0f} mm")
    lines.append(f"  Stokes speed (single cell, no crowding): RBC {V_RBC:.2f} mm/s, WBC {V_WBC:.2f} mm/s")
    lines.append("  In whole blood, crowding (hindered settling) slows this greatly,")
    lines.append("  which is why several minutes of spinning are needed.")
    lines.append("")
    lines.append("LAYER POSITIONS (radius from disc centre)")
    for h in (0.30, 0.40, 0.50):
        t, b, r = layer_positions(h)
        inside = "inside" if 53.0 <= b <= 64.0 else "OUTSIDE"
        lines.append(f"  Hct {h:.2f}: liquid top {t:.1f} mm, buffy coat {b:.1f}-{r:.1f} mm "
                     f"({inside} the narrow section)")
    text = "\n".join(lines)
    with open("design_calculations.txt", "w") as f:
        f.write(text + "\n")
    print(text)
 
 
# ======================================================================
# SPIN PROTOCOL / TIMELINE
# Each stage: name, real duration (s), animation frames, rpm(progress)
# ======================================================================
 
def ramp(a, b):
    return lambda p: a + (b - a) * p
 
 
STAGES = [
    ("1. Load sample (capillary filling)",  10,  22, lambda p: 0),
    ("2. Metering",                          15,  26, lambda p: METERING_RPM * min(1, 4 * p)),
    ("3. Transfer to mixing (V1 bursts)",     5,  18, lambda p: TRANSFER1_RPM),
    ("4. Shake-mode mixing with stain",      60,  34,
        lambda p: (SHAKE_LOW + SHAKE_HIGH) / 2 + (SHAKE_HIGH - SHAKE_LOW) / 2 * np.sin(2 * np.pi * 6 * p)),
    ("5. Transfer to separation (V2 bursts)", 5,  18, lambda p: TRANSFER2_RPM),
    ("6. Separation (stratification)",      480,  48,
        lambda p: TRANSFER2_RPM + (SEPARATION_RPM - TRANSFER2_RPM) * min(1, 8 * p)),
    ("7. Slow stop (prevents remixing)",     30,  16, lambda p: SEPARATION_RPM * (1 - p)),
    ("8. Radial fluorescence scan",          60,  50, lambda p: 0),
    ("9. Result",                             5,  20, lambda p: 0),
]
 
FRAME_STAGE = []
for si, (_, dur, nfr, _) in enumerate(STAGES):
    for k in range(nfr):
        FRAME_STAGE.append((si, k / max(1, nfr - 1)))
N_FRAMES = len(FRAME_STAGE)
STAGE_START_T = np.cumsum([0] + [s[1] for s in STAGES])
 
 
def frame_info(f):
    si, p = FRAME_STAGE[f]
    rpm = max(0.0, STAGES[si][3](p))
    t_real = STAGE_START_T[si] + p * STAGES[si][1]
    return si, p, rpm, t_real
 
 
# ======================================================================
# FLUID STATE (lumped volumes per chamber)
# ======================================================================
 
def fluid_state(si, p):
    s = dict(well=0.0, meter=0.0, over=0.0, mix=0.0, sep=0.0,
             stain=0.0, strat=0.0, flow=None)
    excess = SAMPLE_LOADED_UL - METERED_UL
    if si == 0:
        s["well"] = SAMPLE_LOADED_UL * min(1, 1.5 * p)
    elif si == 1:
        moved = SAMPLE_LOADED_UL * p
        s["well"] = SAMPLE_LOADED_UL - moved
        s["meter"] = min(METERED_UL, moved)
        s["over"] = max(0.0, moved - METERED_UL)
        s["flow"] = "well->meter" if moved < METERED_UL else "meter->over"
    elif si == 2:
        s["over"] = excess
        s["meter"] = METERED_UL * (1 - p)
        s["mix"] = METERED_UL * p
        s["flow"] = "meter->mix"
    elif si == 3:
        s["over"], s["mix"], s["stain"] = excess, METERED_UL, p
    elif si == 4:
        s["over"], s["stain"] = excess, 1.0
        s["mix"] = METERED_UL * (1 - p)
        s["sep"] = METERED_UL * p
        s["flow"] = "mix->sep"
    else:
        s["over"], s["stain"], s["sep"] = excess, 1.0, METERED_UL
        if si == 5:
            s["strat"] = p
        elif si >= 6:
            s["strat"] = 1.0
    return s
 
 
# ======================================================================
# PARTICLES IN THE SEPARATION CHAMBER
# ======================================================================
 
rng = np.random.default_rng(7)
N_RBC, N_WBC = 260, 22
parasitaemia = PARASITE_DENSITY / 5.0e6          # fraction of red cells infected
N_IRBC = int(np.clip(round(N_RBC * parasitaemia * 25), 0, 60))   # x25 for visibility
 
 
def chamber_halfwidth(y):
    return 0.5 * (1.5 * 2 if y < 53.0 else 1.0)
 
 
def make_particles():
    kinds = np.array(["rbc"] * N_RBC + ["wbc"] * N_WBC + ["irbc"] * N_IRBC)
    n = len(kinds)
    y0 = rng.uniform(Y_TOP, 64.0, n)
    yt = np.empty(n)
    yt[kinds == "rbc"] = rng.uniform(Y_RBC + 0.1, 63.9, (kinds == "rbc").sum())
    yt[kinds == "wbc"] = rng.uniform(Y_BUFFY, Y_RBC, (kinds == "wbc").sum())
    # infected red cells are slightly lighter: they gather near the top of the red-cell layer
    n_i = (kinds == "irbc").sum()
    yt[kinds == "irbc"] = np.clip(Y_RBC + rng.exponential(1.6, n_i), Y_RBC, 63.9)
    tau = np.where(kinds == "rbc", 0.22, np.where(kinds == "irbc", 0.30, 0.42))
    u = rng.uniform(-0.85, 0.85, n)            # position across the chamber width
    return kinds, y0, yt, tau, u
 
 
P_KIND, P_Y0, P_YT, P_TAU, P_U = make_particles()
 
 
def particle_positions(strat):
    frac = 1 - np.exp(-strat / P_TAU)
    frac = frac / (1 - np.exp(-1 / P_TAU))      # reaches the target exactly at strat = 1
    y = P_Y0 + (P_YT - P_Y0) * np.clip(frac, 0, 1)
    x = np.array([P_U[i] * chamber_halfwidth(y[i]) for i in range(len(y))])
    return x, y
 
 
# ======================================================================
# FLUORESCENCE MODEL FOR THE SCAN
# ======================================================================
 
SCAN_Y = np.linspace(46.0, 64.0, 361)
 
 
def fluorescence_profile(noise=True):
    """Green-channel signal versus radius (arbitrary units).
       One leukocyte ~ 280 parasite-equivalents of DNA (genome size ratio)."""
    wbc_total = WBC_PER_UL * METERED_UL * 280.0
    par_total = PARASITE_DENSITY * METERED_UL
    slit = 0.30                                              # mm aperture
    centre = 0.5 * (Y_BUFFY + Y_RBC)
    width = max(Y_RBC - Y_BUFFY, 0.05)
    wbc = wbc_total / (np.sqrt(2 * np.pi) * (width / 2 + slit / 2)) * \
        np.exp(-0.5 * ((SCAN_Y - centre) / (width / 2 + slit / 2)) ** 2)
    zone = (SCAN_Y >= Y_RBC) & (SCAN_Y <= 64.0)
    dec = np.exp(-(SCAN_Y - Y_RBC) / 1.6) * zone
    par = par_total * dec / (_trapz(dec, SCAN_Y) if dec.sum() else 1)
    background = np.where((SCAN_Y >= Y_TOP) & (SCAN_Y <= 64.0), 4000.0, 800.0)
    sig = background + wbc + par
    if noise:
        sig = sig * (1 + 0.04 * rng.standard_normal(sig.size)) + 150 * rng.standard_normal(sig.size)
    return np.clip(sig, 50, None)
 
 
PROFILE = fluorescence_profile()
BLANK_SIGNAL = 4000.0 * (1 + 0.02 * rng.standard_normal())   # background read in the blank chamber
 
 
def estimate_density(profile):
    """Integrate the net green signal in the red-cell zone, skipping the leukocyte peak.
       The fraction of parasite signal that falls inside the integration window is the
       calibration constant; in the real device it is found from samples of known density."""
    sigma = max(Y_RBC - Y_BUFFY, 0.05) / 2 + 0.15
    start = 0.5 * (Y_BUFFY + Y_RBC) + 5 * sigma          # well clear of the leukocyte peak
    zone = (SCAN_Y >= start) & (SCAN_Y <= 64.0)
    net = np.clip(profile[zone] - BLANK_SIGNAL, 0, None)
    integrated = _trapz(net, SCAN_Y[zone])
    full = (SCAN_Y >= Y_RBC) & (SCAN_Y <= 64.0)
    dec = np.exp(-(SCAN_Y - Y_RBC) / 1.6)
    calib = _trapz(dec[zone], SCAN_Y[zone]) / _trapz(dec[full], SCAN_Y[full])
    return integrated / (calib * METERED_UL), start
 
 
def density_level(d):
    if d < 100:
        return 0, "Negative / below detection"
    if d < 1000:
        return 1, "Low"
    if d < 10000:
        return 2, "Moderate"
    if d < 100000:
        return 3, "High"
    return 4, "Hyperparasitaemia"
 
 
EST_DENSITY, INTEGRATION_START = estimate_density(PROFILE)
LEVEL, LEVEL_TEXT = density_level(EST_DENSITY)
 
# ======================================================================
# DRAWING
# ======================================================================
 
def draw_static_layout(ax, labels=True):
    """Chambers, channels, valves and vents in the rotating frame."""
    ax.add_patch(Rectangle((-15, 12), 30, 54, facecolor="#EEF2F6", edgecolor="none", zorder=0))
    # channels (centre lines)
    ch = dict(color="#6B7C8F", lw=2.2, zorder=1, solid_capstyle="round")
    ax.plot([0, 0], [19, 22], **ch)                       # well -> metering
    ax.plot([2, 7.5], [22.6, 22.6], **ch)                 # metering -> overflow (enters at inner edge)
    ax.plot([0, 0], [27, 30], **ch)                       # V1 -> mixing
    ax.plot([0, 0], [35, 46], **ch)                       # V2 -> separation
    # vents towards the centre
    vent = dict(color="#9AA8B6", lw=0.9, ls=":", zorder=1)
    for x, y in [(-1.6, 22), (12.4, 22), (-6.5, 30), (1.2, 46)]:
        ax.plot([x, x], [y, y - 2.2], **vent)
        ax.add_patch(Circle((x, y - 2.4), 0.35, fc="white", ec="#9AA8B6", lw=0.8, zorder=2))
    # chambers
    for name, c in CH.items():
        col = COLOURS[name]
        if c["kind"] == "circle":
            ax.add_patch(Circle((c["cx"], c["cy"]), c["r"], fc="white", ec="#263238", lw=1.2, zorder=2))
        else:
            x0, x1, y0, y1 = c["box"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white", ec="#263238", lw=1.1, zorder=2))
    # dried stain dots in the mixing chamber
    # valves
    for k, v in VALVES.items():
        ax.add_patch(Rectangle((-0.9, v["y"] - 0.35), 1.8, 0.7, fc="#F4A261", ec="#7A3E00", lw=1.0, zorder=4))
    # no-return (check) points: top-entry inlets
    for (x, y) in [(0, 45.4), (7.0, 22.6)]:
        ax.add_patch(Polygon([[x - 0.8, y - 0.6], [x + 0.8, y - 0.6], [x, y + 0.5]],
                             closed=True, fc="#E63946", ec="#7A0010", lw=0.8, zorder=4))
    if labels:
        t = dict(fontsize=7, zorder=6)
        ax.text(3.8, 16, "Loading well\n(EDTA-coated)", **t, va="center")
        ax.text(-2.6, 24.5, "Metering\n20 µL", **t, ha="right", va="center")
        ax.text(13.4, 25.5, "Overflow /\nwaste", **t, va="center")
        ax.text(7.6, 32.5, "Mixing chamber\n(dried acridine\norange)", **t, va="center")
        ax.text(2.0, 49.5, "Separation /\ndetection\nchamber", **t, va="center")
        ax.text(1.0, 58.5, "narrow zone\n(spreads\nbuffy coat)", fontsize=6.3, zorder=6, va="center")
        ax.text(-10.5, 64.8, "Blank", **t, ha="center")
        ax.text(10.5, 64.8, "Standard", **t, ha="center")
        ax.text(1.2, 27.0, f"V1 (~{V1_BURST:.0f} rpm)", fontsize=6.5, color="#7A3E00", zorder=6, va="center")
        ax.text(1.2, 35.6, f"V2 (~{V2_BURST:.0f} rpm)", fontsize=6.5, color="#7A3E00", zorder=6, va="bottom")
        ax.text(1.0, 44.3, "no-return\n(top entry)", fontsize=6.3, color="#7A0010", zorder=6, va="center")
        ax.annotate("", xy=(-14, 65), xytext=(-14, 14),
                    arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1.3))
        ax.text(-13.6, 40, "centrifugal force\n(outward)", rotation=90, fontsize=7, color="#2B6CB0",
                va="center")
 
 
def fill_rect(ax, box, vol, colour, alpha=0.9, z=3):
    """Liquid pressed against the outer wall: fill from the outer edge inward."""
    x0, x1, y0, y1 = box
    if vol <= 0:
        return
    h = min(y1 - y0, vol / ((x1 - x0) * CHAMBER_DEPTH))
    ax.add_patch(Rectangle((x0, y1 - h), x1 - x0, h, fc=colour, ec="none", alpha=alpha, zorder=z))
 
 
def fill_separation(ax, vol, colour, strat):
    if vol <= 0:
        return
    y_top = radius_holding_volume(vol)
    if strat < 0.02:
        for name in ("Separation chamber (wide)", "Separation chamber (narrow)"):
            x0, x1, y0, y1 = CH[name]["box"]
            lo = max(y0, y_top)
            if lo < y1:
                ax.add_patch(Rectangle((x0, lo), x1 - x0, y1 - lo, fc=colour, ec="none", alpha=0.85, zorder=3))
        return
    # stratified: plasma colour grows from the top as cells move out
    plasma_alpha = min(1, strat * 1.4)
    for name in ("Separation chamber (wide)", "Separation chamber (narrow)"):
        x0, x1, y0, y1 = CH[name]["box"]
        lo = max(y0, y_top)
        if lo < y1:
            ax.add_patch(Rectangle((x0, lo), x1 - x0, y1 - lo, fc=PLASMA, ec="none",
                                   alpha=0.35 + 0.5 * plasma_alpha, zorder=3))
 
 
def draw_flow(ax, flow, p):
    """Moving tracer dots along the channel that is currently carrying liquid."""
    paths = {"well->meter": [(0, 19), (0, 22)],
             "meter->over": [(2, 22.6), (7.5, 22.6)],
             "meter->mix": [(0, 27), (0, 30)],
             "mix->sep": [(0, 35), (0, 46)]}
    if flow not in paths:
        return
    pts = np.array(paths[flow])
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(L)])
    for k in range(6):
        s = ((p * 3 + k / 6) % 1) * cum[-1]
        i = min(np.searchsorted(cum, s) - 1, len(seg) - 1)
        i = max(i, 0)
        f = (s - cum[i]) / L[i]
        x, y = pts[i] + f * seg[i]
        ax.add_patch(Circle((x, y), 0.45, fc=BLOOD_RED, ec="white", lw=0.5, zorder=5))
 
 
def draw_rotating_frame(ax, f):
    si, p, rpm, t_real = frame_info(f)
    s = fluid_state(si, p)
    ax.clear()
    ax.set_xlim(-16, 16)
    ax.set_ylim(67, 11)            # rim at the bottom of the panel, centre at the top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([20, 30, 40, 50, 60])
    ax.set_ylabel("radius from disc centre (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Inside one test unit (rotating-frame view)", fontsize=10)
    draw_static_layout(ax)
 
    blood_col = BLOOD_RED
    stained_col = STAINED if s["stain"] > 0.5 else BLOOD_RED
    # loading well (circle filled by fraction)
    if s["well"] > 0:
        c = CH["Loading well"]
        frac = s["well"] / SAMPLE_LOADED_UL
        ax.add_patch(Circle((c["cx"], c["cy"]), c["r"] * np.sqrt(frac), fc=blood_col, alpha=0.9, zorder=3))
    fill_rect(ax, CH["Metering chamber"]["box"], s["meter"], blood_col)
    fill_rect(ax, CH["Overflow (waste)"]["box"], s["over"], blood_col, alpha=0.6)
    # mixing chamber, with shake-mode slosh
    if s["mix"] > 0:
        x0, x1, y0, y1 = CH["Mixing chamber"]["box"]
        h = s["mix"] / ((x1 - x0) * CHAMBER_DEPTH)
        if si == 3:
            tilt = 0.9 * np.sin(2 * np.pi * 6 * p)
            poly = [[x0, y1], [x1, y1], [x1, y1 - h + tilt], [x0, y1 - h - tilt]]
            mixcol = matplotlib.colors.to_hex(
                (1 - p) * np.array(matplotlib.colors.to_rgb(BLOOD_RED)) + p * np.array(matplotlib.colors.to_rgb(STAINED)))
            ax.add_patch(Polygon(poly, closed=True, fc=mixcol, ec="none", alpha=0.9, zorder=3))
            # swirling particles
            for k in range(14):
                ang = 2 * np.pi * (k / 14 + 2 * p)
                xx = 0.5 * (x0 + x1) + 5.5 * np.cos(ang + k)
                yy = y1 - h / 2 + 0.45 * h * np.sin(2 * ang)
                ax.add_patch(Circle((xx, yy), 0.25, fc="#8B0000", ec="none", zorder=4))
        else:
            fill_rect(ax, (x0, x1, y0, y1), s["mix"], stained_col)
    # undissolved stain
    if s["stain"] < 1:
        x0, x1, y0, y1 = CH["Mixing chamber"]["box"]
        for k in range(10):
            ax.add_patch(Circle((x0 + 1 + 1.3 * k, y0 + 0.6), 0.22 * (1 - s["stain"]),
                                fc="#E07B00", ec="none", zorder=4))
    # separation chamber
    fill_separation(ax, s["sep"], stained_col, s["strat"])
    if s["sep"] > 0 and (si >= 5 or (si == 4 and p > 0.9)):
        x, y = particle_positions(s["strat"])
        glow = si >= 7
        for kind, col, size in (("rbc", "#9E1B1B", 7), ("wbc", "#F7F7F7", 11), ("irbc", "#9E1B1B", 8)):
            m = P_KIND == kind
            ax.scatter(x[m], y[m], s=size, c=col, edgecolors="#555" if kind == "wbc" else "none",
                       linewidths=0.4, zorder=5)
        mi = P_KIND == "irbc"
        ax.scatter(x[mi], y[mi], s=6 if not glow else 14, c="#39FF14", zorder=6)
        mw = P_KIND == "wbc"
        if glow:
            ax.scatter(x[mw], y[mw], s=26, c="#39FF14", alpha=0.55, zorder=6)
    elif s["sep"] > 0:
        pass
    draw_flow(ax, s["flow"], p)
 
    # reference chambers
    fill_rect(ax, CH["Blank chamber"]["box"], 33, "#DDEFF2", alpha=0.9)
    fill_rect(ax, CH["Standard chamber"]["box"], 33, "#D7F5C8" if si >= 7 else "#EFE6FB", alpha=0.9)
 
    # optical head during the scan
    if si == 7:
        if p < 0.12:
            yh, xh, tag = 58.5, -10.5, "reading blank"
        elif p < 0.24:
            yh, xh, tag = 58.5, 10.5, "reading standard"
        else:
            q = (p - 0.24) / 0.76
            yh, xh, tag = 46 + 18 * q, 0.0, "scanning sample"
        ax.add_patch(FancyBboxPatch((xh - 2.2, yh - 0.6), 4.4, 1.2, boxstyle="round,pad=0.1",
                                    fc="#1E88E5", ec="#0D47A1", alpha=0.35, zorder=7))
        ax.add_patch(Rectangle((xh - 2.2, yh - 0.15), 4.4, 0.3, fc="#1E88E5", alpha=0.9, zorder=8))
        ax.text(xh + 2.6, yh, tag, fontsize=7, color="#0D47A1", va="center", zorder=8)
 
    # layer labels after separation
    if s["strat"] > 0.95:
        ax.text(-1.0, 0.5 * (Y_TOP + Y_BUFFY), "plasma", fontsize=6.5, ha="right", va="center", zorder=8)
        ax.text(-1.0, Y_BUFFY, "buffy coat (WBC)", fontsize=6.5, ha="right", va="center", zorder=8)
        ax.text(-1.0, Y_RBC + 1.8, "red cells (infected\ncells near top)", fontsize=6.5, ha="right",
                va="center", zorder=8)
    return si, p, rpm, t_real
 
 
def draw_full_disc(ax, angle):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(-DISC_RADIUS - 6, DISC_RADIUS + 6)
    ax.set_ylim(-DISC_RADIUS - 6, DISC_RADIUS + 6)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), DISC_RADIUS, fc="#E9EDF2", ec="#526273", lw=1.8))
    ax.add_patch(Circle((0, 0), CENTRE_HOLE_DIAMETER / 2, fc="white", ec="black", lw=1))
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    # test unit outline (rectangle in the rotating frame)
    box = np.array([[-15, 12], [15, 12], [15, 66], [-15, 66]])
    ax.add_patch(Polygon(box @ R.T, closed=True, fc="#C7D7EA", ec="#2B6CB0", lw=1.1))
    # index magnet / hole at 270 degrees in the disc frame
    ix = np.array([0, -62]) @ R.T
    ax.add_patch(Circle(ix, 2.2, fc="#222", ec="none"))
    ax.text(0, -DISC_RADIUS - 4.5, "Full disc (display rotation slowed)", fontsize=7.5, ha="center")
 
 
def draw_side_panels(ax_rpm, ax_info, ax_scan, f, si, p, rpm, t_real):
    # RPM timeline
    ax_rpm.clear()
    xs = np.arange(N_FRAMES)
    rp = [frame_info(k)[2] for k in xs]
    ax_rpm.plot(xs, rp, color="#999", lw=1)
    ax_rpm.plot(xs[: f + 1], rp[: f + 1], color="#2B6CB0", lw=2)
    ax_rpm.scatter([f], [rpm], color="#E63946", zorder=5, s=18)
    ax_rpm.axhline(V1_BURST, color="#F4A261", ls="--", lw=0.9)
    ax_rpm.axhline(V2_BURST, color="#D97706", ls="--", lw=0.9)
    ax_rpm.text(N_FRAMES * 0.99, V1_BURST, "V1 burst", fontsize=6.5, ha="right", va="bottom", color="#B45309")
    ax_rpm.text(N_FRAMES * 0.99, V2_BURST, "V2 burst", fontsize=6.5, ha="right", va="bottom", color="#B45309")
    ax_rpm.set_ylim(0, SEPARATION_RPM * 1.12)
    ax_rpm.set_xlim(0, N_FRAMES)
    ax_rpm.set_xticks([])
    ax_rpm.set_ylabel("rpm", fontsize=8)
    ax_rpm.tick_params(labelsize=7)
    ax_rpm.set_title("Spin protocol (time not to scale)", fontsize=9)
 
    # status text
    ax_info.clear()
    ax_info.axis("off")
    v1 = "OPEN" if si >= 2 else "closed"
    v2 = "OPEN" if si >= 4 else "closed"
    r_here = 58.0
    lines = [
        (STAGES[si][0], 11, "bold", "#111"),
        (f"Speed: {rpm:,.0f} rpm      RCF at 58 mm: {rcf(r_here, rpm):,.0f} g", 9, "normal", "#111"),
        (f"Real time: {int(t_real // 60)} min {int(t_real % 60):02d} s", 9, "normal", "#111"),
        (f"V1 (metering outlet): {v1}     V2 (mixing outlet): {v2}", 9, "normal", "#7A3E00"),
        ("No-return inlets: overflow and separation chambers enter at their inner edge", 8, "normal", "#7A0010"),
    ]
    y = 0.95
    for txt, fs, fw, col in lines:
        ax_info.text(0.0, y, txt, fontsize=fs, fontweight=fw, color=col, va="top", transform=ax_info.transAxes)
        y -= 0.19 if fs >= 11 else 0.16
 
    # fluorescence scan
    ax_scan.clear()
    ax_scan.set_title("Green fluorescence along the chamber (log scale)", fontsize=9)
    ax_scan.set_xlim(46, 64)
    ax_scan.set_ylim(300, PROFILE.max() * 3)
    ax_scan.set_yscale("log")
    ax_scan.set_xlabel("radius (mm)", fontsize=8)
    ax_scan.tick_params(labelsize=7)
    ax_scan.axvspan(Y_BUFFY - 0.6, INTEGRATION_START, color="#BBBBBB", alpha=0.35)
    ax_scan.axvspan(INTEGRATION_START, 64, color="#39FF14", alpha=0.10)
    if si == 7:
        q = max(0.0, (p - 0.24) / 0.76)
        n = int(q * len(SCAN_Y))
        if n > 1:
            ax_scan.plot(SCAN_Y[:n], PROFILE[:n], color="#2E7D32", lw=1.6)
    elif si == 8:
        ax_scan.plot(SCAN_Y, PROFILE, color="#2E7D32", lw=1.6)
    ax_scan.text(Y_BUFFY - 0.4, PROFILE.max() * 1.6, "leukocyte peak\n(excluded)", fontsize=6.5, ha="right",
                 va="top")
    ax_scan.text(61.5, 600, "parasite zone\n(integrated)", fontsize=6.5, ha="center")
    if si == 8:
        ax_scan.text(0.02, 0.95,
                     f"Estimated density ~ {EST_DENSITY:,.0f} parasites/µL\nLEVEL {LEVEL}: {LEVEL_TEXT}",
                     transform=ax_scan.transAxes, fontsize=9, fontweight="bold", va="top",
                     bbox=dict(fc="white", ec="#2E7D32"))
 
 
# ======================================================================
# OUTPUT FILES
# ======================================================================
 
def save_layout_png():
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(-16, 16)
    ax.set_ylim(67, 11)
    ax.set_aspect("equal")
    ax.set_yticks([20, 30, 40, 50, 60])
    ax.set_xticks([-10, 0, 10])
    ax.set_xlabel("mm", fontsize=8)
    ax.set_ylabel("radius from disc centre (mm)", fontsize=9)
    draw_static_layout(ax, labels=True)
    ax.set_title("Malaria lab-on-a-disc: one test unit\n"
                 "(orange = capillary burst valve, red triangle = no-return inlet, dotted = air vent)",
                 fontsize=10)
    fig.savefig("disc_layout.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("Saved: disc_layout.png")
 
 
def save_protocol_png():
    fig, ax = plt.subplots(figsize=(10, 4))
    t, r = [], []
    for si, (_, dur, _, fn) in enumerate(STAGES):
        for q in np.linspace(0, 1, 60):
            t.append(STAGE_START_T[si] + q * dur)
            r.append(max(0, fn(q)))
    ax.plot(np.array(t) / 60, r, color="#2B6CB0")
    ax.axhline(V1_BURST, color="#F4A261", ls="--", lw=1, label=f"V1 burst ~{V1_BURST:.0f} rpm")
    ax.axhline(V2_BURST, color="#D97706", ls="--", lw=1, label=f"V2 burst ~{V2_BURST:.0f} rpm")
    for si, (name, *_ ) in enumerate(STAGES):
        ax.axvline(STAGE_START_T[si] / 60, color="#DDD", lw=0.8)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("rpm")
    ax.set_title("Spin protocol for one test")
    ax.legend(fontsize=8)
    fig.savefig("spin_protocol.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved: spin_protocol.png")
 
 
def main():
    print("Malaria lab-on-a-disc simulation\n")
    write_calculations()
    print()
    save_layout_png()
    save_protocol_png()
 
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.45, 0.7, 1.5], height_ratios=[1, 0.62, 1.1],
                          wspace=0.28, hspace=0.42)
    ax_frame = fig.add_subplot(gs[:, 0])
    ax_disc = fig.add_subplot(gs[0, 1])
    ax_rpm = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, 1:])
    ax_scan = fig.add_subplot(gs[2, 1:])
    fig.suptitle("Malaria Lab-on-a-Disc: semi-quantitative optical analyser (simplified simulation)",
                 fontsize=12, fontweight="bold")
 
    disc_angle = [0.0]
 
    def update(f):
        si, p, rpm, t_real = draw_rotating_frame(ax_frame, f)
        disc_angle[0] += rpm / 60.0 * 2.5          # slowed display rotation
        draw_full_disc(ax_disc, disc_angle[0])
        draw_side_panels(ax_rpm, ax_info, ax_scan, f, si, p, rpm, t_real)
        if f % 25 == 0:
            print(f"  rendering frame {f + 1}/{N_FRAMES}")
        return []
 
    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=False)
    anim.save(GIF_NAME, writer=PillowWriter(fps=FPS), dpi=80)
    plt.close(fig)
    print(f"Saved: {GIF_NAME}")
    print(f"\nResult in this run: estimated {EST_DENSITY:,.0f} parasites/µL -> Level {LEVEL} ({LEVEL_TEXT})")
    print("Reminder: simplified model for demonstration, not a diagnostic result.")
 
 
if __name__ == "__main__":
    main()
