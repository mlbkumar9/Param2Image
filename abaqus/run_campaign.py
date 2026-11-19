"""
ABAQUS AUTOMATED IMPACT SIMULATION SCRIPT - GNN DATASET GENERATOR
=====================================================================

OVERVIEW:
Automates high-velocity spherical-projectile impacts on a clamped aluminium
plate using ABAQUS/Explicit, and produces a clean, balanced dataset for a deep
learning (GNN) impact-prediction model. The model is a FINAL-STATE surrogate:
given the impact conditions it reconstructs the final deformed 3-D mesh with
per-node stress / displacement and the per-element hole (element deletion), for
interactive inspection. Each valid run therefore stores only the INITIAL and
FINAL states (no time history). Downstream the ODB is converted to the training
representation (PyTorch Geometric mesh graph and/or voxel grid). Because the
plate mesh is UNIFORM and identical across runs, a mesh-native graph on the
fixed reference mesh maps directly to the interactive per-node output.

MODEL (current set - pure aluminium):
  - Plate   : 600 x 600 x 5 mm monolithic aluminium (Johnson-Cook plasticity
              + Johnson-Cook ductile damage with element deletion)
  - Impactor: RIGID sphere (excluded from field output)
  - A FUTURE set will reintroduce the Al+CFRP layup (4+1 mm) with Hashin damage.

DATASET DESIGN (current configuration):
  - Total target  : chosen at run time via --start/--end. With tiny final-state
    ODBs, 5000+ runs fit easily; more sims = the main accuracy lever.
    (e.g. --start 1 --end 5000)
  - Velocity 50-800 km/h, STRATIFIED:
        ~62.5% of runs in 400-800 km/h, split EQUALLY into four 100-km/h bands
        (400-500 / 500-600 / 600-700 / 700-800)
        ~37.5% of runs uniform in 50-400 km/h
  - Radius 25-100 mm: evenly STRATIFIED across the full range, drawn
    independently of velocity (even coverage of small AND large impactors)
  - Counts scale proportionally to whatever --start/--end total is used
    (the band ratios are referenced to the 1600 design size).
  - 'split' column tags ~250/1600 of runs 'validation', the rest 'train'
    (scales with total).

MESH (uniform, dense for the GNN):
  - 3.0 mm uniform in-plane seed, N_THROUGH = 6 layers through the 5 mm
    thickness -> ~240,000 C3D8R elements per run (kept >= 200k by design).
  - Uniform (no radius-scaled refinement) so element count and .odb size are
    constant across runs and voxelization is clean/consistent.

TIME SAMPLING (FINAL-STATE - no playback/rollout needed):
  - DROP_GAP = 2 mm so contact starts almost immediately for EVERY velocity
  - EXPLICIT_TOTAL_TIME = 2 ms window (long enough for perforation + immediate
    settling; the post-perforation state is taken as the "final" result)
  - NUM_FRAMES = 1 -> only 2 ODB frames are written: frame 0 (undeformed
    reference) and the final frame at 2 ms. Intermediate states are NOT stored.

STORAGE BUDGET:
  - Field output is PLATE ONLY (rigid impactor excluded) and OMITS 'V'.
  - With ~240k elements and only 2 frames, each plate-only .odb is ~58 MB
    (constant across runs), so storage is no longer the constraint - COMPUTE is:
        2000 runs ~= 116 GB    |   5000 runs ~= 290 GB    |   8000 runs ~= 465 GB
    All fit a ~760 GB drive. Spend the freed budget on MORE simulations
    (parametric diversity) - the main accuracy lever for a final-state surrogate.
  - Raise NUM_FRAMES only if time history is ever needed (~+11 MB per frame/run).

PIPELINE / RUN-MANAGEMENT BEHAVIOUR:
1. .ODB result files are PRESERVED in each RUN_XXX folder (not deleted).
   Each RUN_XXX folder keeps: before image, after image, .odb, and the job .log.
2. CSV has an 'odb_path' column with a portable RELATIVE path
   (RUN_XXX/Job.odb, forward-slashed) that resolves on any device relative to
   the Abaqus_OUT results folder - no hardcoded absolute paths.
3. Field output is restricted to the PLATE ONLY (rigid impactor excluded), so
   every element in the dataset carries STATUS/SDEG/PEEQ and participates in
   element deletion (fracture).
4. Only VALID runs are kept: failed/aborted attempts are discarded (folder
   removed, NO CSV row) and re-run with NEW in-band parameters until N valid
   runs exist, numbered contiguously RUN_001..RUN_N. The stratified
   distribution is preserved regardless of how many attempts fail.

KEY FEATURES:
- Batch processing with a target VALID-run count (auto-retry on failure)
- Stratified velocity / radius sampling; random impact position
- Johnson-Cook plasticity with element deletion for realistic hole formation
- Uniform dense plate mesh (~240k elements) for consistent GNN/voxel data
- Through-thickness partitioning for accurate stress gradients
- General contact with friction
- Plate-only field output (clean, homogeneous features for the GNN)
- Before/after PNG previews + CSV logging of all parameters and results

Runs under Abaqus/CAE (Abaqus Python), not the system Python. Results go to
--outdir: a bare folder name is created next to this script, an absolute path is
used as given. No other path is configured in the file.

USAGE (full design batch -> 5000 valid runs):
    abaqus cae noGUI=abaqus/run_campaign.py -- --start 1 --end 5000 --outdir /path/to/Abaqus_OUT

USAGE (quick sanity check -> 1 valid run):
    abaqus cae noGUI=abaqus/run_campaign.py -- --start 1 --end 1
"""

from __future__ import print_function
import os, sys, time, math, csv, datetime, traceback, tempfile, random, shutil, argparse

# =============================================================================
#                          COMMAND LINE ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='Run Abaqus impact simulations in batches.')
parser.add_argument('--start', type=int, default=1, help='Starting run index (inclusive).')
parser.add_argument('--end', type=int, default=100, help='Ending run index (inclusive).')
# --- Sampling-window overrides (defaults reproduce the original 5000-run campaign) ---
# Added for the low-energy / non-perforating extension batch. The plate geometry,
# mesh, material model and step time are deliberately NOT exposed here: every run
# in every batch must share them so the datasets remain directly comparable.
parser.add_argument('--vmin', type=float, default=50.0, help='Min impact velocity (km/h).')
parser.add_argument('--vmax', type=float, default=800.0, help='Max impact velocity (km/h).')
parser.add_argument('--rmin', type=float, default=25.0, help='Min projectile radius (mm).')
parser.add_argument('--rmax', type=float, default=100.0, help='Max projectile radius (mm).')
parser.add_argument('--seed', type=int, default=12345, help='RNG seed for parameter sampling.')
parser.add_argument('--outdir', type=str, default='Abaqus_OUT', help='Output folder name under the script directory.')
parser.add_argument('--frames', type=int, default=1, help='Field-output intervals (1 = initial+final only).')
parser.add_argument('--plan', type=str, default='stratified',
                    choices=['stratified', 'uniform', 'grid'],
                    help="'stratified' = original high-velocity-weighted plan; "
                         "'uniform' = uniform over [vmin,vmax]x[rmin,rmax]; "
                         "'grid' = deterministic sweep from --grid-csv.")
parser.add_argument('--grid-csv', type=str, default='',
                    help="CSV with columns r_mm,v_kmh giving explicit runs (--plan grid).")
parser.add_argument('--val-frac', type=float, default=250.0 / 1600.0,
                    help='Fraction of runs tagged validation.')
args, unknown = parser.parse_known_args()
START_RUN = args.start
END_RUN = args.end

# =============================================================================
#                          CONFIGURATION PARAMETERS
# =============================================================================

# --- Directory Setup (FIXED) ---
# Get the script directory reliably
try:
    # Try to get the script's actual location
    if '__file__' in dir():
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        log_msg = "Script directory from __file__: %s" % SCRIPT_DIR
    else:
        # Fallback: In Abaqus noGUI mode, use sys.argv to find script location
        import sys
        for arg in sys.argv:
            if arg.endswith('.py') and os.path.exists(arg):
                SCRIPT_DIR = os.path.dirname(os.path.abspath(arg))
                log_msg = "Script directory from sys.argv: %s" % SCRIPT_DIR
                break
        else:
            # Last resort: use current working directory at script start
            SCRIPT_DIR = os.getcwd()
            log_msg = "Script directory from cwd (fallback): %s" % SCRIPT_DIR
except Exception as e:
    SCRIPT_DIR = os.getcwd()
    log_msg = "Script directory from cwd (exception fallback): %s - Error: %s" % (SCRIPT_DIR, e)

# FIXED: Ensure OUT_ROOT uses SCRIPT_DIR
OUT_ROOT = SCRIPT_DIR
# --outdir may be a bare folder name (kept under the script dir, original
# behaviour) or an absolute path, so a batch can be written to another drive.
OUT_DIR = args.outdir if os.path.isabs(args.outdir) else os.path.join(OUT_ROOT, args.outdir)
DEBUG_LOG = os.path.join(OUT_DIR, "debug.log")
FALLBACK_LOG = os.path.join(tempfile.gettempdir(), "Abaqus_PS_debug.log")

# --- Simulation Control ---
RANDOM_SEED = args.seed          # For reproducible randomization
EXPORT_IMAGES = True             # Enable/disable image generation
VELOCITY_MIN_KMH = args.vmin     # Minimum impact velocity (km/h)
VELOCITY_MAX_KMH = args.vmax     # Maximum impact velocity (km/h)

# Only VALID runs (COMPLETED + readable .odb) are kept. Failed/aborted attempts
# are discarded (folder removed, NO CSV row) and re-run with new random
# parameters until the requested number of valid runs is reached.
# MAX_ATTEMPT_FACTOR caps total attempts = target_count * MAX_ATTEMPT_FACTOR so a
# pathological failure rate can't loop forever.
MAX_ATTEMPT_FACTOR = 5

# --- Plate Geometry (millimeters) ---
# 600 x 600 x 5 mm aluminium plate (matches the future Al+CFRP layup envelope,
# where the composite set will be 4 mm Al + 1 mm CFRP = 5 mm total).
PLATE_L = 600.0                  # Length: 600 mm
PLATE_W = 600.0                  # Width:  600 mm
PLATE_T = 5.0                    # Thickness: 5 mm

# --- Material Properties ---
PLATE_MATERIAL = 'Aluminium'

# Johnson-Cook constitutive model parameters for Aluminum 2024-T351
JC = {
    # Plasticity parameters
    'A': 503.0,                  # Yield stress (MPa)
    'B': 370.0,                  # Hardening constant (MPa)
    'n': 0.34,                   # Hardening exponent
    'C': 0.025,                  # Strain rate constant
    'm': 1.0,                    # Thermal softening exponent
    'Tref': 293.0,               # Reference temperature (K)
    'Tmelt': 933.0,              # Melting temperature (K)
    'ref_strain_rate': 1.0,      # Reference strain rate (1/s)
    'rho': 2810.0e-9,            # Density (kg/mm³)
    
    # Failure parameters for element deletion
    'd1': 0.096,                 # Failure parameter 1
    'd2': 0.049,                 # Failure parameter 2
    'd3': -3.465,                # Failure parameter 3
    'd4': 0.016,                 # Failure parameter 4
    'd5': 1.099                  # Failure parameter 5
}

# --- Projectile Properties ---
PROJ_DENSITY = 7850.0e-9         # Steel density (kg/mm³)
PROJECTILE_MIN_RADIUS = args.rmin  # Minimum projectile radius (mm)
PROJECTILE_MAX_RADIUS = args.rmax  # Maximum projectile radius (mm)
PLATE_AREA = PLATE_L * PLATE_W   # (kept for reference / KE reporting)

# --- Mesh Parameters ---
# UNIFORM in-plane mesh on the small 600x600 plate: predictable element count,
# constant .odb size across runs, and clean/consistent voxelization for the GNN
# (a radius-scaled refinement zone would blow up element count for large r).
#   elems ~= (PLATE_L/GLOBAL_SEED)^2 * N_THROUGH
#   3.0 mm seed, 6 layers -> ~(600/3)^2 * 6 = 240,000 elements (>= 200k target)
#   even the smallest 25 mm projectile's hole spans ~17 elements (50 mm / 3 mm).
PLATE_GLOBAL_SEED = 3.0          # Uniform in-plane mesh size (mm)
PLATE_LOCAL_SEED = 3.0           # Same as global -> uniform mesh
N_THROUGH = 6                    # element layers through the 5 mm thickness
# Projectile (rigid) mesh seed is set PER RUN from its radius (see loop):
#   proj_seed = clamp(r / 12, 3, 25) mm

# --- Solver Configuration ---
# Tuned for a GNN rollout/dynamics dataset (MeshGraphNets/GNS style):
#   - small DROP_GAP so contact starts almost immediately for every velocity
#   - short EXPLICIT_TOTAL_TIME focused on impact + immediate damage response
#   - NUM_FRAMES dense uniform snapshots -> ~NUM_FRAMES transitions per sim
DROP_GAP = 2.0                   # mm clearance: sphere bottom above plate top
EXPLICIT_TOTAL_TIME = 0.002      # Simulation duration (s) - 2 ms impact window
NUM_FRAMES = args.frames         # FINAL-STATE dataset: only initial + final are
                                 # needed -> 2 ODB frames (frame 0 reference +
                                 # final at EXPLICIT_TOTAL_TIME). ~58 MB / .odb.
                                 # Raise ONLY if time history is ever required.
NUM_CPUS = 4                     # Number of CPU cores to use

# --- Output Configuration ---
CSV_NAME = "results_%d-%d.csv" % (START_RUN, END_RUN)

# =============================================================================
#                          LOGGING SYSTEM
# =============================================================================

def _ensure_dir(p):
    """Create directory if it doesn't exist"""
    if not os.path.exists(p):
        os.makedirs(p)

_ensure_dir(OUT_DIR)
_log_handles = []

def _open_log(path):
    """Safely open a log file"""
    try:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)
        return open(path, "a")
    except Exception:
        return None

# Initialize log files
for lp in (DEBUG_LOG, FALLBACK_LOG):
    h = _open_log(lp)
    if h:
        _log_handles.append((lp, h))

def log(msg):
    """Write timestamped message to all log files and console"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (ts, msg)
    try:
        print(line)
    except Exception:
        pass
    dead = []
    for p, h in _log_handles:
        try:
            h.write(line + "\n")
            h.flush()
            os.fsync(h.fileno())
        except Exception:
            dead.append((p, h))
    for d in dead:
        try:
            _log_handles.remove(d)
        except:
            pass

def _fatal_handler(et, ev, tb):
    """Handle fatal exceptions with detailed logging"""
    log("FATAL: %s: %s" % (getattr(et, '__name__', str(et)), ev))
    log("TRACE:\n" + "".join(traceback.format_exception(et, ev, tb)))
    for p, h in _log_handles:
        try:
            h.close()
        except:
            pass
    sys.exit(1)

sys.excepthook = _fatal_handler
log("Script started")
log(log_msg)  # Log the directory detection method used
log("Output directory will be: %s" % OUT_DIR)

# =============================================================================
#                          IMPORT ABAQUS MODULES
# =============================================================================
try:
    from abaqus import mdb, session
    from abaqusConstants import *
    import regionToolset, assembly, part, mesh, step, job, visualization, odbAccess, interaction
    import displayGroupOdbToolset as dgo
    log("Abaqus modules imported successfully")
except Exception as e:
    log("ERROR: Abaqus modules not available. Run with 'abaqus cae noGUI=...'.")
    raise

# =============================================================================
#                          CSV RESULTS FILE SETUP
# =============================================================================
csv_path = os.path.join(OUT_DIR, CSV_NAME)
csv_fields = [
    'run_id', 'split', 'timestamp', 'job_name',
    'plate_material', 'plate_L_mm', 'plate_W_mm', 'plate_T_mm',
    'proj_shape', 'proj_radius_mm', 'proj_density_kgmm3', 'proj_mass_kg',
    'impact_x_mm', 'impact_y_mm', 'impact_z_mm',
    'velocity_kmh', 'velocity_ms', 'velocity_mms',
    'KE_J',
    'plate_elem_count', 'plate_node_count', 'proj_elem_count', 'proj_node_count',
    'job_status', 'runtime_s',
    'max_mises_mpa', 'max_peeq', 'max_deflection_mm',
    'img_before', 'img_after', 'odb_path', 'solver_tail'
]

csv_exists = os.path.isfile(csv_path)
csv_file = open(csv_path, 'a', newline='')
csv_writer = csv.writer(csv_file)
if not csv_exists:
    csv_writer.writerow(csv_fields)
    log("Created new CSV with header at: %s" % csv_path)
else:
    log("Appending to existing CSV at: %s" % csv_path)
csv_file.flush()

# =============================================================================
#                          HELPER FUNCTIONS
# =============================================================================

def kmh_to_ms(kmh):
    """Convert km/h to m/s"""
    return kmh * 1000.0 / 3600.0

def ms_to_mms(ms):
    """Convert m/s to mm/s"""
    return ms * 1000.0

def tail_of_files(job_name, lines=120):
    """Extract last N lines from job output files for debugging"""
    out = ""
    for fn in (job_name + ".msg", job_name + ".sta", job_name + ".dat"):
        if os.path.exists(fn):
            try:
                with open(fn, 'r') as f:
                    L = f.readlines()[-lines:]
                    out += "\n--- tail of %s ---\n%s" % (fn, "".join(L))
            except Exception:
                pass
    return out

def _cleanup_run_folder(run_folder, keep_paths):
    """
    Remove all files except those specified in keep_paths.
    This saves significant disk space by removing large ODB and other files.
    """
    try:
        keep_abs = set(os.path.abspath(p) for p in keep_paths if p)
        for name in os.listdir(run_folder):
            p = os.path.join(run_folder, name)
            abs_p = os.path.abspath(p)
            
            if abs_p in keep_abs:
                continue
                
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                    log("Cleaned up file: %s" % name)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
                    log("Cleaned up directory: %s" % name)
            except Exception as e:
                log("Cleanup skip %s: %s" % (name, e))
            
    except Exception as e:
        log("Cleanup error (non-fatal): %s" % e)

# =============================================================================
#                          MAIN SIMULATION LOOP
# =============================================================================

TARGET_SUCCESS = END_RUN                          # desired number of VALID runs
N_TARGET = END_RUN - START_RUN + 1                # how many we must produce
MAX_ATTEMPTS = N_TARGET * MAX_ATTEMPT_FACTOR      # safety cap on total attempts
success_count = START_RUN - 1                     # index of last VALID run written
attempt = 0

# =============================================================================
#   STRATIFIED PARAMETER PLAN  (one entry per VALID run, length N_TARGET)
# =============================================================================
# Velocity (scaled to N_TARGET; exact counts at the design size of 1600):
#   - 1000/1600 (62.5%) of runs in 400-800 km/h, split EQUALLY into four
#     100-km/h bands: 400-500, 500-600, 600-700, 700-800 (250 each at N=1600)
#   - 600/1600 (37.5%) of runs uniform in 50-400 km/h
# Radius: evenly STRATIFIED across 50-500 mm (one equal slice per run) so the
#   whole range is covered uniformly; drawn independently of velocity.
# Split: exactly round(250/1600 * N) runs flagged 'validation', rest 'train'.
# A failed run re-draws WITHIN ITS OWN velocity & radius slice (see loop), so
# the distribution is preserved no matter how many attempts fail.
def _build_param_plan(n_total, seed):
    rng = random.Random(seed)

    # ---- 'grid': deterministic (r, v) sweep read from --grid-csv -------------
    # Used to bracket the ballistic limit with a small pilot before committing
    # a large batch. Each row becomes exactly one run, in file order.
    if args.plan == 'grid':
        pairs = []
        with open(args.grid_csv, 'r') as gh:
            for rec in csv.DictReader(gh):
                pairs.append((float(rec['r_mm']), float(rec['v_kmh'])))
        if not pairs:
            raise RuntimeError('--grid-csv contained no rows')
        out = []
        for i in range(n_total):
            r_mm, v_kmh = pairs[i % len(pairs)]
            out.append({'vlo': v_kmh, 'vhi': v_kmh,      # degenerate band = exact value
                        'rlo': r_mm, 'rhi': r_mm,
                        'split': 'train'})
        return out, 0

    # ---- 'uniform': flat over the requested window ---------------------------
    # For the low-energy / non-perforating extension batch, where the original
    # high-velocity weighting is inappropriate.
    if args.plan == 'uniform':
        rmin, rmax = PROJECTILE_MIN_RADIUS, PROJECTILE_MAX_RADIUS
        vmin, vmax = VELOCITY_MIN_KMH, VELOCITY_MAX_KMH
        wv = (vmax - vmin) / float(n_total)
        wr = (rmax - rmin) / float(n_total)
        vel = [(vmin + i * wv, vmin + (i + 1) * wv) for i in range(n_total)]
        rad = [(rmin + i * wr, rmin + (i + 1) * wr) for i in range(n_total)]
        rng.shuffle(vel)
        rng.shuffle(rad)
        plan = list(zip(vel, rad))
        n_val = int(round(n_total * args.val_frac))
        val_idx = set(rng.sample(range(n_total), min(n_val, n_total)))
        out = []
        for i, ((vlo, vhi), (rlo, rhi)) in enumerate(plan):
            out.append({'vlo': vlo, 'vhi': vhi, 'rlo': rlo, 'rhi': rhi,
                        'split': 'validation' if i in val_idx else 'train'})
        return out, n_val

    # ---- 'stratified': the original 5000-run campaign plan -------------------
    # --- velocity slices ---
    n_high = int(round(n_total * 1000.0 / 1600.0))
    n_low = n_total - n_high
    high_bands = [(400.0, 500.0), (500.0, 600.0), (600.0, 700.0), (700.0, 800.0)]
    base, rem = divmod(n_high, 4)
    band_counts = [base + (1 if i < rem else 0) for i in range(4)]
    vel = []
    for (lo, hi), c in zip(high_bands, band_counts):
        vel += [(lo, hi)] * c
    vel += [(50.0, 400.0)] * n_low
    # --- radius slices: equal-width strata over [50, 500] mm ---
    rmin, rmax = PROJECTILE_MIN_RADIUS, PROJECTILE_MAX_RADIUS
    w = (rmax - rmin) / float(n_total)
    rad = [(rmin + i * w, rmin + (i + 1) * w) for i in range(n_total)]
    rng.shuffle(rad)                              # decouple radius from velocity
    # --- pair, then shuffle run order so the batch isn't band-ordered ---
    plan = list(zip(vel, rad))
    rng.shuffle(plan)
    # --- train/validation split (exact count, random subset) ---
    n_val = int(round(n_total * 250.0 / 1600.0))
    val_idx = set(rng.sample(range(n_total), min(n_val, n_total)))
    out = []
    for i, ((vlo, vhi), (rlo, rhi)) in enumerate(plan):
        out.append({'vlo': vlo, 'vhi': vhi, 'rlo': rlo, 'rhi': rhi,
                    'split': 'validation' if i in val_idx else 'train'})
    return out, n_val

PARAM_PLAN, N_VAL = _build_param_plan(N_TARGET, RANDOM_SEED)
log("Target: %d valid runs (RUN_%03d..RUN_%03d); split = %d train / %d validation; max attempts=%d"
    % (N_TARGET, START_RUN, END_RUN, N_TARGET - N_VAL, N_VAL, MAX_ATTEMPTS))

while success_count < TARGET_SUCCESS:
    if attempt >= MAX_ATTEMPTS:
        log("ABORT: hit MAX_ATTEMPTS=%d with only %d/%d valid runs"
            % (MAX_ATTEMPTS, success_count - (START_RUN - 1), N_TARGET))
        break
    attempt += 1

    # Next valid run takes the next contiguous index -> RUN_001..RUN_N, no gaps
    candidate_idx = success_count + 1
    run_id = "RUN_%03d" % candidate_idx
    timestamp = datetime.datetime.now().isoformat()
    job_name = "ImpactJob_%s" % run_id
    run_folder = os.path.join(OUT_DIR, run_id)
    # Clear any leftover folder from a previously discarded attempt at this index
    if os.path.exists(run_folder):
        try: shutil.rmtree(run_folder)
        except Exception: pass
    _ensure_dir(run_folder)
    run_ok = False                                 # set True only on a valid run
    log("=== Attempt %d -> %s (valid so far: %d/%d) ==="
        % (attempt, run_id, success_count - (START_RUN - 1), N_TARGET))
    t_start = time.time()

    # -------------------------------------------------------------------------
    #                    RANDOMIZE SIMULATION PARAMETERS
    # -------------------------------------------------------------------------
    # Seed tied to ATTEMPT (not run index) so a re-run after a failure gets NEW
    # parameters - retrying identical params would just fail again.
    run_seed = RANDOM_SEED + attempt
    random.seed(run_seed)
    log("Seeding attempt %d with %d" % (attempt, run_seed))

    # This valid-run slot's stratified parameter slice (velocity + radius bands).
    # On a retry after failure we keep the SAME slot, so the draw stays within the
    # same bands -> the stratified distribution is preserved exactly.
    slot = success_count - (START_RUN - 1)         # 0-based index of run to produce
    spec = PARAM_PLAN[slot]
    split_tag = spec['split']

    # Projectile radius - uniform within this slot's radius stratum
    r = random.uniform(spec['rlo'], spec['rhi'])

    # Impact velocity - uniform within this slot's velocity band
    v_kmh = random.uniform(spec['vlo'], spec['vhi'])
    v_ms = kmh_to_ms(v_kmh)
    v_mms = ms_to_mms(v_ms)
    log("Params: split=%s  r=%.1f mm (band %.0f-%.0f)  v=%.1f km/h (band %.0f-%.0f)"
        % (split_tag, r, spec['rlo'], spec['rhi'], v_kmh, spec['vlo'], spec['vhi']))

    # Per-run projectile (rigid) mesh seed, scaled to its radius
    proj_seed = max(min(r / 12.0, 25.0), 3.0)

    # Random impact position (ensuring projectile stays on plate)
    x = random.uniform(-PLATE_L/2.0 + r, PLATE_L/2.0 - r)
    y = random.uniform(-PLATE_W/2.0 + r, PLATE_W/2.0 - r)
    # Sphere centre height so its bottom sits DROP_GAP above the plate top face
    # (plate top is at z = PLATE_T) -> near-immediate, consistent contact.
    z0 = PLATE_T + DROP_GAP + r

    # Calculate projectile mass and kinetic energy
    rho = PROJ_DENSITY
    vol = 4.0/3.0 * math.pi * (r**3)
    mass = rho * vol
    KE = 0.5 * mass * (v_ms**2)
    I = 2.0/5.0 * mass * (r**2)  # Moment of inertia for sphere

    plate_material_name = PLATE_MATERIAL
    
    try:
        # =====================================================================
        #                          MODEL CREATION
        # =====================================================================
        model_name = "Model_" + run_id
        if model_name in mdb.models:
            del mdb.models[model_name]
        model = mdb.Model(name=model_name)
        log("Model created: %s" % model_name)

        # =====================================================================
        #                     CREATE PROJECTILE PART
        # =====================================================================
        log("Creating spherical projectile with radius=%.3f mm" % r)
        s = model.ConstrainedSketch(name='projSketch', sheetSize=max(2.0*r, 1.0))
        s.ConstructionLine(point1=(0.0, -r), point2=(0.0, r))
        s.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, -r), 
                         point2=(0.0, r), direction=CLOCKWISE)
        proj_part = model.Part(name='ProjPart', dimensionality=THREE_D, 
                              type=DISCRETE_RIGID_SURFACE)
        proj_part.BaseShellRevolve(sketch=s, angle=360.0)
        del s

        # Mesh the projectile
        proj_part.seedPart(size=proj_seed, deviationFactor=0.1, minSizeFactor=0.1)
        proj_faces = proj_part.faces[:]
        proj_part.setMeshControls(regions=proj_faces, elemShape=QUAD, 
                                 technique=FREE, algorithm=MEDIAL_AXIS)
        et_r3d4 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
        et_r3d3 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
        proj_part.setElementType(regions=regionToolset.Region(faces=proj_faces), 
                                elemTypes=(et_r3d4, et_r3d3))
        proj_part.generateMesh()
        proj_elem_count = len(proj_part.elements)
        proj_node_count = len(proj_part.nodes)
        log("Projectile meshed: elems=%d nodes=%d" % (proj_elem_count, proj_node_count))

        # =====================================================================
        #                     CREATE PLATE PART (3D SOLID)
        # =====================================================================
        log("Creating plate (C3D8R solid) LxWxT=%.1fx%.1fx%.1f mm" % 
            (PLATE_L, PLATE_W, PLATE_T))
        
        sk_plate = model.ConstrainedSketch(name='plateSketch', 
                                          sheetSize=max(PLATE_L, PLATE_W)*2.0)
        sk_plate.rectangle(point1=(-PLATE_L/2.0, -PLATE_W/2.0), 
                          point2=(PLATE_L/2.0, PLATE_W/2.0))
        plate_part = model.Part(name='PlatePart', dimensionality=THREE_D, 
                               type=DEFORMABLE_BODY)
        plate_part.BaseSolidExtrude(sketch=sk_plate, depth=PLATE_T)
        del sk_plate
        log("Plate solid extruded")

        # =====================================================================
        #              THROUGH-THICKNESS PARTITIONING
        # =====================================================================
        # Create N_THROUGH-1 evenly-spaced partition planes -> N_THROUGH element
        # layers through the thickness (good z-gradient capture for perforation
        # without an extreme through-thickness aspect ratio on the thin plate).
        try:
            log("Creating through-thickness partitions (%d layers)..." % N_THROUGH)
            for i in range(1, N_THROUGH):
                offset = (PLATE_T / float(N_THROUGH)) * i
                datum_plane = plate_part.DatumPlaneByPrincipalPlane(
                    principalPlane=XYPLANE, offset=offset)
                plate_part.PartitionCellByDatumPlane(
                    datumPlane=plate_part.datums[datum_plane.id], 
                    cells=plate_part.cells[:])
            log("Through-thickness partitions created")
        except Exception as e:
            log("Partition warning (non-critical): %s" % e)

        # =====================================================================
        #                     MATERIAL DEFINITION
        # =====================================================================
        log("Assigning Aluminium with Johnson-Cook plasticity and failure")
        mat = model.Material(name='PlateMat_Al')
        mat.Density(table=((JC['rho'],),))
        mat.Elastic(table=((70000.0, 0.3), ))  # Young's modulus (MPa), Poisson's ratio
        
        # Johnson-Cook plasticity model
        mat.Plastic(hardening=JOHNSON_COOK, 
                   table=((JC['A'], JC['B'], JC['n'], JC['m'], 
                          JC['Tmelt'], JC['Tref']), ))
        mat.plastic.RateDependent(type=JOHNSON_COOK, 
                                 table=((JC['C'], JC['ref_strain_rate']),))
        
        # Johnson-Cook failure criterion for element deletion (hole formation)
        mat.JohnsonCookDamageInitiation(
            table=((JC['d1'], JC['d2'], JC['d3'], JC['d4'], JC['d5'], 
                   JC['Tmelt'], JC['Tref'], JC['ref_strain_rate']), ))
        mat.johnsonCookDamageInitiation.DamageEvolution(
            type=DISPLACEMENT, table=((0.01,),))
        
        # Create and assign section
        section_name = 'PlateSection_Al'
        model.HomogeneousSolidSection(name=section_name, material='PlateMat_Al', 
                                     thickness=PLATE_T)
        plate_region_cells = regionToolset.Region(cells=plate_part.cells[:])
        plate_part.SectionAssignment(region=plate_region_cells, sectionName=section_name)

        # =====================================================================
        #                     PLATE MESHING (UNIFORM)
        # =====================================================================
        # PLATE_LOCAL_SEED == PLATE_GLOBAL_SEED, so the impact-zone partition
        # below is effectively a no-op and the plate mesh is UNIFORM (~240k elems,
        # constant .odb size, clean voxelization). Set LOCAL < GLOBAL to re-enable
        # local refinement (note: that makes element count / file size vary with r).

        # Uniform seed across the whole plate
        plate_part.seedPart(size=PLATE_GLOBAL_SEED, deviationFactor=0.1, minSizeFactor=0.1)
        
        # Set element type: C3D8R with enhanced hourglass control
        et_c3d8r = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT, 
                                kinematicSplit=AVERAGE_STRAIN,
                                hourglassControl=ENHANCED,
                                elemDeletion=ON)
        plate_part.setElementType(regions=regionToolset.Region(cells=plate_part.cells[:]), 
                                 elemTypes=(et_c3d8r,))

        # =====================================================================
        #         LOCAL REFINEMENT AROUND IMPACT ZONE
        # =====================================================================
        # Create a cylindrical partition around the impact point for finer mesh
        try:
            log("Creating cylindrical partition for local refinement...")
            cx, cy = x, y
            local_ref_rad = 2.0 * r  # Refinement zone radius (~projectile footprint)

            # Find top face and create sketch
            top_face = plate_part.faces.findAt((0, 0, PLATE_T))
            transform = plate_part.MakeSketchTransform(
                sketchPlane=top_face, sketchPlaneSide=SIDE1, 
                origin=(cx, cy, PLATE_T))
            
            sk = model.ConstrainedSketch(name='__refinement_sketch__', 
                                        sheetSize=local_ref_rad * 3, 
                                        transform=transform)
            sk.CircleByCenterPerimeter(center=(0, 0), point1=(local_ref_rad, 0))

            # Partition the cell with the circle
            plate_part.PartitionCellBySketch(cells=plate_part.cells, sketch=sk)
            log("Cylindrical partition created successfully")

            # Assign different mesh sizes to inner and outer regions
            inner_cell = plate_part.cells.findAt(((cx, cy, PLATE_T/2.0),))
            all_cells = plate_part.cells[:]
            outer_cells = [c for c in all_cells if c.index != inner_cell.index]

            plate_part.setMeshControls(regions=inner_cell, size=PLATE_LOCAL_SEED)
            plate_part.setMeshControls(regions=outer_cells, size=PLATE_GLOBAL_SEED)

            log("Mesh controls: inner=%.2f mm, outer=%.2f mm" % 
                (PLATE_LOCAL_SEED, PLATE_GLOBAL_SEED))

        except Exception as e:
            log("Local refinement skipped (non-fatal): %s" % e)
            plate_part.seedPart(size=PLATE_GLOBAL_SEED, deviationFactor=0.1, 
                               minSizeFactor=0.1)

        # Generate the mesh
        plate_part.generateMesh()
        plate_elem_count = len(plate_part.elements)
        plate_node_count = len(plate_part.nodes)
        log("Plate meshed: elems=%d nodes=%d" % (plate_elem_count, plate_node_count))

        # =====================================================================
        #                     ASSEMBLY
        # =====================================================================
        log("Creating assembly")
        a = model.rootAssembly
        a.DatumCsysByDefault(CARTESIAN)
        plate_inst = a.Instance(name='PlateInst', part=plate_part, dependent=ON)
        proj_inst = a.Instance(name='ProjInst', part=proj_part, dependent=ON)
        
        # Position projectile above impact point
        a.translate(instanceList=('ProjInst',), vector=(x, y, z0))
        log("Projectile positioned at (%.2f, %.2f, %.2f)" % (x, y, z0))

        # Create reference point for rigid body control
        rp = a.ReferencePoint(point=(x, y, z0))
        rp_set = a.Set(referencePoints=(a.referencePoints[rp.id],), name='RP_Proj')
        proj_faces_asm = a.Set(faces=proj_inst.faces, name='ProjFaces_Asm')
        model.RigidBody(name='ProjRigid', refPointRegion=rp_set, 
                       bodyRegion=proj_faces_asm)

        # =====================================================================
        #              ASSIGN MASS AND INERTIA TO PROJECTILE
        # =====================================================================
        log("Assigning mass and inertia to projectile reference point")
        try:
            a.engineeringFeatures.PointMassInertia(
                name='Proj_RP_MassInertia', 
                region=rp_set, 
                mass=mass, 
                i11=I, i22=I, i33=I)
            log("PointMassInertia: mass=%.6f kg, I=%.6f kg·mm²" % (mass, I))
        except Exception as e_pm:
            log("ERROR assigning PointMassInertia: %s" % str(e_pm))
            try:
                model.engineeringFeatures.PointMassInertia(
                    name='Proj_RP_MassInertia', 
                    region=rp_set, 
                    mass=mass, 
                    i11=I, i22=I, i33=I)
                log("PointMassInertia via model.engineeringFeatures")
            except Exception as e2:
                log("FATAL: Could not assign PointMassInertia. %s | %s" % 
                    (str(e_pm), str(e2)))
                raise

        # =====================================================================
        #              BOUNDARY CONDITIONS (CLAMPED EDGES)
        # =====================================================================
        log("Applying encastre boundary condition on plate edges")
        outer_edges = plate_inst.edges.getByBoundingBox(
            xMin=-PLATE_L/2.0 - 1.0, yMin=-PLATE_W/2.0 - 1.0, zMin=-1.0,
            xMax=PLATE_L/2.0 + 1.0, yMax=PLATE_W/2.0 + 1.0, zMax=PLATE_T + 1.0)
        if len(outer_edges) > 0:
            bc_region = a.Set(edges=outer_edges, name='PlateEdgesClamp')
        else:
            bc_region = a.Set(faces=plate_inst.faces[:], name='PlateFaceClamp')
        model.EncastreBC(name='PlateFix', createStepName='Initial', region=bc_region)

        # =====================================================================
        #              CONTACT INTERACTION WITH FRICTION
        # =====================================================================
        log("Creating contact interaction with friction coefficient = 0.3")
        if 'IntProp' not in model.interactionProperties.keys():
            model.ContactProperty('IntProp')
        
        # Tangential behavior (friction)
        model.interactionProperties['IntProp'].TangentialBehavior(
            formulation=PENALTY, 
            directionality=ISOTROPIC, 
            slipRateDependency=OFF, 
            pressureDependency=OFF, 
            temperatureDependency=OFF, 
            dependencies=0, 
            table=((0.3,),),  # Coulomb friction coefficient
            shearStressLimit=None, 
            maximumElasticSlip=FRACTION, 
            fraction=0.005, 
            elasticSlipStiffness=None)
        
        # Normal behavior (hard contact)
        model.interactionProperties['IntProp'].NormalBehavior(
            pressureOverclosure=HARD, 
            allowSeparation=ON, 
            constraintEnforcementMethod=DEFAULT)
        
        # Create general contact
        model.ContactExp(name='GenContact', createStepName='Initial')
        model.interactions['GenContact'].includedPairs.setValuesInStep(
            stepName='Initial', useAllstar=ON)
        model.interactions['GenContact'].contactPropertyAssignments.appendInStep(
            stepName='Initial', assignments=((GLOBAL, SELF, 'IntProp'), ))

        # =====================================================================
        #              EXPLICIT DYNAMICS STEP
        # =====================================================================
        log("Creating Explicit Dynamics step")
        model.ExplicitDynamicsStep(
            name='ImpactStep', 
            previous='Initial', 
            timePeriod=EXPLICIT_TOTAL_TIME, 
            improvedDtMethod=ON,
            nlgeom=ON,
            massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 0.0, 0.0, 
                         None, 0, 0, 0.0, 0.0, 0, None), ))

        # =====================================================================
        #              INITIAL VELOCITY
        # =====================================================================
        log("Applying initial velocity: %.2f km/h => %.3f m/s => %.3f mm/s" % 
            (v_kmh, v_ms, v_mms))
        model.Velocity(
            name='InitVel', 
            region=rp_set, 
            field='', 
            distributionType=MAGNITUDE, 
            velocity1=0.0, 
            velocity2=0.0, 
            velocity3=-v_mms,  # Negative Z direction (downward)
            omega=0.0)

        # =====================================================================
        #              OUTPUT REQUESTS  (PLATE ONLY)
        # =====================================================================
        # Restrict field output to the deformable plate ONLY, so the rigid
        # impactor is EXCLUDED from the .odb dataset. This gives a single,
        # consistent body in the training data - every element carries STATUS /
        # SDEG / PEEQ and participates in element deletion (fracture). The rigid
        # projectile (no STATUS, cannot fracture) is not written to field output.
        try:
            plate_out_region = a.Set(cells=plate_inst.cells[:],
                                     name='PlateFieldRegion')
            log("Plate-only field-output region created (cells)")
        except Exception as _ereg:
            log("Cell region failed (%s); falling back to faces" % _ereg)
            plate_out_region = a.Set(faces=plate_inst.faces[:],
                                     name='PlateFieldRegion')

        try:
            # 'V' (velocity) intentionally OMITTED to save disk - it is derivable
            # from the displacement (U) history by finite difference, which is how
            # GNS / MeshGraphNets models compute node velocity anyway.
            model.fieldOutputRequests['F-Output-1'].setValues(
                variables=('S', 'PEEQ', 'U', 'STATUS', 'SDEG'),
                region=plate_out_region,
                numIntervals=NUM_FRAMES)
        except Exception:
            model.FieldOutputRequest(
                name='F-Output-1',
                createStepName='ImpactStep',
                variables=('S', 'PEEQ', 'U', 'STATUS', 'SDEG'),
                region=plate_out_region)
        log("Field output restricted to plate region (impactor excluded)")

        # =====================================================================
        #              JOB CREATION AND SUBMISSION
        # =====================================================================
        log("Creating and submitting job: %s" % job_name)
        cwd_before = os.getcwd()
        os.chdir(run_folder)
        
        job_obj = mdb.Job(
            name=job_name, 
            model=model_name, 
            description='Impact simulation', 
            type=ANALYSIS, 
            explicitPrecision=SINGLE, 
            numCpus=NUM_CPUS, 
            numDomains=NUM_CPUS, 
            multiprocessingMode=THREADS)
        
        job_obj.setValues(historyPrint=ON, echoPrint=ON, modelPrint=ON, 
                         contactPrint=ON)
        job_obj.submit(consistencyChecking=OFF)
        log("Job submitted - waiting for completion...")
        job_obj.waitForCompletion()
        log("Job completed")

        # =====================================================================
        #              POST-PROCESSING
        # =====================================================================
        
        # Check job status
        job_status = "FAILED"
        try:
            sta_text = ""
            if os.path.exists(job_name + ".sta"):
                with open(job_name + ".sta", 'r') as f: 
                    sta_text = f.read()
                if "COMPLETED SUCCESSFULLY" in sta_text:
                    job_status = "COMPLETED"
                elif "THE ANALYSIS HAS BEEN TERMINATED" in sta_text:
                    job_status = "TERMINATED"
        except Exception:
            pass

        solver_tail = tail_of_files(job_name, lines=120)
        odb_path_abs = os.path.join(run_folder, job_name + ".odb") if os.path.exists(
            os.path.join(run_folder, job_name + ".odb")) else ""
        # Portable, device-independent path for the CSV: relative to the results
        # folder (Abaqus_OUT) and forward-slashed so it resolves on any machine.
        odb_path_rel = (os.path.join(run_id, job_name + ".odb").replace(os.sep, '/')
                        if odb_path_abs else "")
        
        img_before_abs, img_after_abs = "", ""
        img_before_rel, img_after_rel = "", ""
        max_mises = ""
        max_peeq = ""
        max_defl = ""

        # =====================================================================
        #         VISUALIZATION AND DATA EXTRACTION
        # =====================================================================
        if job_status == "COMPLETED" and odb_path_abs:
            try:
                log("Opening ODB for post-processing")
                odb = visualization.openOdb(path=odb_path_abs, readOnly=True)
                
                # Create viewport for visualization
                vp = session.Viewport(name='Viewport_' + run_id, origin=(0, 0), 
                                     width=1200, height=1200)
                vp.setValues(displayedObject=odb)
                vp.view.setValues(session.views['Front'])
                vp.view.fitView()
                
                # Set rendering options with MESH VISIBLE
                vp.odbDisplay.commonOptions.setValues(
                    visibleEdges=FEATURE,     # Feature edges visible
                    renderStyle=SHADED        # Shaded rendering
                )
                
                # Disable annotations for cleaner images
                vp.viewportAnnotationOptions.setValues(
                    triad=OFF, 
                    legend=OFF, 
                    title=OFF, 
                    state=OFF
                )

                # =================================================================
                #     HIDE PROJECTILE - SHOW ONLY PLATE
                # =================================================================
                try:
                    log("Hiding projectile from viewport")
                    insts = odb.rootAssembly.instances
                    plate_key = [k for k in insts.keys() if 'PLATE' in k.upper()][0]
                    leaf = dgo.LeafFromPartInstance(partInstanceName=(plate_key,))
                    vp.odbDisplay.displayGroup.replace(leaf=leaf)
                    log("Projectile hidden - showing only: %s" % plate_key)
                except Exception as e_hide:
                    log("Instance-hiding failed (non-fatal): %s" % e_hide)

                # =================================================================
                #     SET CONSISTENT VIEW AND SCALE FOR BOTH IMAGES
                # =================================================================
                # Navigate to last frame to determine the view bounds
                last_step_name = list(odb.steps.keys())[-1]
                frames = odb.steps[last_step_name].frames
                last_index = len(frames) - 1
                
                log("Setting up consistent view for both images")
                vp.odbDisplay.setFrame(step=last_step_name, frame=last_index)
                vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
                vp.view.fitView()
                
                # Store the view parameters (camera position, scale, etc.)
                try:
                    stored_view = {
                        'cameraPosition': vp.view.cameraPosition,
                        'cameraUpVector': vp.view.cameraUpVector,
                        'cameraTarget': vp.view.cameraTarget,
                        'viewOffsetX': vp.view.viewOffsetX,
                        'viewOffsetY': vp.view.viewOffsetY
                    }
                    log("View parameters stored for consistent scaling")
                except Exception as e_view:
                    log("Warning: Could not store all view parameters: %s" % e_view)
                    stored_view = None

                # =================================================================
                #     BEFORE IMAGE: BLUE COLORED (Contours on deformed at frame 0)
                # =================================================================
                try:
                    log("Generating BEFORE image (frame 0 with stress contours - BLUE)")
                    vp.odbDisplay.setFrame(step=0, frame=0)
                    
                    # Use CONTOURS_ON_DEF to get blue color scheme
                    vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
                    
                    # Set primary variable to von Mises stress for blue contours
                    try:
                        vp.odbDisplay.setPrimaryVariable(
                            variableLabel='S', 
                            outputPosition=INTEGRATION_POINT, 
                            refinement=(INVARIANT, 'Mises'))
                    except Exception:
                        pass
                    
                    # Make mesh edges visible in before image
                    vp.odbDisplay.commonOptions.setValues(
                        visibleEdges=ALL,              # All edges visible
                        renderStyle=SHADED,            # Shaded with contours
                        edgeColorWireHide='#000000',   # Black edges
                        edgeColorFillShade='#303030',  # Dark gray for shaded edges
                        edgeLineThickness=VERY_THIN    # Thin lines for clarity
                    )
                    
                    # Apply the stored view to maintain consistent scale
                    if stored_view:
                        try:
                            vp.view.setValues(**stored_view)
                            log("Applied stored view to before image")
                        except Exception as e_sv:
                            log("Could not apply stored view, using fitView: %s" % e_sv)
                            vp.view.fitView()
                    else:
                        vp.view.fitView()
                    
                    log("Before image configured successfully")
                    
                except Exception as e_before:
                    log("Before-image setup warning: %s" % e_before)
                    log(traceback.format_exc())

                # Save BEFORE image
                if EXPORT_IMAGES:
                    try:
                        img_before_abs = os.path.join(run_folder, job_name + "_before.png")
                        img_before_rel = os.path.join(run_id, job_name + "_before.png")
                        session.printOptions.setValues(vpBackground=OFF, vpDecorations=OFF)
                        session.pngOptions.setValues(imageSize=(1200, 1200))
                        session.printToFile(fileName=img_before_abs, format=PNG, 
                                           canvasObjects=(vp,))
                        log("Saved BEFORE image (BLUE): %s" % img_before_abs)
                        
                        # Verify file was created
                        if os.path.exists(img_before_abs):
                            log("BEFORE image file verified: size=%d bytes" % os.path.getsize(img_before_abs))
                        else:
                            log("WARNING: BEFORE image file not found after save!")
                            img_before_abs = ""
                            img_before_rel = ""
                    except Exception as e_save_before:
                        log("ERROR saving BEFORE image: %s" % str(e_save_before))
                        log(traceback.format_exc())
                        img_before_abs = ""
                        img_before_rel = ""

                # =================================================================
                #     AFTER IMAGE: Deformed with stress contours and mesh
                # =================================================================
                try:
                    log("Generating AFTER image (deformed with stress contours)")
                    vp.odbDisplay.setFrame(step=last_step_name, frame=last_index)
                    
                    # Show contours on deformed shape
                    vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
                    
                    # Set primary variable to von Mises stress
                    try:
                        vp.odbDisplay.setPrimaryVariable(
                            variableLabel='S', 
                            outputPosition=INTEGRATION_POINT, 
                            refinement=(INVARIANT, 'Mises'))
                        log("Primary variable set to von Mises stress")
                    except Exception as e_var:
                        log("Variable setting warning: %s" % e_var)
                    
                    # Ensure mesh edges are visible on deformed plot
                    vp.odbDisplay.commonOptions.setValues(
                        visibleEdges=ALL,           # All edges visible
                        renderStyle=SHADED,         # Shaded with contours
                        edgeColorWireHide='#000000', # Black edges
                        edgeColorFillShade='#202020', # Very dark gray
                        edgeLineThickness=VERY_THIN  # Thin lines
                    )
                    
                    # Apply the same stored view to maintain consistent scale
                    if stored_view:
                        try:
                            vp.view.setValues(**stored_view)
                            log("Applied stored view to after image")
                        except Exception as e_sv2:
                            log("Could not apply stored view, using fitView: %s" % e_sv2)
                            vp.view.fitView()
                    else:
                        vp.view.fitView()
                    
                    log("After image configured successfully")
                    
                except Exception as e_after_setup:
                    log("After-image setup warning: %s" % e_after_setup)
                    log(traceback.format_exc())
                
                if EXPORT_IMAGES:
                    img_after_abs = os.path.join(run_folder, job_name + "_after.png")
                    img_after_rel = os.path.join(run_id, job_name + "_after.png")
                    session.printToFile(fileName=img_after_abs, format=PNG, 
                                       canvasObjects=(vp,))
                    log("Saved AFTER image: %s" % img_after_abs)
                    
                    # Verify file was created
                    if os.path.exists(img_after_abs):
                        log("AFTER image file verified: size=%d bytes" % os.path.getsize(img_after_abs))
                    else:
                        log("WARNING: AFTER image file not found after save!")
                        img_after_abs = ""
                        img_after_rel = ""

                # =================================================================
                #     EXTRACT RESULT DATA FROM LAST FRAME
                # =================================================================
                log("Extracting result data from last frame")
                last_frame = frames[-1]
                fo = last_frame.fieldOutputs
                
                # Maximum von Mises stress
                if 'S' in fo.keys():
                    s_field = fo['S']
                    stress_vals = [v.mises for v in s_field.values if hasattr(v, 'mises')]
                    if stress_vals:
                        max_mises = max(stress_vals)
                        log("Max von Mises stress: %.2f MPa" % max_mises)
                
                # Maximum equivalent plastic strain (PEEQ)
                if 'PEEQ' in fo.keys():
                    pe_field = fo['PEEQ']
                    pe_vals = [v.data for v in pe_field.values]
                    if pe_vals:
                        max_peeq = max(pe_vals)
                        log("Max PEEQ: %.6f" % max_peeq)
                
                # Maximum deflection (displacement magnitude)
                if 'U' in fo.keys():
                    u_field = fo['U']
                    u_vals = [v.magnitude for v in u_field.values]
                    if u_vals:
                        max_defl = max(u_vals)
                        log("Max deflection: %.4f mm" % max_defl)

                odb.close()
                vp.setValues(displayedObject=None)
                log("ODB closed, extraction complete")

                # Reached here => COMPLETED status + ODB opened + data read OK
                run_ok = True

            except Exception as e_odb:
                log("Post-processing error: %s" % str(e_odb))
                log(traceback.format_exc())
                solver_tail += "\nPost-processing error: %s\n" % str(e_odb)
                run_ok = False
                # Release the ODB/viewport so the folder can be deleted on discard
                try: vp.setValues(displayedObject=None)
                except Exception: pass
                try: odb.close()
                except Exception: pass
        else:
            log("Job did not complete successfully: status=%s" % job_status)
            run_ok = False

        # Return to original working directory FIRST, so a discarded folder is
        # not the current directory when we try to delete it (Windows lock).
        try:
            os.chdir(cwd_before)
        except Exception:
            pass

        if run_ok:
            # =================================================================
            #              WRITE RESULTS TO CSV  (valid run only)
            # =================================================================
            row = [
                run_id, split_tag, timestamp, job_name,
                plate_material_name, PLATE_L, PLATE_W, PLATE_T,
                'sphere', round(r, 6), PROJ_DENSITY, round(mass, 6),
                round(x, 4), round(y, 4), round(z0, 4),
                round(v_kmh, 4), round(v_ms, 6), round(v_mms, 6),
                round(KE, 6),
                plate_elem_count, plate_node_count, proj_elem_count, proj_node_count,
                job_status, round(time.time() - t_start, 3),
                (round(max_mises, 2) if max_mises != "" else ""),
                (round(max_peeq, 6) if max_peeq != "" else ""),
                (round(max_defl, 4) if max_defl != "" else ""),
                img_before_rel, img_after_rel, odb_path_rel, solver_tail
            ]
            csv_writer.writerow(row)
            csv_file.flush()
            success_count += 1
            log("VALID run kept as %s (%d/%d) - CSV row written"
                % (run_id, success_count - (START_RUN - 1), N_TARGET))

            # =============================================================
            #     CLEANUP: KEEP BEFORE/AFTER IMAGES, .ODB, AND LOG
            # =============================================================
            # Remove only solver scratch files; the .odb is PRESERVED.
            try:
                keep_files = []
                if img_before_abs:
                    keep_files.append(img_before_abs)
                if img_after_abs:
                    keep_files.append(img_after_abs)
                if odb_path_abs and os.path.exists(odb_path_abs):
                    keep_files.append(odb_path_abs)
                job_log_path = os.path.join(run_folder, job_name + ".log")
                if os.path.exists(job_log_path):
                    keep_files.append(job_log_path)
                _cleanup_run_folder(run_folder, keep_files)
                log("Cleanup completed for %s" % run_id)
            except Exception as e_clean:
                log("Cleanup error (non-fatal): %s" % e_clean)
        else:
            # =================================================================
            #   DISCARD failed/aborted run: NO CSV row, remove the whole folder
            # =================================================================
            log("DISCARDING attempt %d (run_id %s, status=%s) - "
                "removing folder, no CSV row" % (attempt, run_id, job_status))
            try:
                if os.path.exists(run_folder):
                    shutil.rmtree(run_folder)
                    log("Discarded folder removed: %s" % run_id)
            except Exception as e_rm:
                log("Could not fully remove discarded folder %s: %s" % (run_id, e_rm))

        # =====================================================================
        #   RELEASE THE CAE MODEL / JOB / VIEWPORT  (memory-leak fix)
        # =====================================================================
        # Every model carries a 240,000-element mesh. The delete at model
        # creation only removes a model of the SAME name, so each *successful*
        # run left its model resident in the mdb. After ~200 runs the CAE
        # process exhausted memory and could no longer spawn a solver: job
        # submission then returned status=FAILED within the same second.
        # Observed in the 1000-run low-energy batch -- 207 valid runs, then
        # 3,909 consecutive instant failures. Releasing the per-run objects
        # keeps CAE's footprint flat for arbitrarily long campaigns.
        try:
            if model_name in mdb.models.keys():
                del mdb.models[model_name]
        except Exception as e_free:
            log("Model release warning (non-fatal): %s" % e_free)
        try:
            if job_name in mdb.jobs.keys():
                del mdb.jobs[job_name]
        except Exception as e_free:
            log("Job release warning (non-fatal): %s" % e_free)
        try:
            vp_name = 'Viewport_' + run_id
            if vp_name in session.viewports.keys():
                del session.viewports[vp_name]
        except Exception as e_free:
            log("Viewport release warning (non-fatal): %s" % e_free)

    except Exception as e_main:
        # =====================================================================
        #     ERROR HANDLING - DISCARD the attempt (no CSV row, remove folder)
        # =====================================================================
        # A build/solver exception means this attempt produced no valid data, so
        # it is discarded entirely and the while-loop will re-run with new params.
        log("ERROR on attempt %d (%s): %s"
            % (attempt, run_id if 'run_id' in locals() else '?', str(e_main)))
        log(traceback.format_exc())

        # Step out of the run folder before deleting it
        try:
            if 'cwd_before' in locals():
                os.chdir(cwd_before)
        except Exception:
            pass

        # Remove the errored folder entirely - nothing valid to keep
        try:
            if 'run_folder' in locals() and os.path.exists(run_folder):
                shutil.rmtree(run_folder)
                log("Discarded errored attempt folder: %s" % run_id)
        except Exception as e_rm:
            log("Could not remove errored folder: %s" % e_rm)

# =============================================================================
#                          SCRIPT COMPLETION
# =============================================================================
csv_file.close()
for p, h in _log_handles:
    try: 
        h.close()
    except: 
        pass

_n_valid = success_count - (START_RUN - 1)
log("=" * 60)
log("BATCH COMPLETE: %d/%d valid runs in %d attempts (%d discarded)"
    % (_n_valid, N_TARGET, attempt, attempt - _n_valid))
if _n_valid < N_TARGET:
    log("WARNING: target not reached - raise MAX_ATTEMPT_FACTOR or check why runs abort")
log("Results saved to: %s" % csv_path)
log("=" * 60)