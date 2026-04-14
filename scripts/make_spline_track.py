import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from argparse import ArgumentParser

def main(outfile: str, num_points: int, preview: bool):
    # 1. Defined larger, more complex "race-track" waypoints
    # Scaled up roughly 10x from your original for a realistic F10/AutoRally footprint
    waypoints = np.array([
        [0.0, 0.0],    # Start/Finish
        [40.0, 2.0],   # Long high-speed slight curve
        [60.0, 10.0],  # Entry to large sweeper
        [75.0, 30.0],  # Apex of sweeper
        [60.0, 50.0],  # Exit of sweeper
        [30.0, 45.0],  # Mid-field straight
        [20.0, 55.0],  # Technical "chicane" entry
        [10.0, 50.0],  # Technical "chicane" exit
        [-10.0, 40.0], # Hairpin entry
        [-15.0, 20.0], # Hairpin apex
        [-5.0, 10.0],  # Return straight
    ])

    center = waypoints.mean(axis=0).shape
    waypoints -= center
    waypoints *= 0.5

    # 2. Setup periodic spline
    data_points = np.vstack((waypoints, waypoints[0]))
    
    # Parameterize by distance (chordal parameterization)
    distances = np.sqrt(np.sum(np.diff(data_points, axis=0)**2, axis=1))
    t_steps = np.concatenate(([0], np.cumsum(distances)))
    t_max = t_steps[-1]
    
    # Cubic spline with periodic boundary conditions
    spline = make_interp_spline(t_steps, data_points, k=3, bc_type='periodic')

    # 3. Generate the centerline points
    t_eval = np.linspace(0, t_max, num_points, endpoint=False)
    pts = spline(t_eval)

    # 4. Calculate Derivatives for Curvature
    deriv = spline.derivative(1)(t_eval)
    dx, dy = deriv[:, 0], deriv[:, 1]
    
    deriv2 = spline.derivative(2)(t_eval)
    ddx, ddy = deriv2[:, 0], deriv2[:, 1]

    mag_sq = dx**2 + dy**2
    # Curvature formula: (x'y'' - y'x'') / (x'^2 + y'^2)^(1.5)
    curvature = (dx * ddy - dy * ddx) / np.power(mag_sq, 1.5)

    # 5. Calculate Boundary Offsets (Optional visual aid)
    track_width = 3.0  # Total width of 3 meters
    mag = np.sqrt(mag_sq)
    nx = -dy / mag
    ny = dx / mag

    if preview:
        fig, axs = plt.subplots(nrows=2, figsize=(10, 12), layout="constrained")
        
        # Plot Track
        axs[0].plot(pts[:, 0], pts[:, 1], 'r-', label="Centerline", linewidth=2)
        axs[0].plot(waypoints[:, 0], waypoints[:, 1], 'ko', label="Waypoints", alpha=0.5)
        
        # Draw boundaries to visualize scale
        inner = pts + np.column_stack((-dy/mag * 1.5, dx/mag * 1.5))
        outer = pts - np.column_stack((-dy/mag * 1.5, dx/mag * 1.5))
        axs[0].plot(inner[:,0], inner[:,1], 'k--', alpha=0.3)
        axs[0].plot(outer[:,0], outer[:,1], 'k--', alpha=0.3)
        
        axs[0].set_aspect("equal")
        axs[0].legend()
        axs[0].set_title(f"F10 Race Track (Length: {t_max:.2f}m)")
        axs[0].grid(True, linestyle=':', alpha=0.6)
        
        # Plot Curvature
        axs[1].plot(t_eval, curvature, color='blue')
        axs[1].axhline(0, color='black', lw=1)
        axs[1].set_title("Curvature ($\kappa$) - Essential for CBF Training")
        axs[1].set_xlabel("Distance along track (m)")
        axs[1].set_ylabel("1/Radius")

        plt.show()
    else:
        np.savez(outfile, pts=pts, curvature=curvature)
        print(f"Saved track ({t_max:.1f}m long) with {num_points} points to {outfile}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--outfile")
    parser.add_argument("-n", "--num_points", type=int, default=500) # Increased default for longer track
    parser.add_argument("--preview", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.outfile or args.preview:
        main(args.outfile, args.num_points, args.preview)