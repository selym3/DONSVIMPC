import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser

def main(outfile: str, num_points: int, preview: bool):
    # 1. Define a set of waypoints that form a loop.
    # The last point should NOT be a duplicate of the first; 
    # the periodic spline handles the closure automatically.
    waypoints = np.array([
        [1.0, 1.0],
        [4.0, 1.0],
        [5.0, 3.0],
        [3.0, 5.0],
        [0.0, 4.0]
    ])
    
    # Track parameters
    track_width = 3.0
    
    # 2. Setup periodic spline
    # We close the loop by appending the first waypoint to the end for the spline fit
    # and using bc_type='periodic'
    data_points = np.vstack((waypoints, waypoints[0]))
    # Create an evaluation parameter t based on point distance
    distances = np.sqrt(np.sum(np.diff(data_points, axis=0)**2, axis=1))
    t_steps = np.concatenate(([0], np.cumsum(distances)))
    t_max = t_steps[-1]
    
    # make_interp_spline with k=3 (cubic) for smooth curvature
    spline = make_interp_spline(t_steps, data_points, k=3, bc_type='periodic')

    # 3. Generate the centerline points
    # We generate num_points. To work with your MapCA wrap-around logic, 
    # pts[0] and pts[-1] should be the same point to close the geometry.
    t_eval = np.linspace(0, t_max, num_points, endpoint=False)
    pts = spline(t_eval) # Centerline (N, 2)

    # 4. Calculate Derivatives for Curvature and Boundaries
    # First derivative (tangent)
    deriv = spline.derivative(1)(t_eval)
    dx = deriv[:, 0]
    dy = deriv[:, 1]
    
    # Second derivative
    deriv2 = spline.derivative(2)(t_eval)
    ddx = deriv2[:, 0]
    ddy = deriv2[:, 1]

    # Curvature calculation: kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^(1.5)
    mag_sq = dx**2 + dy**2
    curvature = (dx * ddy - dy * ddx) / np.power(mag_sq, 1.5)

    # 5. Calculate Boundary Offsets
    # Unit normal vector is (-dy, dx) / magnitude
    mag = np.sqrt(mag_sq)
    nx = -dy / mag
    ny = dx / mag

    
    # Offset points for inner and outer boundaries
    X_in = pts[:, 0] + nx * (track_width / 2)
    Y_in = pts[:, 1] + ny * (track_width / 2)
    X_out = pts[:, 0] - nx * (track_width / 2)
    Y_out = pts[:, 1] - ny * (track_width / 2)

    if preview:
        fig, axs = plt.subplots(nrows=2, figsize=(8, 10), layout="constrained")
        
        # Plot Track
        axs[0].plot(pts[:, 0], pts[:, 1], 'r--', label="Centerline")
        axs[0].plot(X_in, Y_in, 'k', label="Inner Boundary")
        axs[0].plot(X_out, Y_out, 'k', label="Outer Boundary")
        axs[0].set_aspect("equal")
        axs[0].legend()
        axs[0].set_title("Closed Spline Track")
        
        # Plot Curvature
        axs[1].plot(t_eval, curvature)
        axs[1].set_title("Curvature ($\kappa$) along track")

        # p = pts
        # print(p[0], p[-1])

        plt.show()
    else:
        # Save in the format expected by MapCA and AutorallyMatplotlibRenderer
        np.savez(
            outfile,
            pts=pts,
            curvature=curvature,
            X_in=X_in,
            Y_in=Y_in,
            X_out=X_out,
            Y_out=Y_out
        )
        print(f"Saved track with {num_points} points to {outfile}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--outfile")
    parser.add_argument("-n", "--num_points", type=int, default=200)
    parser.add_argument("--preview", action="store_true", default=False)
    args = parser.parse_args()
    
    if args.outfile or args.preview:
        main(args.outfile, args.num_points, args.preview)