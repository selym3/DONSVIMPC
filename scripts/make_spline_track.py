import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import sys

from argparse import ArgumentParser

# TODO output track must be closed
# TODO config file must include start and goal


# Quadratic Bezier curve to get curvature
def main(outfile: str, num_points: int, preview: bool):
    # waypoints = np.array([[1, 1], [3, 1], [4, 4], [2, 2], [1, 1]])
    waypoints = np.array([[1, 1], [3, 1], [4, 4]])
    s = np.linspace(0, 1, len(waypoints))
    x = UnivariateSpline(s, np.array(waypoints[:, 0]), k=2)
    y = UnivariateSpline(s, np.array(waypoints[:, 1]), k=2)

    def curvature(t):
        dx = x.derivative()
        ddx = dx.derivative()
        dy = y.derivative()
        ddy = dy.derivative()
        num = dx(t) * ddy(t) - dy(t) * ddx(t)
        denom = np.power(dx(t) ** 2 + dy(t) ** 2, 3 / 2)
        return num / denom

    t = np.linspace(0, 1, num_points)

    if preview:
        _, axs = plt.subplots(nrows=2, layout="constrained")
        axs[0].scatter(x(t), y(t))
        axs[0].set_title("track center")
        axs[0].set_aspect("equal")
        axs[1].plot(t, curvature(t))
        axs[1].set_title("curvature")
        plt.show()
    else:
        pts = np.stack([x(t), y(t)], axis=1)
        np.savez(
            outfile,
            # Need to use pts so map_coords.py initializes self.rho
            pts=np.stack([x(t), y(t)], axis=1),
            curvature=curvature(t),
            # Needed for visualization
            X_in=pts[:-1, 0],
            Y_in=pts[:-1, 1],
            X_out=pts[1:, 0],
            Y_out=pts[1:, 1],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--outfile")
    parser.add_argument("-n", "--num_points", default=100)
    parser.add_argument("--preview", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    assert args.outfile or args.preview
    main(args.outfile, args.num_points, args.preview)
    # Recommended run:
    # python make_spline_track.py -o ../robot_planning/environment/dynamics/autorally_dynamics/spline_track.npz
