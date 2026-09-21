"""Retiming that preserves path geometry and reports actual feasible speed.

Limits are conservative translational reference checks, not a proof of full
actuator feasibility (drag, torque and tracking corrections need headroom).
"""
import numpy as np
from scipy.interpolate import BPoly, CubicSpline
from scipy.ndimage import gaussian_filter1d


def cruise_phase(u, ramp_fraction=.15):
    """Unit distance with smooth velocity ramps and a constant-speed middle."""
    r = ramp_fraction
    u = np.asarray(u)
    x = np.clip(u / r, 0., 1.)
    y = np.clip((1. - u) / r, 0., 1.)
    integral = lambda z: 2.5*z**4 - 3*z**5 + z**6
    smooth = lambda z: 10*z**3 - 15*z**4 + 6*z**5
    phase = np.where(u < r, r*integral(x),
                     np.where(u > 1-r, 1-r-r*integral(y), u-r/2)) / (1-r)
    rate = np.where(u < r, smooth(x), np.where(u > 1-r, smooth(y), 1.)) / (1-r)
    accel = np.where(u < r, 30*x**2*(1-x)**2/r,
                     np.where(u > 1-r, -30*y**2*(1-y)**2/r, 0.)) / (1-r)
    return phase, rate, accel


def retime_reference(position, velocity, acceleration, dt, *, speed_scale=1.,
                     time_profile='quintic', max_speed=5., max_acceleration=8.,
                     max_jerk=40., max_thrust_ratio=1.5, min_vertical_thrust_ratio=.45):
    if time_profile not in ('quintic', 'cruise'):
        raise ValueError('time_profile must be quintic or cruise')
    if not np.isfinite(speed_scale) or speed_scale <= 0:
        raise ValueError('speed_scale must be finite and positive')
    source_duration = (len(position)-1)*dt
    source_time = np.arange(len(position))*dt
    temporal = BPoly.from_derivatives(source_time, list(zip(position, velocity, acceleration)))
    if time_profile == 'cruise':
        # Dense arc-length coordinates remove the long quintic time easing
        # without shrinking the spatial path. Drop repeated hover/end points.
        dense = temporal(np.linspace(0., source_duration, 4001))
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        keep = np.r_[True, np.diff(distance) > 1e-9]
        if distance[-1] > 1e-6:
            arc_geometry = CubicSpline(distance[keep] / distance[-1], dense[keep], axis=0)
            arc = np.linspace(0., 1., 1001)
            # Uniform knots avoid ill-conditioned derivatives at nearly
            # coincident zero-speed endpoints of float32 dataset curves.
            arc_geometry = CubicSpline(arc, arc_geometry(arc), axis=0)
            d1, d2 = arc_geometry(arc, 1), arc_geometry(arc, 2)
            curvature = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1), 1e-9)**3
            cruise_speed = distance[-1]*speed_scale/(source_duration*.85)
            # Allocate more time locally at sharp corners instead of slowing
            # the whole curve to its tightest bend's constant-speed limit.
            weight = (1+(cruise_speed**2*curvature/3.)**2)**.25
            weight = gaussian_filter1d(weight, sigma=10., mode='nearest')
            clock = np.r_[0., np.cumsum((weight[:-1]+weight[1:])/2)]
            clock /= clock[-1]
            geometry = CubicSpline(clock, arc_geometry(arc), axis=0)
        else:
            geometry = None

    def sample(duration, step):
        t = np.linspace(0., duration, int(round(duration/step))+1)
        if time_profile == 'quintic':
            ratio = source_duration / duration
            return tuple(temporal(t*ratio, nu=k) * ratio**k for k in range(3))
        if geometry is None:
            return np.tile(position[0], (len(t), 1)), np.zeros((len(t), 3)), np.zeros((len(t), 3))
        s, sd, sdd = cruise_phase(t/duration)
        d1, d2 = geometry(s, 1), geometry(s, 2)
        return geometry(s), d1*(sd/duration)[:, None], (
            d2*(sd/duration)[:, None]**2 + d1*(sdd/duration**2)[:, None])

    requested_duration = source_duration / speed_scale
    duration = np.ceil(requested_duration/dt - 1e-9)*dt
    # Check at twice the control frequency; stretch time rather than geometry.
    for _ in range(60):
        pos, vel, acc = sample(duration, dt/2)
        peak_v = np.linalg.norm(vel, axis=1).max()
        peak_a = np.linalg.norm(acc, axis=1).max()
        peak_j = np.linalg.norm(np.gradient(acc, dt/2, axis=0), axis=1).max()
        specific_thrust = acc + [0., 0., 9.8]
        peak_thrust = np.linalg.norm(specific_thrust, axis=1).max()/9.8
        min_vertical = specific_thrust[:, 2].min()/9.8
        if (peak_v <= max_speed and peak_a <= max_acceleration and peak_j <= max_jerk
                and peak_thrust <= max_thrust_ratio and min_vertical >= min_vertical_thrust_ratio):
            break
        stretch = max(1.025, peak_v/max_speed, np.sqrt(peak_a/max_acceleration),
                      np.cbrt(peak_j/max_jerk))
        duration = np.ceil(duration*stretch/dt)*dt
    else:
        raise ValueError('Could not retime path within reference dynamics limits')
    # Some very tight curves already have a better speed allocation in the
    # original schedule. Do not turn a requested speedup into a long crawl:
    # keep the faster feasible candidate and disclose the fallback explicitly.
    if time_profile == 'cruise' and duration > requested_duration*1.10:
        original = retime_reference(position, velocity, acceleration, dt,
            speed_scale=speed_scale, time_profile='quintic', max_speed=max_speed,
            max_acceleration=max_acceleration, max_jerk=max_jerk,
            max_thrust_ratio=max_thrust_ratio, min_vertical_thrust_ratio=min_vertical_thrust_ratio)
        if original[3]['reference_duration'] < duration:
            original[3].update(time_profile='cruise', effective_time_profile='quintic_fallback',
                               cruise_fallback=True)
            return original
    arrays = sample(duration, dt)
    metadata = dict(time_profile=time_profile, effective_time_profile=time_profile, cruise_fallback=False, requested_speed_scale=float(speed_scale),
                    effective_speed_scale=float(source_duration/duration),
                    reference_duration=float(duration),
                    reference_mean_speed=float(np.mean(np.linalg.norm(arrays[1], axis=1))),
                    reference_peak_speed=float(peak_v), reference_peak_acceleration=float(peak_a),
                    reference_peak_jerk=float(peak_j), reference_peak_thrust_ratio=float(peak_thrust),
                    speed_limited=bool(duration > requested_duration + dt + 1e-9))
    return *arrays, metadata
