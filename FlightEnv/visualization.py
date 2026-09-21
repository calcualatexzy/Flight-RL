"""Trajectory overlays for the PyBullet viewer and evaluation figures."""
import numpy as np
import pybullet as p

REFERENCE = (0.12, 0.82, 0.94)
ACTUAL = (1.0, 0.43, 0.16)
START = (0.35, 0.91, 0.59)
FINISH = (0.80, 0.55, 1.0)
TEXT = (0.90, 0.94, 1.0)


class TrajectoryViewer:
    """Visual-only objects: no collisions, forces or changes to simulation state."""

    def __init__(self, client):
        self.client = client
        self.items = []
        self.bodies = []
        self.hud = {}
        self.paused = False
        self.reference = None
        self.top_view = False

    def clear(self):
        for item in self.items + list(self.hud.values()):
            if item >= 0:
                p.removeUserDebugItem(item, physicsClientId=self.client)
        for body in self.bodies:
            p.removeBody(body, physicsClientId=self.client)
        self.items.clear()
        self.bodies.clear()
        self.hud.clear()

    def _line(self, start, end, color, width=1):
        item = p.addUserDebugLine(start, end, color, width, lifeTime=0,
                                  physicsClientId=self.client)
        self.items.append(item)
        return item

    def _marker(self, position, color, radius):
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                     rgbaColor=[*color, 1.], physicsClientId=self.client)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=shape, basePosition=position,
                                 physicsClientId=self.client)
        self.bodies.append(body)
        return body

    def reset(self, reference, position, ground_id, duration):
        self.reference = np.asarray(reference)
        self.duration = duration
        self.position = np.asarray(position)
        self.target = self.reference[0].copy()
        self.positions = [self.position.copy()]
        self.last_drawn_position = self.position.copy()
        self.low = np.minimum(self.reference.min(axis=0), self.position)
        self.high = np.maximum(self.reference.max(axis=0), self.position)
        self.elapsed = self.error = 0.
        self.last_draw = -np.inf
        self.done = self.paused = False
        self.status = 'FLYING'
        self.top_view = False
        direction = self.reference[-1, :2] - self.reference[0, :2]
        self.camera_yaw = float(np.rad2deg(np.arctan2(direction[1], direction[0]))) if np.linalg.norm(direction) > .01 else 35.
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client)
        p.changeVisualShape(ground_id, -1, rgbaColor=[.12, .16, .22, 1.],
                            textureUniqueId=-1, physicsClientId=self.client)
        # Limit reference drawing cost, keeping both endpoints even on long runs.
        indices = np.unique(np.linspace(0, len(reference)-1, min(241, len(reference))).astype(int))
        for a, b in zip(indices[:-1], indices[1:]):
            self._line(reference[a], reference[b], REFERENCE, 3)
        # Faint ground projection and grid make height/depth easier to judge.
        floor = -0.045
        for i, (a, b) in enumerate(zip(indices[:-1], indices[1:])):
            if i % 2 == 0:
                self._line([*reference[a, :2], floor], [*reference[b, :2], floor],
                           (.13, .32, .39), 1)
        grid_step = max(1., np.ceil(np.max(self.high[:2] - self.low[:2]) / 12))
        lo = np.floor((self.low[:2] - 1) / grid_step) * grid_step
        hi = np.ceil((self.high[:2] + 1) / grid_step) * grid_step
        for x in np.arange(lo[0], hi[0] + grid_step/2, grid_step):
            self._line([x, lo[1], floor], [x, hi[1], floor], (.22, .28, .35))
        for y in np.arange(lo[1], hi[1] + grid_step/2, grid_step):
            self._line([lo[0], y, floor], [hi[0], y, floor], (.22, .28, .35))
        self._marker(reference[0], START, .08)
        self._marker(reference[-1], FINISH, .09)
        self.target_body = self._marker(self.target, REFERENCE, .075)
        self.drone_body = self._marker(self.position, ACTUAL, .055)
        for label, point, color in [('START', reference[0], START), ('FINISH', reference[-1], FINISH)]:
            item = p.addUserDebugText(label, point + [0, 0, .22], color, textSize=1.3,
                                      physicsClientId=self.client)
            self.items.append(item)
        self.error_line = self._line(self.position, self.target, TEXT, 1)
        self.fit_camera()
        self.refresh()

    def fit_camera(self):
        self.camera_center = (self.low + self.high) / 2
        self.camera_radius = max(.7, np.linalg.norm(self.high - self.low) / 2)
        pitch = -89 if self.top_view else -35
        camera = p.getDebugVisualizerCamera(physicsClientId=self.client)
        projection = np.asarray(camera[3]).reshape(4, 4, order='F')
        tan_x = 1 / projection[0, 0] if projection[0, 0] > 0 else .9
        tan_y = 1 / projection[1, 1] if projection[1, 1] > 0 else .56
        view = np.asarray(p.computeViewMatrixFromYawPitchRoll(
            self.camera_center, 1., self.camera_yaw, pitch, 0, 2)).reshape(4, 4, order='F')
        points = np.vstack((self.reference, self.positions))
        local = (points - self.camera_center) @ view[:3, :3].T
        # Fit the projected bounds, leaving space above for the legend/HUD.
        distance = max(2.2, np.max(local[:, 2] + np.abs(local[:, 0]) / (tan_x * .85)),
                       np.max(local[:, 2] + np.abs(local[:, 1]) / (tan_y * .63))) + .3
        p.resetDebugVisualizerCamera(float(distance), self.camera_yaw, pitch,
                                     self.camera_center, physicsClientId=self.client)

    def update(self, position, target, elapsed, info, done=False):
        self.position = np.asarray(position)
        self.target = np.asarray(target)
        self.positions.append(self.position.copy())
        self.low = np.minimum(self.low, self.position)
        self.high = np.maximum(self.high, self.position)
        self.elapsed, self.error, self.done = elapsed, info['position_error'], done
        if done:
            self.status = ('COLLISION' if info['collision'] else
                           'OUT OF BOUNDS' if info['out_of_bounds'] else 'COMPLETED')
        if elapsed - self.last_draw < 1/30 and not done:
            return
        self.last_draw = elapsed
        self._line(self.last_drawn_position, self.position, ACTUAL, 5)
        self.last_drawn_position = self.position.copy()
        for body, point in ((self.target_body, self.target), (self.drone_body, self.position)):
            p.resetBasePositionAndOrientation(body, point, [0, 0, 0, 1], physicsClientId=self.client)
        p.addUserDebugLine(self.position, self.target, TEXT, lineWidth=1,
                           replaceItemUniqueId=self.error_line, physicsClientId=self.client)
        # Expand only when needed, so a drifting drone does not leave the scene.
        radius = max(.7, np.linalg.norm(self.high - self.low) / 2)
        if radius > self.camera_radius * 1.15:
            self.fit_camera()
        self.refresh()

    def refresh(self):
        keys = p.getKeyboardEvents(physicsClientId=self.client)
        if keys.get(ord('f'), 0) & p.KEY_WAS_TRIGGERED:
            self.fit_camera()
        if keys.get(ord('t'), 0) & p.KEY_WAS_TRIGGERED:
            self.top_view = not self.top_view
            self.fit_camera()
        if not self.done and keys.get(ord(' '), 0) & p.KEY_WAS_TRIGGERED:
            self.paused = not self.paused
        camera = p.getDebugVisualizerCamera(physicsClientId=self.client)
        view = np.asarray(camera[2]).reshape(4, 4, order='F')
        projection = np.asarray(camera[3]).reshape(4, 4, order='F')
        if not np.any(projection):  # DIRECT clients have no GUI camera.
            return
        inverse = np.linalg.inv(projection @ view)
        status = 'PAUSED' if self.paused and not self.done else self.status
        labels = [
            ('title', 'FLIGHT-RL  /  TRAJECTORY TRACKING', TEXT, .90, 1.4),
            ('reference', 'CYAN     Reference trajectory + target', REFERENCE, .80, 1.2),
            ('actual', 'ORANGE   Actual flight + drone', ACTUAL, .72, 1.2),
            ('time', f'{status}   {self.elapsed:5.2f} / {self.duration:.1f} s', TEXT, .60, 1.2),
            ('error', f'Position error   {self.error:.3f} m', TEXT, .52, 1.2),
            ('keys', 'F: fit view   T: top view   SPACE: pause', (.63, .72, .84), -.90, 1.0),
        ]
        for key, text, color, y, size in labels:
            # Unproject fixed screen coordinates close to the camera, forming a
            # readable HUD that stays in place when orbiting or zooming.
            point = inverse @ np.array([-.94, y, -.9, 1.])
            point = point[:3] / point[3]
            self.hud[key] = p.addUserDebugText(text, point, color, textSize=size,
                                              replaceItemUniqueId=self.hud.get(key, -1),
                                              physicsClientId=self.client)


def save_tracking_plot(tracks, references, results, path):
    """Compare the entire planned path with the flown portion, including crashes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = {'figure.facecolor': '#0b1220', 'axes.facecolor': '#111e30',
             'axes.edgecolor': '#43546c', 'axes.labelcolor': '#b7c6db',
             'text.color': '#e8eff9', 'xtick.color': '#9eafc8', 'ytick.color': '#9eafc8',
             'grid.color': '#34465c', 'font.size': 10, 'savefig.facecolor': '#0b1220'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 8))
        grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1), hspace=.34, wspace=.2)
        ax3d = fig.add_subplot(grid[:, 0], projection='3d')
        xy = fig.add_subplot(grid[0, 1])
        error = fig.add_subplot(grid[1, 1])
        track, reference = tracks[0], references[0]
        actual = track[:, 1:4]
        planned = reference[:, 1:4]
        for ax in (ax3d, xy):
            dims = 3 if ax is ax3d else 2
            ax.plot(*planned[:, :dims].T, color=REFERENCE, linewidth=4.5, linestyle='--', alpha=.8)
            ax.plot(*actual[:, :dims].T, color=ACTUAL, linewidth=2)
            ax.scatter(*planned[0, :dims], color=START, s=55, zorder=5)
            ax.scatter(*planned[-1, :dims], color=FINISH, marker='X', s=75, zorder=5)
            ax.scatter(*actual[-1, :dims], color=ACTUAL, s=55, zorder=5)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.set_pane_color((.067, .118, .188, 1))
        bounds = np.vstack((planned, actual))
        extent = np.maximum(np.ptp(bounds, axis=0), .5)
        ax3d.set_box_aspect(extent)
        ax3d.view_init(elev=28, azim=-55)
        ax3d.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        ax3d.set_title('01 / SPATIAL PATH  -  episode 0', loc='left', pad=18, fontsize=11)
        xy.set(title='02 / TOP VIEW  -  episode 0', xlabel='X (m)', ylabel='Y (m)')
        xy.set_aspect('equal', adjustable='datalim')
        xy.grid(alpha=.4)
        for i, flight in enumerate(tracks[:8]):
            color = ACTUAL if i == 0 else plt.get_cmap('tab10')(i % 10)
            error.plot(flight[:, 0], flight[:, 7], color=color, linewidth=2,
                       label=f'Episode {i}')
        error_title = '03 / POSITION ERROR'
        if len(tracks) > 8:
            error_title += f'  (first 8 of {len(tracks)} episodes)'
        error.set(title=error_title, xlabel='Time (s)', ylabel='Error (m)')
        error.set_xlim(0, max(r[-1, 0] for r in references))
        error.set_ylim(bottom=0)
        error.grid(alpha=.4)
        if len(tracks) > 1:
            error.legend(frameon=False, fontsize=8, ncol=2)
        handles = [Line2D([0], [0], color=REFERENCE, linestyle='--', lw=2.3, label='Reference / full path'),
                   Line2D([0], [0], color=ACTUAL, lw=3, label='Actual flight'),
                   Line2D([0], [0], color=START, marker='o', linestyle='', label='Start'),
                   Line2D([0], [0], color=FINISH, marker='X', linestyle='', label='Finish')]
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.055, .92),
                   frameon=False, ncol=4)
        first = results[0]
        status = 'COMPLETE' if first['truncated'] else 'ENDED EARLY'
        fig.suptitle('REFERENCE  /  FLIGHT COMPARISON', x=.065, ha='left',
                     fontsize=19, fontweight='bold', y=.97)
        fig.text(.065, .04, f'Episode 0  |  {status}  |  RMSE {first["position_rmse"]:.3f} m'
                 f'  |  Flown {track[-1, 0]:.2f} / planned {reference[-1, 0]:.2f} s',
                 color='#b7c6db', fontsize=11)
        fig.subplots_adjust(top=.79, bottom=.14, left=.06, right=.96)
        fig.savefig(path, dpi=170)
        plt.close(fig)


def save_family_gallery(tracks, references, results, path):
    """Show the first evaluated path in each family, without choosing the best."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    selected = {}
    for i, result in enumerate(results):
        selected.setdefault(result['trajectory_family'], i)
    cols = min(4, len(selected))
    rows = (len(selected) + cols - 1) // cols
    with plt.rc_context({'figure.facecolor': '#0b1220', 'axes.facecolor': '#111e30',
                         'text.color': '#e8eff9', 'axes.labelcolor': '#b7c6db',
                         'xtick.color': '#9eafc8', 'ytick.color': '#9eafc8', 'font.size': 9}):
        fig = plt.figure(figsize=(4.5*cols, 4.3*rows+1))
        for panel, (family, i) in enumerate(selected.items()):
            ax = fig.add_subplot(rows, cols, panel+1, projection='3d')
            actual, planned = tracks[i][:, 1:4], references[i][:, 1:4]
            ax.plot(*planned.T, color=REFERENCE, linestyle='--', linewidth=4, alpha=.8)
            ax.plot(*actual.T, color=ACTUAL, linewidth=1.8)
            ax.scatter(*planned[0], color=START, s=22)
            ax.scatter(*planned[-1], color=FINISH, marker='X', s=30)
            bounds = np.vstack((actual, planned))
            ax.set_box_aspect(np.maximum(np.ptp(bounds, axis=0), .5))
            ax.view_init(elev=28, azim=-55)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((.067, .118, .188, 1))
                axis.set_major_locator(MaxNLocator(nbins=3))
            ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            status = 'complete' if results[i]['truncated'] else 'ended early'
            ax.set_title(f"{family} | {status}\nRMSE {results[i]['position_rmse']:.3f} m", pad=8)
        fig.suptitle('TRAJECTORY TRACKING / FIRST EVALUATED PATH OF EACH FAMILY', y=.98, fontsize=15)
        handles = [Line2D([0], [0], color=REFERENCE, linestyle='--', lw=4, label='Reference'),
                   Line2D([0], [0], color=ACTUAL, lw=2, label='Actual flight')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
        fig.subplots_adjust(left=.04, right=.96, top=.87, bottom=.06, hspace=.25, wspace=.12)
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
