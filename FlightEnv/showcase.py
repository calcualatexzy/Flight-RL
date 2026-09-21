"""Render measured policy rollouts with PyBullet cameras for web video/GIF."""
import argparse
from dataclasses import dataclass
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
from scipy.spatial.transform import Rotation, Slerp
import torch

ROOT = Path(__file__).resolve().parents[1]
CYAN = (35, 211, 230)
ORANGE = (255, 135, 66)
WHITE = (233, 241, 252)
MUTED = (147, 168, 195)
BG = (9, 17, 30)


@dataclass(frozen=True)
class Demo:
    family: str
    trajectory_id: int
    seed: int
    title: str
    subtitle: str


DEMOS = (
    Demo('figure8', 96, 2028, 'Figure eight', 'Crossing turns in three dimensions'),
    Demo('helix', 153, 2035, 'Spiral ascent', 'Two full turns with continuous ascent'),
    Demo('vertical_loop', 224, 2032, 'Vertical loop', 'A complete loop through the vertical plane'),
    Demo('lissajous', 269, 2137, 'Lissajous', 'Coupled oscillations and repeated crossings'),
)


def collect_rollout(model, demo, directory, *, config_overrides=None,
                    success_rmse=.35, success_max_error=1.):
    from main import create_env
    config = dict(model.flight_env_config, dataset_split='test', difficulty=1.,
                  mode='trajectory', trajectory_family=demo.family, render_mode=None)
    config.update(config_overrides or {})
    with create_env(config) as env:
        obs, info = env.reset(seed=demo.seed, options={'trajectory_id': demo.trajectory_id})
        if info['trajectory_family'] != demo.family:
            raise ValueError('Curated trajectory IDs require the original V3 test dataset with 32 paths per family')
        reference = env.state_goal[:, [0, 2, 4]].copy()
        reference_velocity = env.state_goal[:, [1, 3, 5]].copy()
        positions = [np.array(env.quadrotor.pos)]
        quaternions = [np.array(env.quadrotor.quat)]
        velocities = [np.array(env.quadrotor.vel)]
        actions, controls, errors, rewards = [], [], [], []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            positions.append(np.array(env.quadrotor.pos))
            quaternions.append(np.array(env.quadrotor.quat))
            velocities.append(np.array(env.quadrotor.vel))
            actions.append(np.array(action))
            controls.append(env.last_control.copy())
            errors.append(info['position_error'])
            rewards.append(reward)
            if terminated or truncated:
                break
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        max_error = float(max(errors))
        completed = bool(truncated and not terminated)
        success = bool(completed and rmse <= success_rmse and max_error <= success_max_error)
        if not success:
            raise RuntimeError(f'{demo.family} did not pass tracking criteria; refusing to label it successful')
        trace = dict(time=np.arange(len(positions))*env._time_step,
                     position=np.asarray(positions), quaternion=np.asarray(quaternions),
                     reference=reference, reference_velocity=reference_velocity,
                     velocity=np.asarray(velocities), applied_control=np.asarray(controls),
                     action=np.asarray(actions), error=np.asarray(errors))
        metrics = dict(family=demo.family, title=demo.title, subtitle=demo.subtitle,
                       trajectory_id=demo.trajectory_id, seed=demo.seed, split='test', difficulty=1.,
                       completed=completed, tracking_success=success, steps=len(errors),
                       duration_seconds=float(trace['time'][-1]), position_rmse_m=rmse,
                       max_position_error_m=max_error,
                       peak_reference_speed_m_s=float(np.linalg.norm(reference_velocity,axis=1).max()),
                       mean_reference_speed_m_s=float(np.linalg.norm(reference_velocity[1:],axis=1).mean()),
                       peak_actual_speed_m_s=float(np.linalg.norm(velocities[1:],axis=1).max()),
                       mean_actual_speed_m_s=float(np.linalg.norm(velocities[1:],axis=1).mean()),
                       reward=float(sum(rewards)), environment=config,
                       acceptance=dict(rmse_m=success_rmse, max_error_m=success_max_error, must_complete=True),
                       model_timesteps=int(model.num_timesteps))
        metrics.update({key: info[key] for key in ('time_profile', 'requested_speed_scale',
                        'effective_speed_scale', 'reference_duration', 'speed_limited',
                        'effective_time_profile', 'cruise_fallback') if key in info})
        residual = config.get('profile') == 'residual'
        zero_residual = residual and bool(np.all(np.abs(trace['action']) <= 1e-12))
        metrics['control_mode'] = ('geometric_feedforward_zero_residual' if zero_residual else
                                   'geometric_feedforward_plus_rl_residual' if residual else 'ppo_direct_control')
        metrics['controller_label'] = ('GEOMETRIC / ZERO RESIDUAL' if zero_residual else
                                       'GEOMETRIC + RL RESIDUAL' if residual else 'FULL DIFFICULTY / PPO')
        np.savez_compressed(directory/f'{demo.family}_rollout.npz', **trace)
        (directory/f'{demo.family}_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        return trace, metrics


def tube_mesh(starts, ends, radius, sides=8):
    """Joined visual cylinders. No collision shapes and no physics are involved."""
    starts, ends = np.asarray(starts), np.asarray(ends)
    delta = ends-starts
    length = np.linalg.norm(delta, axis=1)
    mask = length > 1e-7
    starts, ends, delta, length = starts[mask], ends[mask], delta[mask], length[mask]
    if not len(starts):
        return None
    direction = delta/length[:,None]
    helper = np.tile([0.,0.,1.], (len(starts),1))
    helper[np.abs(direction[:,2]) > .9] = [1.,0.,0.]
    x = np.cross(direction,helper)
    x /= np.linalg.norm(x,axis=1)[:,None]
    y = np.cross(direction,x)
    angles = np.arange(sides)*2*np.pi/sides
    offsets = radius*(x[:,None,:]*np.cos(angles)[None,:,None]
                      + y[:,None,:]*np.sin(angles)[None,:,None])
    vertices = np.stack((starts[:,None,:]+offsets, ends[:,None,:]+offsets),axis=1).reshape(-1,3)
    indices=[]
    for j in range(len(starts)):
        base=j*2*sides
        for k in range(sides):
            n=(k+1)%sides
            indices.extend((base+k,base+n,base+sides+k,
                            base+n,base+sides+n,base+sides+k))
    return vertices.tolist(), indices


def uniform_path(points, spacing):
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    keep = np.r_[True,np.diff(length)>1e-8]
    if length[-1] < 1e-7:
        return points[:1]
    query=np.linspace(0,length[-1],max(2,int(np.ceil(length[-1]/spacing))+1))
    return np.column_stack([np.interp(query,length[keep],points[keep,j]) for j in range(3)])


class BulletScene:
    """A separate PyBullet scene replays measured poses at recorded coordinates."""
    def __init__(self, trace, width, height, supersample=2, renderer='tiny'):
        self.temp=tempfile.TemporaryDirectory(prefix='flight-rl-render-')
        self.mesh_counter=0
        self.client=p.connect(p.DIRECT)
        self.renderer=p.ER_TINY_RENDERER
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0,physicsClientId=self.client)
        if renderer=='egl':
            plugin=importlib.util.find_spec('eglRenderer')
            if plugin is None:
                self.close()
                raise RuntimeError('The PyBullet EGL renderer plugin is unavailable')
            if p.loadPlugin(plugin.origin,'_eglRendererPlugin',physicsClientId=self.client)<0:
                self.close()
                raise RuntimeError('PyBullet EGL initialization failed; use --renderer tiny')
            self.renderer=p.ER_BULLET_HARDWARE_OPENGL
        self.width,self.height=width,height
        self.supersample=supersample
        self.trace=trace
        self.reference=trace['reference']
        self.actual=trace['position']
        self.bodies=[]
        self.active_trail=None
        self.next_chunk=0
        self.chunk_size=20
        bounds=np.vstack((self.reference,self.actual))
        self.center=(bounds.min(0)+bounds.max(0))/2
        span=np.ptp(bounds,axis=0)
        self.radius=float(np.clip(np.linalg.norm(span)*.0038,.012,.023))
        self.rotation=Slerp(trace['time'],Rotation.from_quat(trace['quaternion']))
        floor=p.createVisualShape(p.GEOM_BOX,halfExtents=[100,100,.025],
                                  rgbaColor=[.064,.092,.137,1],physicsClientId=self.client)
        self.floor=p.createMultiBody(baseMass=0,baseVisualShapeIndex=floor,
                                     baseCollisionShapeIndex=-1,basePosition=[0,0,-.075],
                                     physicsClientId=self.client)
        grid_extent=12
        start,end=[],[]
        for i in range(-grid_extent,grid_extent+1):
            start.extend(([i,-grid_extent,-.044],[-grid_extent,i,-.044]))
            end.extend(([i,grid_extent,-.044],[grid_extent,i,-.044]))
        self.mesh(start,end,.006,[.13,.185,.255,1])
        sampled=uniform_path(self.reference,self.radius*2)
        ids=np.arange(len(sampled)-1)
        visible=ids%9<6
        self.mesh(sampled[:-1][visible],sampled[1:][visible],self.radius,
                  [c/255 for c in CYAN]+[1])
        # A dim footprint gives height context without flattening the trajectory.
        footprint=sampled.copy();footprint[:,2]=-.035
        self.mesh(footprint[:-1][visible],footprint[1:][visible],self.radius*.45,[.10,.25,.30,1])
        self.marker(self.reference[0],self.radius*3.2,[.33,.86,.62,1])
        if np.linalg.norm(self.reference[-1]-self.reference[0])>.1:
            self.marker(self.reference[-1],self.radius*3.2,[.76,.58,1.,1])
        self.target=self.marker(self.reference[0],self.radius*2.8,[c/255 for c in CYAN]+[1])
        self.drone=p.loadURDF(str(ROOT/'FlightEnv/assets/250.urdf'),useFixedBase=True,
                              flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                              physicsClientId=self.client)
        p.changeVisualShape(self.drone,-1,rgbaColor=[.92,.94,.99,1],
                            specularColor=[.4,.4,.4],physicsClientId=self.client)
        self.pointer=self.marker(self.actual[0],self.radius*2.1,[c/255 for c in ORANGE]+[1])
        # Look obliquely across the dominant horizontal direction of the path.
        xy=self.reference[:,:2]-self.reference[:,:2].mean(0)
        _,_,vh=np.linalg.svd(xy,full_matrices=False)
        principal=np.rad2deg(np.arctan2(vh[0,1],vh[0,0]))
        yaw=float(principal+25)
        pitch=-28. if span[2]>max(span[0],span[1])*.8 else -43.
        view=np.array(p.computeViewMatrixFromYawPitchRoll(self.center,1,yaw,pitch,0,2)).reshape(4,4,order='F')
        local=(bounds-self.center)@view[:3,:3].T
        tan_y=np.tan(np.deg2rad(42/2));tan_x=tan_y*width/height
        distance=max(2.4,float(np.max(local[:,2]+np.abs(local[:,0])/(tan_x*.76))),
                     float(np.max(local[:,2]+np.abs(local[:,1])/(tan_y*.57))))+.5
        self.view=p.computeViewMatrixFromYawPitchRoll(self.center,distance,yaw,pitch,0,2)
        self.projection=p.computeProjectionMatrixFOV(42,width/height,.05,200)
        self.view_matrix=np.array(self.view).reshape(4,4,order='F')
        self.projection_matrix=np.array(self.projection).reshape(4,4,order='F')

    def mesh(self,starts,ends,radius,color):
        arrays=tube_mesh(starts,ends,radius)
        if arrays is None:return None
        vertices,indices=arrays
        # EGL does not support vertex-only GEOM_MESH; use a temporary OBJ with
        # exactly the same coordinates for both native PyBullet renderers.
        mesh_path=Path(self.temp.name)/f'tube_{self.mesh_counter:05d}.obj'
        self.mesh_counter+=1
        with mesh_path.open('w') as handle:
            handle.write('o path\n')
            for x,y,z in vertices:
                handle.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
            for i in range(0,len(indices),3):
                a,b,c=indices[i:i+3]
                handle.write(f'f {a+1} {b+1} {c+1}\n')
        shape=p.createVisualShape(p.GEOM_MESH,fileName=str(mesh_path),rgbaColor=color,
                                   specularColor=[.1,.1,.1],physicsClientId=self.client)
        body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,baseVisualShapeIndex=shape,
                               physicsClientId=self.client)
        p.changeVisualShape(body,-1,rgbaColor=color,textureUniqueId=-1,
                            specularColor=[.05,.05,.05],physicsClientId=self.client)
        return body

    def marker(self,position,radius,color):
        shape=p.createVisualShape(p.GEOM_SPHERE,radius=radius,rgbaColor=color,physicsClientId=self.client)
        return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,baseVisualShapeIndex=shape,
                                  basePosition=position,physicsClientId=self.client)

    def project(self,position):
        clip=self.projection_matrix@self.view_matrix@np.r_[position,1.]
        ndc=clip[:3]/clip[3]
        return ((ndc[0]+1)*self.width/2,(1-ndc[1])*self.height/2)

    def render(self,at):
        times=self.trace['time']
        at=float(np.clip(at,0,times[-1]))
        position=np.array([np.interp(at,times,self.actual[:,j]) for j in range(3)])
        reference=np.array([np.interp(at,times,self.reference[:len(times),j]) for j in range(3)])
        quaternion=self.rotation([at]).as_quat()[0]
        p.resetBasePositionAndOrientation(self.drone,position,quaternion,physicsClientId=self.client)
        for body,point in ((self.pointer,position),(self.target,reference)):
            p.resetBasePositionAndOrientation(body,point,[0,0,0,1],physicsClientId=self.client)
        idx=int(np.searchsorted(times,at,side='right')-1)
        while (self.next_chunk+1)*self.chunk_size<=idx:
            first=self.next_chunk*self.chunk_size
            points=self.actual[first:first+self.chunk_size+1]
            self.mesh(points[:-1],points[1:],self.radius*.9,[c/255 for c in ORANGE]+[1])
            self.next_chunk+=1
        if self.active_trail is not None:
            p.removeBody(self.active_trail,physicsClientId=self.client)
        first=self.next_chunk*self.chunk_size
        points=np.vstack((self.actual[first:idx+1],position))
        self.active_trail=self.mesh(points[:-1],points[1:],self.radius*.9,[c/255 for c in ORANGE]+[1])
        result=p.getCameraImage(self.width*self.supersample,self.height*self.supersample,self.view,self.projection,
                                 lightDirection=[-3,-4,8],lightColor=[1.,1.,1.],
                                 lightAmbientCoeff=.7,lightDiffuseCoeff=.55,lightSpecularCoeff=.15,
                                 shadow=0,renderer=self.renderer,
                                 flags=p.ER_NO_SEGMENTATION_MASK,physicsClientId=self.client)
        frame=Image.fromarray(np.asarray(result[2],dtype=np.uint8).reshape(self.height*self.supersample,self.width*self.supersample,4)[:,:,:3])
        if self.supersample>1:
            frame=frame.resize((self.width,self.height),Image.Resampling.LANCZOS)
        return frame,position,reference

    def close(self):
        if p.isConnected(self.client):p.disconnect(self.client)
        self.temp.cleanup()


class Overlay:
    def __init__(self,width,height,demo,metrics):
        self.w,self.h,self.demo,self.metrics=width,height,demo,metrics
        self.scale=width/1280
        base=Path('/usr/share/fonts/truetype/dejavu')
        self.fonts={size:ImageFont.truetype(str(base/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),
                                          max(10,round(size*self.scale)))
                    for size,bold in ((15,False),(18,False),(20,False),(22,True),(30,True),(40,True))}

    def apply(self,scene,at,position,reference,project,finished=False):
        frame=scene.convert('RGBA')
        ink=Image.new('RGBA',frame.size)
        draw=ImageDraw.Draw(ink)
        s=self.scale
        def box(bounds,fill):draw.rectangle(tuple(round(v*s) for v in bounds),fill=fill)
        def text(x,y,value,size=18,color=WHITE):draw.text((round(x*s),round(y*s)),value,font=self.fonts[size],fill=color)
        box((0,0,1280,146),(*BG,241))
        box((0,601,1280,720),(*BG,248))
        box((48,34,52,79),(*CYAN,255))
        text(69,25,self.demo.title,40)
        text(70,79,self.demo.subtitle,18,MUTED)
        text(968,33,'FLIGHT / RL',22)
        text(960,70,'PYBULLET SIMULATION',15,MUTED)
        for x,color,label in ((72,CYAN,'Reference path'),(297,ORANGE,'Actual flight')):
            draw.line((x*s,121*s,(x+34)*s,121*s),fill=color,width=max(2,round(3*s)))
            text(x+46,109,label,18)
        text(925,109,self.metrics.get('controller_label','FULL DIFFICULTY / PPO'),15,MUTED)
        if 'effective_speed_scale' in self.metrics:
            text(570,111,f"PEAK {self.metrics['peak_reference_speed_m_s']:.2f} m/s  |  {self.metrics['effective_speed_scale']:.2f}x",15,MUTED)
        error=float(np.linalg.norm(position-reference))
        # Projected marker annotation is measured from the same camera matrices.
        if .1<at<self.metrics['duration_seconds']-.1:
            a,b=project(position),project(reference)
            draw.line((*a,*b),fill=(237,243,255,150),width=max(1,round(s)))
        box((48,607,1232,609),(48,65,87,255))
        box((48,607,48+1184*at/self.metrics['duration_seconds'],609),(*CYAN,255))
        text(49,630,'POSITION ERROR',15,MUTED)
        text(48,652,f'{error:.2f} m',30)
        text(330,630,'FLIGHT TIME',15,MUTED)
        text(330,652,f'{at:04.1f} / {self.metrics["duration_seconds"]:.1f} s',30)
        text(620,630,'EPISODE RMSE',15,MUTED)
        text(620,652,f'{self.metrics["position_rmse_m"]:.3f} m',30)
        color=(116,232,178) if finished else CYAN
        text(1010,635,'COMPLETED' if finished else 'TRACKING',22,color)
        text(1008,669,'Test trajectory '+str(self.demo.trajectory_id),15,MUTED)
        return Image.alpha_composite(frame,ink).convert('RGB')


class VideoWriter:
    def __init__(self,path,width,height,fps):
        self.path=Path(path)
        self.temp=self.path.with_name(self.path.stem+'.tmp.mp4')
        self.log=self.path.with_suffix('.encode.log').open('wb')
        self.process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
            '-threads','2','-movflags','+faststart',str(self.temp)],stdin=subprocess.PIPE,stderr=self.log)

    def write(self,frame):
        self.process.stdin.write(frame.tobytes())

    def close(self,success=True):
        self.process.stdin.close()
        code=self.process.wait()
        self.log.close()
        if code:raise RuntimeError(f'FFmpeg failed; see {self.path.with_suffix(".encode.log")}')
        if success:self.temp.replace(self.path)


def make_gif(mp4,path,width,fps):
    graph=(f'fps={fps},scale={width}:-2:flags=lanczos,split[a][b];'
           '[a]palettegen=max_colors=160:stats_mode=diff[p];'
           '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(mp4),
                     '-filter_complex_threads','1','-lavfi',graph,'-loop','0',str(path)],check=True)


def render_demo(trace,metrics,demo,args):
    scene=BulletScene(trace,args.width,args.height,args.supersample,args.renderer)
    overlay=Overlay(args.width,args.height,demo,metrics)
    output=Path(args.output)
    try:
        if args.preview:
            at=trace['time'][-1]
            raw,actual,target=scene.render(at)
            overlay.apply(raw,at,actual,target,scene.project,True).save(output/f'{demo.family}_poster.png')
            return
        writer=VideoWriter(output/f'{demo.family}.mp4',args.width,args.height,args.fps)
        success=False
        try:
            timeline=np.linspace(0,trace['time'][-1],round(trace['time'][-1]*args.fps)+1)
            begin=time.monotonic()
            for index,at in enumerate(timeline):
                raw,actual,target=scene.render(at)
                finished=index==len(timeline)-1
                frame=overlay.apply(raw,at,actual,target,scene.project,finished)
                writer.write(frame)
                if index==0:
                    for _ in range(round(.5*args.fps)):writer.write(frame)
                if index%args.fps==0:
                    print(f'{demo.family}: {index}/{len(timeline)-1} frames ({time.monotonic()-begin:.1f}s)',flush=True)
                if finished:
                    frame.save(output/f'{demo.family}_poster.png')
                    for _ in range(round(1.5*args.fps)):writer.write(frame)
            success=True
        finally:
            writer.close(success)
        if not getattr(args, 'mp4_only', False):
            make_gif(output/f'{demo.family}.mp4',output/f'{demo.family}.gif',args.gif_width,args.gif_fps)
    finally:
        scene.close()



def build_showcase(output, manifest):
    """Build a four-panel loop, a local preview page and a portable media pack."""
    output = Path(output)
    demos = manifest['demos']
    sources = [output / f"{demo['family']}.mp4" for demo in demos]
    if len(sources) == 4:
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
        for source in sources:
            command.extend(['-i', str(source)])
        width, height = manifest['width'], manifest['height']
        graph = ';'.join(f'[{i}:v]scale={width//2}:{height//2}:flags=lanczos,setpts=PTS-STARTPTS[v{i}]'
                         for i in range(4))
        graph += ';[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=0x09111e[v]'
        command.extend(['-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[v]',
                        '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'medium', '-threads', '2',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output/'showcase.mp4')])
        subprocess.run(command, check=True)
        poster = Image.new('RGB', (width, height), BG)
        for i, demo in enumerate(demos):
            with Image.open(output/f"{demo['family']}_poster.png") as source:
                frame = source.resize((width//2, height//2), Image.Resampling.LANCZOS)
            poster.paste(frame, ((i%2)*width//2, (i//2)*height//2))
        poster.save(output/'showcase_poster.png')
    else:
        shutil.copy2(sources[0], output/'showcase.mp4')
        shutil.copy2(output/f"{demos[0]['family']}_poster.png", output/'showcase_poster.png')
    make_gif(output/'showcase.mp4', output/'showcase.gif', min(960,manifest['width']), manifest['gif_fps'])
    cards=[]
    for demo in demos:
        family=demo['family']
        cards.append(f'''
        <article><div class="caption"><h2>{html.escape(demo['title'])}</h2>
          <span>RMSE {demo['position_rmse_m']:.3f} m</span></div>
          <video controls loop muted playsinline preload="none" poster="{family}_poster.png"
                 aria-label="{html.escape(demo['title'])} trajectory tracking">
            <source src="{family}.mp4" type="video/mp4"></video>
          <p>{html.escape(demo['subtitle'])}</p>
          <div class="links"><a href="{family}.mp4" download>MP4</a>
            <a href="{family}.gif" download>GIF</a><a href="{family}_poster.png" download>Poster</a></div>
        </article>''')
    page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flight RL — trajectory tracking</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:#09111e;color:#e9f1fc}
main{max-width:1280px;margin:auto;padding:56px 28px 60px}.eyebrow{color:#23d3e6;font-size:13px;letter-spacing:.2em}
h1{font-size:clamp(32px,5vw,56px);letter-spacing:-.035em;margin:14px 0 18px;max-width:850px;line-height:1.08}
p{color:#93a8c3;line-height:1.7}header>p{max-width:760px;font-size:18px}
.legend{display:flex;gap:28px;margin:30px 0 20px;font-size:14px}.legend i{display:inline-block;width:28px;margin-right:10px;border-top:3px solid #23d3e6;vertical-align:middle}.legend .actual{border-color:#ff8742}
video{display:block;width:100%;aspect-ratio:16/9;background:#0b1422;border-radius:12px}
.hero{border:1px solid #26364d;border-radius:13px;overflow:hidden}
.links{display:flex;gap:18px;padding:14px 0 0}.links a{color:#93dae5;font-size:13px;text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:32px;margin-top:40px}
article{min-width:0}.caption{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
h2{font-size:20px;margin:0}.caption span{color:#93a8c3;font-size:13px}article p{font-size:14px;margin-bottom:0}
footer{border-top:1px solid #26364d;margin-top:44px;padding-top:18px;font-size:13px;color:#93a8c3}
@media(max-width:720px){main{padding:32px 16px}.grid{grid-template-columns:1fr;gap:28px}.caption{flex-wrap:wrap}}
</style></head><body><main><header><div class="eyebrow">FLIGHT / RL · PYBULLET</div>
<h1>Learning to follow complex trajectories.</h1>
<p>Real PPO flights through crossing curves, spiral ascents and a vertical loop.
Full-difficulty reference paths and measured flight trajectories, rendered in PyBullet.</p></header>
<div class="legend"><span><i></i>Reference path</span><span><i class="actual"></i>Actual flight</span></div>
<div class="hero"><video id="hero" controls autoplay loop muted playsinline preload="metadata"
poster="showcase_poster.png" aria-label="Four difficult trajectory tracking demonstrations">
<source src="showcase.mp4" type="video/mp4"></video></div>
<div class="links"><a href="showcase.mp4" download>Download combined MP4</a>
<a href="showcase.gif" download>Download combined GIF</a></div><section class="grid">'''
    page+=''.join(cards)
    page+='''</section><footer>Selected successful demonstrations from a held-out test set.
These clips do not represent a 100% success claim. Cyan: reference; orange: actual flight.
Each flight lasts 10 seconds, with brief introductory and final holds. Vertical loops describe the
position path; they are not inverted flips.</footer></main>
<script>if(matchMedia('(prefers-reduced-motion: reduce)').matches){const v=document.getElementById('hero');v.autoplay=false;v.pause();}</script>
</body></html>'''
    (output/'index.html').write_text(page)
    lines=['# Flight-RL 主页视频素材', '',
           f'本目录包含 {len(demos)} 条选定的成功仿真案例。不是整体成功率展示；完整独立测试结果为 252/256，详见仓库 TRAINING_V3.md。', '',
           '## 使用', '',
           '- 打开 `index.html` 本地预览；全部资源都在当前目录，无外部网络依赖。',
           '- 主页优先嵌入 `showcase.mp4` 或单条 MP4，GIF 用于需要图片格式的页面。',
           '- 青色虚线是完整参考路径；橙色实线为测得的实际路径；两者的偏差真实保留。', '',
           '| 文件前缀 | 内容 | 轨迹编号 | 位置 RMSE |', '| --- | --- | ---: | ---: |']
    for demo in demos:
        lines.append(f"| `{demo['family']}` | {demo['title']} | {demo['trajectory_id']} | {demo['position_rmse_m']:.3f} m |")
    lines.extend([f'| `showcase` | {"四画面合辑" if len(demos)==4 else "单条展示"} | — | — |', '',
                  f"MP4：{manifest['width']}×{manifest['height']}、{manifest['fps']} fps、H.264/yuv420p、无音轨、faststart。",
                  f"单条 GIF：{manifest['gif_width']} 像素宽、{manifest['gif_fps']} fps、无限循环。合辑 GIF 最宽 960 像素。",
                  '每条包含完整 10 秒飞行、约 0.5 秒开头停留和 1.5 秒完成停留，视频总长约 12.03 秒。', '',
                  '```html', '<video autoplay loop muted playsinline preload="metadata" poster="showcase_poster.png" style="width:100%;height:auto">',
                  '  <source src="showcase.mp4" type="video/mp4">', '</video>', '```', '',
                  '## 渲染与真实性', '',
                  '策略在独立的物理环境中重新执行；原始位置、姿态、动作和误差保存为 `*_rollout.npz`、`*_metrics.json`。',
                  '之后在单独的 PyBullet 场景中按测得的状态回放并逐帧录制。使用原始尺寸的 250 无人机 URDF、真实坐标和实际姿态。',
                  '轨迹线是无碰撞的 3D 管线；画面标题和数值为排版叠加。50 Hz 物理状态插值为 30 fps 视频：位置线性插值，姿态使用四元数 SLERP。',
                  '首尾停留不修改飞行数据。没有轨迹吸附、结果平滑、飞行片段删除或反向播放。空间回环并不是机体翻转。',
                  '导出默认使用 PyBullet EGL 离屏 OpenGL，具体硬件见运行日志。TinyRenderer 可作为 CPU 回退，但原始高面数机模渲染较慢。',
                  'PyBullet 相机 API 参考：[官方指南](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)。', '',
                  '## 重现导出', '', '在 Flight-RL 仓库根目录运行（需要已有模型、轨迹库、ffmpeg 和 DejaVu Sans 字体）：', '',
                  '```bash', 'conda activate flight-rl', 'python -m FlightEnv.showcase --output exports/homepage',
                  '# CPU 回退：加 --renderer tiny；只导出一条：加 --families helix', '```', '',
                  '模型哈希、轨迹编号、初始随机种子、每次实测指标和渲染配置在 `manifest.json`。'])
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('Built homepage preview, combined MP4 and GIF.',flush=True)



def package_showcase(output):
    output=Path(output)
    archive=output.parent/f'flight_rl_{output.name}.zip'
    manifest=json.loads((output/'manifest.json').read_text())
    selected={'README.md','index.html','manifest.json','verification.json',
              'showcase.mp4','showcase.gif','showcase_poster.png'}
    for demo in manifest['demos']:
        selected.update(demo['family']+suffix for suffix in
                        ('.mp4','.gif','_poster.png','_metrics.json','_rollout.npz'))
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as package:
        for file in sorted(output.iterdir()):
            if file.is_file() and file.name in selected:
                package.write(file,arcname=f'{output.name}/{file.name}')
    print(f'Media pack: {archive.resolve()}',flush=True)
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='runs/v3_stable/best/best_model.zip')
    parser.add_argument('--output',default='exports/homepage')
    parser.add_argument('--families',nargs='+',choices=[d.family for d in DEMOS],default=[d.family for d in DEMOS])
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--fps',type=int,default=30)
    parser.add_argument('--supersample',type=int,choices=(1,2),default=2)
    parser.add_argument('--renderer',choices=('egl','tiny'),default='egl')
    parser.add_argument('--gif-width',type=int,default=768)
    parser.add_argument('--gif-fps',type=int,default=15)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    if min(args.width,args.height,args.fps,args.gif_width,args.gif_fps)<1 or args.width%2 or args.height%2:
        parser.error('Dimensions must be positive/even; frame rates must be positive')
    if args.height*16!=args.width*9:
        parser.error('The presentation layout requires a 16:9 frame')
    if shutil.which('ffmpeg') is None:parser.error('ffmpeg is required')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    from main import load_model
    model=load_model(args.model,'cpu')
    if model.flight_env_config.get('profile')!='tracking':parser.error('Requires a V3 tracking model')
    model_path=Path(args.model).resolve()
    model_label=str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else model_path.name
    if not args.preview:
        (output/'verification.json').unlink(missing_ok=True)
    manifest=dict(model=model_label,
                  sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  renderer='PyBullet EGL' if args.renderer=='egl' else 'PyBullet ER_TINY_RENDERER',camera='Fixed per trajectory',
                  width=args.width,height=args.height,fps=args.fps,supersample=args.supersample,gif_width=args.gif_width,gif_fps=args.gif_fps,
                  selection='Four curated successful held-out rollouts; not aggregate benchmark performance.',
                  scene='Real URDF at original scale; recorded positions/orientations replayed in a separate PyBullet client.',
                  poses='50 Hz physical states; linear position interpolation and quaternion SLERP for video frames.',
                  timing='Real-time flight, 0.5-second introductory hold, 1.5-second final hold.',demos=[])
    for demo in DEMOS:
        if demo.family not in args.families:continue
        print(f'Collecting {demo.family} / test trajectory {demo.trajectory_id}',flush=True)
        trace,metrics=collect_rollout(model,demo,output)
        print(f'  completed, RMSE={metrics["position_rmse_m"]:.3f} m',flush=True)
        render_demo(trace,metrics,demo,args)
        manifest['demos'].append(metrics)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.preview:
        build_showcase(output,manifest)
        package_showcase(output)
    print(f'Exported to {output.resolve()}',flush=True)


if __name__=='__main__':main()
