"""Test curated difficult held-out flights and export PyBullet MP4s."""
import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from FlightEnv.showcase import Demo, ROOT, collect_rollout, package_showcase, render_demo

DEMOS = (
    Demo('figure8', 100, 2060, '01 / Figure eight', 'Fast crossing turns in three dimensions'),
    Demo('helix', 143, 2149, '02 / Double spiral', 'Two full turns with continuous ascent'),
    Demo('vertical_loop', 233, 2104, '03 / Vertical loop', 'A full vertical position loop with continuous tracking'),
    Demo('lissajous', 267, 2121, '04 / Lissajous', 'Coupled oscillations, crossings and tight direction changes'),
    Demo('slalom', 162, 2046, '05 / Slalom', 'Repeated alternating turns with rapid acceleration'),
    Demo('wave', 213, 2199, '06 / Climb and dive', 'Two steep waves with simultaneous lateral turns'),
)
NAMES = dict(figure8='三维 8 字', helix='螺旋爬升', vertical_loop='垂直回环',
             lissajous='三维交叉曲线', slalom='连续蛇形', wave='连续爬升俯冲')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def verify_rollout(trace, metrics):
    for key, value in trace.items():
        if not np.isfinite(value).all():
            raise ValueError(f'{key} contains nonfinite values')
    np.testing.assert_allclose(np.linalg.norm(trace['quaternion'], axis=1), 1., atol=1e-6)
    np.testing.assert_allclose(np.diff(trace['time']), .02, atol=1e-10)
    if trace['position'].shape != trace['reference'].shape:
        raise ValueError('The full reference was not completed')
    error = np.linalg.norm(trace['position'][1:]-trace['reference'][1:], axis=1)
    np.testing.assert_allclose(error, trace['error'], atol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.mean(error**2)), metrics['position_rmse_m'], atol=1e-10)
    assert metrics['completed'] and metrics['tracking_success']
    assert metrics['position_rmse_m'] <= .10 and metrics['max_position_error_m'] <= .30
    return dict(finite=True, full_reference_completed=True, quaternion_norms=True,
                errors_recomputed_from_positions=True, strict_tracking_pass=True)


def compare_audit(trace, metrics, audit_path):
    if not audit_path.is_file():
        return dict(available=False)
    audit = json.loads(audit_path.read_text())
    matches = [(i, ep) for i, ep in enumerate(audit['episodes'])
               if ep['trajectory_id'] == metrics['trajectory_id'] and ep['seed'] == metrics['seed']]
    if len(matches) != 1:
        raise ValueError('Expected exactly one matching audit trajectory and reset seed')
    index, episode = matches[0]
    np.testing.assert_allclose(metrics['position_rmse_m'], episode['rmse'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(metrics['max_position_error_m'], episode['max_error'], atol=1e-9, rtol=0)
    csv = audit_path.parent/f'episode_{index:03d}.csv'
    data = np.loadtxt(csv, delimiter=',', skiprows=1)
    np.testing.assert_allclose(trace['time'][1:], data[:, 0], atol=1e-7, rtol=0)
    np.testing.assert_allclose(trace['position'][1:], data[:, 1:4], atol=1e-7, rtol=0)
    return dict(available=True, source=str(audit_path), episode_index=index,
                matches_prior_physics_rollout=True)


def verify_video(path, trace, args):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames:format=duration,size',
        '-of', 'json', str(path)]))
    stream = probe['streams'][0]
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
    assert (stream['width'], stream['height']) == (args.width, args.height)
    assert stream['r_frame_rate'] == f'{args.fps}/1'
    expected_frames = round(trace['time'][-1]*args.fps)+1+round(.5*args.fps)+round(1.5*args.fps)
    assert int(stream['nb_frames']) == expected_frames
    assert abs(float(probe['format']['duration'])-expected_frames/args.fps) < .002
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                             capture_output=True, text=True, check=True)
    if decoded.stderr.strip():
        raise RuntimeError(f'Video decode errors: {decoded.stderr}')
    atoms = []
    with path.open('rb') as handle:
        while header := handle.read(8):
            size, kind = int.from_bytes(header[:4], 'big'), header[4:].decode('ascii')
            header_size = 8
            if size == 1:
                size, header_size = int.from_bytes(handle.read(8), 'big'), 16
            atoms.append(kind)
            if size == 0:
                break
            handle.seek(size-header_size, 1)
    assert atoms.index('moov') < atoms.index('mdat')
    return dict(file=path.name, **stream, duration_seconds=float(probe['format']['duration']),
                bytes=int(probe['format']['size']), complete_decode=True, faststart=True,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build_gallery(output, manifest):
    cards = []
    count = len(manifest['demos'])
    zero_residual = all(d['control_mode'] == 'geometric_feedforward_zero_residual' for d in manifest['demos'])
    controller = '几何前馈 + 零 RL 残差基线' if zero_residual else '几何前馈 + PPO 残差修正'
    control_note = f"控制方式：{controller}。检查点累计训练交互步数：{manifest['model_timesteps']}。"
    lines = [f'# {count} 条高难度轨迹飞行视频', '',
        '青色虚线：参考轨迹；橙色实线：真实仿真航迹。全部为独立测试集、完整难度、完整飞行。', '',
        control_note + ' 控制基线与学习残差的贡献需要分别报告。', '',
        f'本次选择 {count} 类有难度且通过严格标准的案例，再重新进行 PyBullet 物理仿真。'
        '成功要求完整飞完、位置 RMSE ≤ 0.10 m、最大位置误差 ≤ 0.30 m。'
        '选定案例不能代替整体成功率；既有 256 条测试中严格通过 227 条，完成 254 条。', '',
        '| 文件 | 测试轨迹 / 随机种子 | RMSE | 最大误差 | 参考峰值速度 | 实际时间倍率 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for demo in manifest['demos']:
        family = demo['family']
        fallback = demo['cruise_fallback']
        timing = '急弯受加速度/jerk 等物理约束，采用限速后的五次时间曲线' if fallback else '巡航时间曲线'
        cards.append(f'''<article><div class="caption"><h2>{html.escape(NAMES[family])}</h2>
<span>RMSE {100*demo['position_rmse_m']:.2f} cm</span></div>
<video controls loop muted playsinline preload="metadata" poster="{family}_poster.png">
<source src="{family}.mp4" type="video/mp4"></video>
<p>测试 #{demo['trajectory_id']} · 峰值 {demo['peak_reference_speed_m_s']:.2f} m/s ·
实际时间倍率 {demo['effective_speed_scale']:.2f}× · 飞行 {demo['duration_seconds']:.2f} s</p>
<p>{timing}。最大误差 {100*demo['max_position_error_m']:.2f} cm。</p>
<a href="{family}.mp4" download>下载 MP4</a> · <a href="{family}_metrics.json">测试数据</a></article>''')
        lines.append(f"| [{NAMES[family]}]({family}.mp4) | {demo['trajectory_id']} / {demo['seed']} | "
                     f"{100*demo['position_rmse_m']:.2f} cm | {100*demo['max_position_error_m']:.2f} cm | "
                     f"{demo['peak_reference_speed_m_s']:.2f} m/s | {demo['effective_speed_scale']:.3f}× |")
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Flight / RL · __DEMO_COUNT__ 条高难度飞行</title>
<style>:root{color-scheme:dark;font-family:system-ui,sans-serif}*{box-sizing:border-box}
body{margin:0;background:#09111e;color:#e9f1fc}main{max-width:1280px;margin:auto;padding:48px 28px}
h1{font-size:clamp(30px,5vw,48px);margin:18px 0}p{color:#93a8c3;line-height:1.7}
.eyebrow{color:#23d3e6;letter-spacing:.18em;font-size:13px}.legend{display:flex;gap:28px;margin:28px 0}
.legend i{display:inline-block;width:30px;margin-right:10px;border-top:3px dashed #23d3e6;vertical-align:middle}
.legend .actual{border-top:3px solid #ff8742}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:34px}
.caption{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:14px 0}
h2{font-size:21px;margin:0}.caption span{font-size:14px;color:#93a8c3}
video{display:block;width:100%;aspect-ratio:16/9;border:1px solid #26364d;border-radius:12px;background:#09111e}
a{color:#8fdee8;font-size:14px;text-decoration:none}article p{font-size:14px;margin:10px 0}
footer{margin-top:36px;padding-top:20px;border-top:1px solid #26364d;font-size:13px}
@media(max-width:760px){main{padding:28px 16px}.grid{grid-template-columns:1fr}}</style></head>
<body><main><header><div class="eyebrow">FLIGHT / RL · PYBULLET</div><h1>__DEMO_COUNT__ 条高难度轨迹，完整飞行实测</h1>
<p>8 字交叉、螺旋爬升、垂直回环、三维交叉曲线、连续蛇形和连续爬升俯冲。固定全景镜头保留真实跟踪偏差，按真实时间播放。</p>
<p>__CONTROL_NOTE__ 全部通过完整飞行、RMSE ≤ 10 cm、最大误差 ≤ 30 cm 检查。</p></header>
<div class="legend"><span><i></i>参考轨迹</span><span><i class="actual"></i>实际航迹</span></div><section class="grid">'''
    page += ''.join(cards)
    page += '''</section><footer><p>选定的成功案例，不能代表所有轨迹均成功。既有 256 条测试中严格通过 227 条。
请求时间倍率为 1.5×，实际倍率在各视频旁列出；部分急弯受到物理约束而自动限速。
垂直回环指空间位置路径，不是机体倒飞翻转。每条保留约 0.5 秒开头和 1.5 秒结尾停留。</p></footer></main></body></html>'''
    page = page.replace('__CONTROL_NOTE__', html.escape(control_note)).replace('__DEMO_COUNT__', str(count))
    (output/'index.html').write_text(page)
    lines.extend(['', f"视频格式：{manifest['width']}×{manifest['height']}，{manifest['fps']} fps，H.264/yuv420p，无音轨，faststart。共 {count} 个 MP4。", '',
        '打开 `index.html` 可逐条预览；文件名和标题相互对应。峰值速度为参考轨迹峰值，实测飞行速度另存于 metrics。', '',
        '请求时间倍率为 1.5×，相对于原始 10 秒参考轨迹。三维交叉曲线和蛇形受到急弯物理约束，自动回退并限速；实际倍率见表。'
        '所有视频按真实时间播放，不后期加速。空间回环不是机体倒飞翻转。', '',
        '策略重新驱动 PyBullet 物理仿真，位置、姿态、速度、参考状态、残差动作和实际混控指令保存为 `*_rollout.npz`。'
        '之后单独使用 PyBullet EGL 场景按记录的状态渲染，使用原始尺寸的无人机 URDF。'
        '50 Hz 物理状态转为 30 fps 时位置线性插值、姿态 SLERP；不吸附轨迹，不改写物理仿真。', '',
        '`manifest.json` 保存模型哈希、环境参数和每条测试结果；`verification.json` 保存数据校验、历史测试一致性和视频解码检查。', '',
        '复现（仓库根目录）：', '', '```bash', 'conda activate flight-rl',
        'python -m FlightEnv.difficult_showcase --output exports/difficult_flights_6', '```', ''])
    (output/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='runs/residual_baseline/baseline.zip')
    parser.add_argument('--output', default='exports/difficult_flights_6')
    parser.add_argument('--audit', help='Optional previous metrics.json for exact rollout comparison')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--supersample', type=int, choices=(1, 2), default=2)
    parser.add_argument('--renderer', choices=('egl', 'tiny'), default='egl')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    args.mp4_only = True
    if min(args.width, args.height, args.fps) < 1 or args.width % 2 or args.height % 2:
        parser.error('Dimensions must be positive/even; fps must be positive')
    if args.height*16 != args.width*9:
        parser.error('The presentation layout requires 16:9')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        parser.error('ffmpeg and ffprobe are required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {f'{demo.family}.mp4' for demo in DEMOS}
    if any(file.name not in allowed for file in output.glob('*.mp4')):
        parser.error('Choose a directory without unrelated MP4 files')
    torch.set_num_threads(1)
    from main import load_model
    model = load_model(args.model, 'cpu')
    if model.flight_env_config.get('profile') != 'residual':
        parser.error('This curated export requires the residual-control model')
    model_path = Path(args.model).resolve()
    manifest = dict(created_at=datetime.now().astimezone().isoformat(),
        model=str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), model_timesteps=int(model.num_timesteps),
        renderer=f'PyBullet {args.renderer}', width=args.width, height=args.height, fps=args.fps,
        supersample=args.supersample, requested_speed_scale=1.5, requested_time_profile='cruise',
        selection=f'{len(DEMOS)} diverse successful held-out cases; not aggregate benchmark performance.',
        timing='Real-time flight with 0.5-second introductory and 1.5-second final holds.',
        command=[sys.executable, '-m', 'FlightEnv.difficult_showcase', *sys.argv[1:]], demos=[])
    checks, flights = [], []
    for demo in DEMOS:
        print(f'Testing {demo.family}: trajectory {demo.trajectory_id}, seed {demo.seed}', flush=True)
        trace, metrics = collect_rollout(model, demo, output,
            config_overrides=dict(time_profile='cruise', speed_min=1.5, speed_max=1.5),
            success_rmse=.10, success_max_error=.30)
        check = dict(family=demo.family, rollout=verify_rollout(trace, metrics),
                     prior_audit=compare_audit(trace, metrics, Path(args.audit)) if args.audit else dict(available=False))
        checks.append(check)
        flights.append((demo, trace, metrics))
        manifest['demos'].append(metrics)
        print(f"  PASS: RMSE={100*metrics['position_rmse_m']:.2f} cm, "
              f"max={100*metrics['max_position_error_m']:.2f} cm, "
              f"effective speed={metrics['effective_speed_scale']:.3f}x", flush=True)
    write_json(output/'manifest.json', manifest)
    for (demo, trace, metrics), check in zip(flights, checks):
        render_demo(trace, metrics, demo, args)
        if not args.preview:
            check['video'] = verify_video(output/f'{demo.family}.mp4', trace, args)
        write_json(output/'verification.json', dict(preview_only=args.preview, checks=checks))
    if not args.preview:
        assert {path.name for path in output.glob('*.mp4')} == allowed
        build_gallery(output, manifest)
        package_showcase(output)
    print(f'Exported exactly {0 if args.preview else len(DEMOS)} MP4s to {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
