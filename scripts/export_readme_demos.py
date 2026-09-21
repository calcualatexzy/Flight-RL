"""Convert five measured-flight MP4s into portable, looping README GIF assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image

FAMILIES = ('figure8', 'helix', 'vertical_loop', 'lissajous', 'slalom')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('exports/difficult_flights_5'))
    parser.add_argument('--output', type=Path, default=Path('docs/media'))
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--fps', type=int, default=15)
    args = parser.parse_args()
    if args.width < 2 or args.width % 2 or args.fps < 1:
        parser.error('Width must be positive/even and fps positive')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        parser.error('ffmpeg and ffprobe are required')
    manifest_path = args.input_dir/'manifest.json'
    source = json.loads(manifest_path.read_text())
    demos = {demo['family']: demo for demo in source['demos']}
    for family in FAMILIES:
        if family not in demos or not (args.input_dir/f'{family}.mp4').is_file():
            parser.error(f'Missing measured MP4/metadata for {family}')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(model_sha256=source['sha256'], model_timesteps=source['model_timesteps'],
        source_renderer=source['renderer'], source_manifest_sha256=sha256(manifest_path),
        requested_speed_scale=source['requested_speed_scale'],
        timing='Complete flights at original playback speed; introductory/final holds preserved. Infinite loop.',
        selection='Five selected successful held-out cases, not an aggregate success-rate claim.',
        conversion=dict(width=args.width, fps=args.fps, palette_colors=160, dither='bayer'), demos=[])
    for family in FAMILIES:
        mp4 = args.input_dir/f'{family}.mp4'
        gif = args.output/f'{family}.gif'
        temporary = gif.with_name(f'{family}.tmp.gif')
        graph = (f'fps={args.fps},scale={args.width}:-2:flags=lanczos,split[a][b];'
            '[a]palettegen=max_colors=160:stats_mode=diff[p];'
            '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(mp4),
            '-filter_complex_threads', '1', '-lavfi', graph, '-loop', '0', str(temporary)], check=True)
        with Image.open(temporary) as image:
            if not image.is_animated or image.info.get('loop') != 0:
                raise ValueError(f'{family} is not an infinitely looping animation')
            width, height = image.size
            frames, duration_ms = image.n_frames, 0
            for index in range(frames):
                image.seek(index)
                image.load()
                duration_ms += image.info.get('duration', 0)
        duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(mp4)]))
        if abs(duration_ms/1000-duration) > 2/args.fps+.02:
            raise ValueError(f'{family}: GIF duration differs from full source video')
        temporary.replace(gif)
        demo = demos[family]
        published = {key: demo[key] for key in ('family', 'title', 'subtitle', 'trajectory_id',
            'seed', 'split', 'difficulty', 'completed', 'tracking_success', 'duration_seconds',
            'position_rmse_m', 'max_position_error_m', 'peak_reference_speed_m_s',
            'mean_actual_speed_m_s', 'effective_speed_scale', 'effective_time_profile',
            'cruise_fallback', 'control_mode', 'acceptance')}
        published.update(file=gif.name, width=width, height=height, frames=frames,
            gif_duration_seconds=duration_ms/1000, bytes=gif.stat().st_size,
            sha256=sha256(gif), source_mp4_sha256=sha256(mp4))
        manifest['demos'].append(published)
        print(f'{gif}: {width}x{height}, {frames} frames, {duration_ms/1000:.2f}s, '
              f'{gif.stat().st_size/1024/1024:.2f} MiB', flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
