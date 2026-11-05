"""Standalone local video runner, same pipeline as FastAPI (without HTTP)."""
import argparse
import json
import math
from pathlib import Path

import cv2
from detector import VehicleDetector
from camera_calibration import CameraCalibration
from trajectory import TrajectoryManager
from accident_detector import AccidentDetector
from visualizer import Visualizer
from matrix_builder import build_matrix_13, build_speed_profile


def run(video, output, model='models/yolo11s.pt', view='auto', profile=None):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f'Video gagal dibuka: {video}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    
    if not math.isfinite(fps) or fps<=0: raise RuntimeError('FPS tidak valid')
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened(): raise RuntimeError('VideoWriter gagal')
    cal=CameraCalibration(view, Path(profile).read_text('utf-8') if profile else None)
    detector=VehicleDetector(str(model),tracker=str(Path(__file__).parent/'bytetrack.yaml'))
    trajectories=TrajectoryManager(fps,calibration=cal)
    crash=AccidentDetector();visualizer=Visualizer()
    frames=0; events=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames+=1
            tracked=detector.detect_and_track(frame)
            histories=trajectories.update(tracked)
            candidates=crash.detect(histories,tracked)
            for e in candidates:
                e['time_s']=round((e['event_frame']-1)/fps,3)
                events.append({'event':e,'matrix_13':build_matrix_13(e,histories,fps,{'weather':'Unknown'},cal),
                               'speed_profile':build_speed_profile(e,histories,fps)})
            writer.write(visualizer.draw_all(frame,tracked,histories,candidates,frame_number=frames,fps=fps))
            if frames%50==0:print(f'Processed {frames} frames')
    finally:
        cap.release();writer.release()
    from statistics import median
    vehicles=[]
    for tid,h in trajectories.history.items():
        rows=[r for r in h.get('speed_history',[]) if r.get('speed_px_s') is not None]
        vals=[r['speed_kmh'] for r in rows if r.get('speed_kmh') is not None]
        vehicles.append({'id':tid,'observations':len(rows),
                         'median_speed_kmh':round(median(vals),2) if vals else None,
                         'median_px_s':round(median([r['speed_px_s'] for r in rows]),2) if rows else None})
    data={'video':str(video),'frames':frames,'fps':fps,'calibration':cal.describe(),
          'tracked_vehicles':vehicles,'candidate_event_count':len(events),'candidate_events':events}
    report=output.with_suffix('.json');report.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print('VIDEO:',output,'\nJSON:',report,'\nCANDIDATES:',len(events))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--video',required=True)
    p.add_argument('--output',default='data/output/anchor_vi_result.mp4')
    p.add_argument('--model',default='models/yolo11s.pt')
    p.add_argument('--view',choices=['auto','side','front','rear','diagonal','elevated','top'],default='auto')
    p.add_argument('--profile',help='camera homography JSON file')
    args=p.parse_args()
    run(args.video,args.output,args.model,args.view,args.profile)
