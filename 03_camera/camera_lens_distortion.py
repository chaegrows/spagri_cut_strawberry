#!/usr/bin/env python3
'''
pip install opencv-contrib-python==4.5.5.64
pip install numpy==1.23.2
'''
import argparse, time
import numpy as np
import cv2
import cv2.aruco as aruco
from collections import defaultdict

# ---------- 유틸 ----------
def depth_to_points(depth, K, scale=1.0):
    """depth: (H,W) in meters; K: fx,fy,cx,cy"""
    H, W = depth.shape
    fx, fy, cx, cy = K
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)  # (H,W,3)

def fit_plane(points):
    """points: (N,3) -> plane RMS (orthogonal distance)"""
    P = points.reshape(-1,3)
    P = P[~np.isnan(P).any(axis=1)]
    if len(P) < 100:  # 포인트 너무 적으면 스킵
        return np.nan
    C = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - C)
    normal = Vt[-1]
    d = -C @ normal
    dist = np.abs(P @ normal + d)
    return float(np.sqrt((dist**2).mean()))

def robust_roi_stats(depth_m, x, y, k=5):
    """(x,y) 주변 k반경 ROI 평균/표준편차 (m)"""
    H,W = depth_m.shape
    xi, yi = int(x), int(y)
    x0, x1 = max(0, xi-k), min(W, xi+k+1)
    y0, y1 = max(0, yi-k), min(H, yi+k+1)
    roi = depth_m[y0:y1, x0:x1]
    roi = roi[(roi>0) & np.isfinite(roi)]
    if roi.size == 0: return np.nan, np.nan
    return float(np.mean(roi)), float(np.std(roi))

# ---------- 메인 ----------
def main():
    import pyrealsense2 as rs

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120, help="Sample frame count")
    ap.add_argument("--z_true", type=float, required=True, help="Reference actual distance[m]")
    ap.add_argument("--tile", type=int, default=3, help="Depth tile division count when no color")
    ap.add_argument("--square_mm", type=float, default=13.0, help="ChArUco square length[mm]")
    ap.add_argument("--marker_mm", type=float, default=11.0, help="ChArUco marker length[mm]")
    ap.add_argument("--dict", type=str, default="DICT_5X5_100", help="ArUco dict name")
    ap.add_argument("--show", action="store_true", help="Show preview window")
    args = ap.parse_args()

    # ----- RealSense Configuration -----
    pipe = rs.pipeline()
    cfg = rs.config()
    # D405: Adjust resolution/format according to environment
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # Open color camera if available
    color_available = True
    try:
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    except Exception:
        color_available = False

    prof = pipe.start(cfg)
    time.sleep(1.0)  # Stabilization

    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # Usually m unit scale
    print(f"Depth scale: {depth_scale}")  # Debug output
    
    # intrinsics
    depth_stream = prof.get_stream(rs.stream.depth).as_video_stream_profile()
    d_intr = depth_stream.get_intrinsics()  # fx, fy, ppx, ppy
    Kd = (d_intr.fx, d_intr.fy, d_intr.ppx, d_intr.ppy)

    if color_available:
        align_to_color = rs.align(rs.stream.color)
        # Charuco board
        dmap = getattr(aruco, args.dict)
        aruco_dict = aruco.Dictionary_get(dmap)
        board = aruco.CharucoBoard_create(
            squaresX=13, squaresY=9,
            squareLength=args.square_mm/1000.0,
            markerLength=args.marker_mm/1000.0,
            dictionary=aruco_dict
        )

    N = args.frames
    W=H=None

    # Tracking buffers
    tracks = defaultdict(list)        # cid -> [(x,y)]
    depth_tracks = defaultdict(list)  # cid -> [Z_mean_at_corner]
    tile_depth_stats = []             # (tile_id, meanZ, stdZ, plane_rms)

    # Collection loop
    for i in range(N):
        frames = pipe.wait_for_frames()
        if color_available:
            frames = align_to_color.process(frames)

        d = frames.get_depth_frame()
        if not d: 
            print(f"Frame {i}: No depth frame")
            continue
        depth = np.asanyarray(d.get_data()).astype(np.float32) * depth_scale  # meters

        # Debug: Check depth data
        valid_depth = depth[(depth > 0) & np.isfinite(depth)]
        print(f"Frame {i}: Valid depth pixels: {len(valid_depth)}, Range: {valid_depth.min():.3f}-{valid_depth.max():.3f}m")

        if color_available:
            c = frames.get_color_frame()
            color = np.asanyarray(c.get_data()) if c else None
        else:
            color = None

        H, W = depth.shape[:2]

        if color_available and color is not None:
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
            ch_corners, ch_ids = None, None
            if ids is not None:
                _, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
                    markerCorners=corners, markerIds=ids, image=gray, board=board
                )
            if ch_ids is not None:
                for (pt, cid) in zip(ch_corners.reshape(-1,2), ch_ids.flatten()):
                    x,y = float(pt[0]), float(pt[1])
                    tracks[int(cid)].append((x,y))
                    z_mean, z_std = robust_roi_stats(depth, x, y, k=5)
                    if not np.isnan(z_mean):
                        depth_tracks[int(cid)].append(z_mean)

            if args.show and color is not None:
                vis = color.copy()
                if ids is not None: aruco.drawDetectedMarkers(vis, corners, ids)
                if ch_ids is not None: aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
                cv2.imshow("RGB", vis); cv2.waitKey(1)
        else:
            # No color: tile-based depth statistics
            t = args.tile
            hstep, wstep = H//t, W//t
            # Calculate plane_rms for some frames (e.g., first 10 frames only)
            do_plane = (i % max(1, N//10) == 0)
            
            for r in range(t):
                for cidx in range(t):
                    y0,y1 = r*hstep, (r+1)*hstep
                    x0,x1 = cidx*wstep, (cidx+1)*wstep
                    roi = depth[y0:y1, x0:x1]
                    
                    # More strict depth filtering for D405
                    valid_mask = (roi > 0.1) & (roi < 2.0) & np.isfinite(roi)  # 10cm ~ 2m range
                    valid = roi[valid_mask]
                    
                    print(f"Frame {i}, Tile {r}-{cidx}: Valid pixels: {len(valid)}")
                    
                    if len(valid) < 50:  # Need minimum pixels for reliable stats
                        continue
                        
                    meanZ = float(valid.mean())
                    stdZ  = float(valid.std())
                    plane_rms = np.nan
                    
                    if do_plane and len(valid) > 100:
                        # Downsample for plane RMS calculation
                        roi_ds = cv2.resize(roi, (max(10, wstep//2), max(10, hstep//2)), interpolation=cv2.INTER_NEAREST)
                        pts = depth_to_points(roi_ds, Kd, scale=1.0)
                        plane_rms = fit_plane(pts)
                        
                    tile_depth_stats.append((r*t+cidx, meanZ, stdZ, plane_rms))
                    print(f"Added tile stat: meanZ={meanZ:.3f}m, stdZ={stdZ:.3f}m")

    pipe.stop()
    if args.show: cv2.destroyAllWindows()

    # ---------- RGB/ChArUco 결과 ----------
    if color_available and len(tracks)>0:
        center = np.array([W/2, H/2])
        rmax = np.hypot(W/2, H/2)

        def px_jitter(arr):
            A = np.array(arr)  # (T,2)
            if len(A) < 3: return np.nan
            return float(A.std(axis=0).mean())

        # per corner metrics
        per_id = {}
        for cid, pts in tracks.items():
            pts = np.array(pts)
            p = len(pts) / N
            sigma_px = px_jitter(pts)
            r_norm = float(np.linalg.norm(pts.mean(axis=0) - center) / rmax)

            zvals = np.array(depth_tracks.get(cid, []))
            zvals = zvals[np.isfinite(zvals)]
            if zvals.size>0:
                z_mean = float(zvals.mean())
                z_std  = float(zvals.std())
                abs_err = abs(z_mean - args.z_true)
            else:
                z_mean = z_std = abs_err = np.nan

            per_id[cid] = dict(p=p, sigma_px=sigma_px, r=r_norm,
                               z_mean=z_mean, z_std=z_std, abs_err=abs_err)

        # pass/fail & 신뢰 반경
        def passed(m):
            ok = (m["p"] >= 0.95) and (m["sigma_px"] <= 0.4)
            if not np.isnan(m["z_std"]):   ok &= (m["z_std"] <= 0.003)         # 3 mm
            if not np.isnan(m["abs_err"]) and not np.isnan(m["z_mean"]):
                ok &= (abs(m["abs_err"]) <= max(0.005, 0.02*m["z_mean"]))      # 5mm or 2%
            return ok

        rs = np.array([m["r"] for m in per_id.values()])
        passes = np.array([passed(m) for m in per_id.values()])
        bins = np.linspace(0,1.0,11)
        ratios = []
        for i in range(len(bins)-1):
            mask = (rs>=bins[i]) & (rs<bins[i+1])
            ratios.append(1.0 if mask.sum()==0 else passes[mask].mean())
        r_star = 1.0
        for i,(a,b) in enumerate(zip(bins[:-1], bins[1:])):
            if ratios[i] < 0.95:
                r_star = a; break

        print("\n=== RGB + Depth (ChArUco 기반) ===")
        print(f"신뢰 반경 r* ≈ {r_star*100:.1f}% (화면 반대각 절반 기준 정규화)")
        # 중앙/주변부 요약
        core = (rs < 0.3)
        edge = (rs > 0.7)
        def stat(mask, key):
            vals = [m[key] for m in per_id.values() if not np.isnan(m[key])]
            if key in ("z_std","abs_err"): vals = [v for v in vals if v is not None and not np.isnan(v)]
            if not vals: return np.nan
            return float(np.mean(vals))
        print("평균 σ_px(px):", np.nanmean([m["sigma_px"] for m in per_id.values()]))
        print("평균 σ_Z(mm):", 1e3*np.nanmean([m["z_std"] for m in per_id.values()]))
        print("평균 AbsErr(mm):", 1e3*np.nanmean([m["abs_err"] for m in per_id.values()]))

    # ---------- 색상 없음(타일 기반) 결과 ----------
    if not color_available or len(tracks)==0:
        if len(tile_depth_stats)==0:
            print("깊이 통계가 비었습니다. 장치/프레임/ROI를 확인하세요.")
            return
        arr = np.array(tile_depth_stats, dtype=object)  # tile_id, meanZ, stdZ, plane_rms
        meanZ = arr[:,1].astype(float)
        stdZ  = arr[:,2].astype(float)
        plane = arr[:,3].astype(float)
        absErr = np.abs(meanZ - args.z_true)
        print("\n=== Depth-only (타일 기반) ===")
        print(f"평균 σ_Z(mm): {1e3*np.nanmean(stdZ):.2f}")
        print(f"평균 AbsErr(mm): {1e3*np.nanmean(absErr):.2f}")
        print(f"평균 plane_rms(mm): {1e3*np.nanmean(plane):.2f}")

if __name__ == "__main__":
    main()
'''
python camera_lens_distortion.py --z_true 0.35 --frames 120 --show

# result

=== Depth-only (타일 기반) ===
평균 σ_Z(mm): 0.86
평귀 AbsErr(mm): 4.97
평귀 plane_rms(mm): 1.70

'''