import os
import sys
import cv2
from tqdm import tqdm

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg'}


def find_video_files(root_dir):
    video_files = []
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))

    with tqdm(all_files, desc="Scanning directory", unit="file", colour="cyan") as scan_bar:
        for filepath in scan_bar:
            if os.path.splitext(filepath)[1].lower() in VIDEO_EXTENSIONS:
                video_files.append(filepath)
            scan_bar.set_postfix(found=len(video_files))

    return sorted(video_files)


def select_files(video_files):
    print("\nFound video files:")
    for i, path in enumerate(video_files):
        print(f"  [{i + 1}] {path}")

    print("\nEnter file numbers to process (e.g. 1,3,5), or 'all' to process all:")
    choice = input("> ").strip().lower()

    if choice == 'all':
        return video_files

    selected = []
    for part in choice.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(video_files):
                selected.append(video_files[idx])
            else:
                print(f"  Warning: index {part} out of range, skipping.")
        else:
            print(f"  Warning: '{part}' is not a valid number, skipping.")

    return selected


DEFAULT_FRAME_SKIP = 5


def prompt_frame_skip():
    print(f"\nFrame skip — save every Nth frame (default: {DEFAULT_FRAME_SKIP}):")
    print(f"  e.g. 1 = every frame, 5 = every 5th frame, 10 = every 10th frame")
    raw = input(f"  Enter value or press Enter to use default [{DEFAULT_FRAME_SKIP}]: ").strip()
    if raw == "":
        print(f"  Using default frame skip: {DEFAULT_FRAME_SKIP}")
        return DEFAULT_FRAME_SKIP
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    print(f"  Invalid input '{raw}', using default frame skip: {DEFAULT_FRAME_SKIP}")
    return DEFAULT_FRAME_SKIP


def extract_frames(video_path, overall_bar, frame_skip):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join("out", video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(f"  [ERROR] Could not open: {video_path}")
        overall_bar.update(1)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    estimated_saved = (total_frames + frame_skip - 1) // frame_skip

    tqdm.write(f"\n  Video      : {video_path}")
    tqdm.write(f"  Frames     : {total_frames}  |  FPS: {fps:.2f}  |  Skip: every {frame_skip} frame(s)")
    tqdm.write(f"  Saving ~   : {estimated_saved} frames  |  Output: {out_dir}")

    frame_idx = 0
    saved_count = 0
    pad = len(str(estimated_saved))
    with tqdm(
        total=total_frames,
        desc=f"  {video_name[:30]:<30}",
        unit="frame",
        unit_scale=False,
        colour="green",
        dynamic_ncols=True,
        leave=True,
        postfix={"saved": 0},
    ) as frame_bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                frame_name = f"frame_{saved_count + 1:0{pad}d}.png"
                cv2.imwrite(os.path.join(out_dir, frame_name), frame)
                saved_count += 1
                frame_bar.set_postfix(saved=saved_count)
            frame_idx += 1
            frame_bar.update(1)

    cap.release()
    tqdm.write(f"  ✓ {saved_count} frames saved (scanned {frame_idx}) → {out_dir}\n")
    overall_bar.update(1)
    return saved_count


def main():
    if len(sys.argv) < 2:
        root_dir = input("Enter the directory path to scan for videos: ").strip()
    else:
        root_dir = sys.argv[1]

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        sys.exit(1)

    video_files = find_video_files(root_dir)

    if not video_files:
        print("No video files found in the specified directory.")
        sys.exit(0)

    selected = select_files(video_files)

    if not selected:
        print("No files selected. Exiting.")
        sys.exit(0)

    frame_skip = prompt_frame_skip()

    print(f"\nExtracting frames for {len(selected)} file(s) (skip={frame_skip})...\n")
    summary = {}

    with tqdm(
        total=len(selected),
        desc="Overall progress",
        unit="video",
        colour="yellow",
        dynamic_ncols=True,
        position=0,
        leave=True,
    ) as overall_bar:
        for video_path in selected:
            count = extract_frames(video_path, overall_bar, frame_skip)
            summary[video_path] = count

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    total_frames_all = 0
    for path, count in summary.items():
        name = os.path.basename(path)
        out_name = os.path.splitext(name)[0]
        print(f"  {name}: {count} frames → out/{out_name}/")
        total_frames_all += count
    print("-" * 55)
    print(f"  Total frames extracted: {total_frames_all}")
    print("=" * 55)


if __name__ == "__main__":
    main()