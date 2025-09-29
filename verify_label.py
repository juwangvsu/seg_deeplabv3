import os, glob, cv2, numpy as np, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--num_classes", type=int, default=21)
    ap.add_argument("--ignore_index", type=int, default=255)
    args = ap.parse_args()
    bad_files = 0
    for mp in sorted(glob.glob(os.path.join(args.masks, "*.png"))):
        m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        vals = np.unique(m)
        bad = vals[(vals != args.ignore_index) & ((vals < 0) | (vals > args.num_classes-1))]
        if len(bad):
            bad_files += 1
            print(f"[BAD] {mp}: bad labels {bad.tolist()}")
    if bad_files == 0:
        print("All masks valid.")
    else:
        print(f"{bad_files} mask file(s) contain invalid labels.")
if __name__ == "__main__":
    main()

