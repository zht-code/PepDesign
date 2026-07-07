# copy_peptides_to_cands.py
import os
import shutil
import warnings

# 可选进度条
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x  # 无 tqdm 时仍可运行

warnings.filterwarnings("ignore")

def main():
    # 根目录（按需修改）
    train_root = "/root/autodl-tmp/train_data"
    overwrite = True  # True=覆盖已存在文件；False=存在则跳过

    if not os.path.isdir(train_root):
        raise SystemExit(f"[ERR] train_root not found: {train_root}")

    subdirs = sorted(
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    )

    copied = skipped = missing = 0

    for sub in tqdm(subdirs, desc="Copy peptide.pdb → cands", unit="dir"):
        dir_path = os.path.join(train_root, sub)
        src = os.path.join(dir_path, "peptide.pdb")
        if not os.path.isfile(src):
            missing += 1
            continue

        dst_dir = os.path.join(dir_path, "cands")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "peptide.pdb")

        if (not overwrite) and os.path.exists(dst):
            skipped += 1
            continue

        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            skipped += 1
            print(f"[WARN] Failed to copy {src} → {dst}: {e}")

    print(f"[DONE] copied={copied}, skipped={skipped}, missing_source={missing}")

if __name__ == "__main__":
    main()
