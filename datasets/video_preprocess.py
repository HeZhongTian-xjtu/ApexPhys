import cv2
import os
import argparse
import sys
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.util import crop_faces
from scipy.io import loadmat


def video_preprocess(dataset="", root_dir=""):
    IMG_EXTENSIONS = [
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG",
        ".ppm",
        ".PPM",
        ".bmp",
        ".BMP",
        ".tiff",
    ]

    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(dir):
        images = []
        assert os.path.isdir(dir), "%s is not a valid directory" % dir
        fnames = os.listdir(dir)
        for fname in fnames:
            if fname.split(".")[0][-1] == ")":
                os.remove(os.path.join(dir, fname))
                print(
                    f"---------  remove {os.path.join(dir, fname)} for endwith ([0-9])  ---------"
                )
        for fname in sorted(os.listdir(dir), key=lambda x: int(x.split(".")[0])):
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                fname = fname.split(".")[0]
                images.append((fname, path))
        return images

    def make_dataset_from_video(video):
        # images: list of length T, each value is [00001, (H,W,3)]
        images = []
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fname = f"{i:05d}"
            if ret:
                images.append((fname, frame))
        cap.release()
        return images

    if dataset == "VIPL":
        for p in sorted(os.listdir(root_dir)):
            if not p.startswith("p"):
                continue
            p_path = os.path.join(root_dir, p)
            for v in sorted(os.listdir(p_path)):
                v_path = os.path.join(p_path, v)
                for source in sorted(os.listdir(v_path)):
                    if source == "source4":
                        print(f"jump over {v_path}/source4 -----------------")
                        continue
                    save_path = os.path.join(v_path, source, "align_crop_pic")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    elif os.path.exists(os.path.join(v_path, source, "pic")):
                        if len(os.listdir(save_path)) == len(
                            os.listdir(os.path.join(v_path, source, "pic"))
                        ):
                            print(f"already processed {save_path} -----------------")
                            continue

                    source_path = os.path.join(v_path, source)
                    video_path = os.path.join(source_path, "video.avi")
                    print(f"processing {video_path}  ...")

                    files = make_dataset_from_video(video_path)

                    image_size = 128
                    # scale = 1.0  # align only
                    scale = 0.8  # align + crop
                    center_sigma = 1.0
                    xy_sigma = 3.0
                    use_fa = False

                    crops, orig_images, quads = crop_faces(
                        image_size, files, scale, center_sigma, xy_sigma, use_fa
                    )

                    if crops is None:
                        print(
                            f"too less face detected in video {video_path} -----------------"
                        )
                        continue

                    for i in range(len(crops)):
                        img = crops[i]
                        img.save(os.path.join(save_path, files[i][0] + ".png"))
                    print(f"generated {len(crops)} png images in file {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="VIPL", help="VIPL"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="PATH_TO_DATASET", help="dataset dir"
    )
    args = parser.parse_args()
    video_preprocess(args.dataset, args.dataset_dir)
