import os
import cv2
from xml.etree.ElementTree import Element, SubElement, ElementTree

def detect_blobs(image):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 2000
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(image)

def generate_cvat_xml(image_folder, output_file="cvat_ellipses.xml"):
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    annotations = Element("annotations")

    # Meta
    version = SubElement(annotations, "version")
    version.text = "1.1"

    meta = SubElement(annotations, "meta")
    task = SubElement(meta, "task")
    SubElement(task, "labels")
    SubElement(task, "original_size")

    for idx, fname in enumerate(image_files):
        image_path = os.path.join(image_folder, fname)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        image_tag = SubElement(annotations, "image", {
            "id": str(idx),
            "name": fname,
            "width": str(w),
            "height": str(h)
        })

        keypoints = detect_blobs(img)
        for kp in keypoints:
            cx, cy = kp.pt
            rx = ry = kp.size / 2  # Assuming symmetric blob
            ellipse = SubElement(image_tag, "ellipse", {
                "label": "Bead",
                "cx": f"{cx:.2f}",
                "cy": f"{cy:.2f}",
                "rx": f"{rx:.2f}",
                "ry": f"{ry:.2f}",
                "occluded": "0",
                "z_order": "0"
            })

    tree = ElementTree(annotations)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[DONE] Wrote {output_file} with {len(image_files)} images and ellipses")

# === RUN ===
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FOLDER = os.path.join(BASE_DIR, "fft_filtered")
    generate_cvat_xml(IMAGE_FOLDER, output_file="ellipses2_cvat.xml")
