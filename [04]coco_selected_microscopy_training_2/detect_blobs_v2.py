import os
import cv2
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree

def detect_ellipses_watershed(image, visualize=False, save_path=None):
    # 1. Preprocess
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Morphology
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background and foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4. Markers and watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)

    ellipses = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                ellipses.append(ellipse)
                if visualize:
                    cv2.ellipse(color_img, ellipse, (0, 255, 0), 2)

    # Show or save
    if visualize:
        if save_path:
            cv2.imwrite(save_path, color_img)
        else:
            cv2.imshow("Ellipses", color_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return ellipses

def generate_cvat_xml(image_folder, output_file="cvat_ellipses.xml"):
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    annotations = Element("annotations")
    SubElement(annotations, "version").text = "1.1"
    meta = SubElement(annotations, "meta")
    SubElement(meta, "task")
    SubElement(meta, "labels")
    SubElement(meta, "original_size")

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

        save_vis = os.path.join("[04]coco_selected_microscopy_training_2", "visualized", f"{os.path.splitext(fname)[0]}_ellipses.png")

        os.makedirs(os.path.join("[04]coco_selected_microscopy_training_2", "visualized"), exist_ok=True)
        ellipses = detect_ellipses_watershed(img, visualize=True, save_path=save_vis)

        for (cx, cy), (rx, ry), angle in ellipses:
            SubElement(image_tag, "ellipse", {
                "label": "1",
                "cx": f"{cx:.2f}",
                "cy": f"{cy:.2f}",
                "rx": f"{rx/2:.2f}",
                "ry": f"{ry/2:.2f}",
                "occluded": "0",
                "z_order": "0",
                "rotation": f"{angle:.2f}"
            })

    tree = ElementTree(annotations)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"[DONE] Wrote {output_file} with {len(image_files)} images and ellipses")



# === RUN ===
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
    generate_cvat_xml(IMAGE_FOLDER, output_file="ellipses_cvat.xml")