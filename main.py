import cv2
import numpy as np


def clean_edges(edge_img, min_length=40):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(edge_img)

    for cnt in contours:
        length = cv2.arcLength(cnt, False)
        if length >= min_length:
            cv2.drawContours(clean, [cnt], -1, 255, 1)

    return clean


def overlay_edges_on_original(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 70, 140)
    edges = clean_edges(edges, min_length=35)

    # 선을 조금 이어서 더 자연스럽게
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    result = img.copy()
    result[edges != 0] = (0, 0, 0)

    return result, edges


def main():
    input_path = "input.jpg"

    img_array = np.fromfile(input_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print("input.jpg를 불러오지 못했습니다.")
        print("프로젝트 폴더 안에 input.jpg가 있는지 확인하세요.")
        return

    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    result, edges = overlay_edges_on_original(img)

    cv2.imencode(".jpg", result)[1].tofile("cartoon_result.jpg")
    cv2.imencode(".jpg", edges)[1].tofile("edges.jpg")

    cv2.imshow("Original", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("Result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()