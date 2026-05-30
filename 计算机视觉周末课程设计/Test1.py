import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from moviepy import VideoFileClip

# ---------------------- 工具函数：显示多张图片 ----------------------
def show_images(images, cmap=None):
    cols = 2
    rows = (len(images) + 1) // cols
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

# ---------------------- 1. 读取并显示原始图片（修改为你的路径） ----------------------
test_images = [plt.imread(path) for path in glob.glob(r"D:\计算机视觉周末课程设计\test_images\*.jpg")]
print(f"读取到 {len(test_images)} 张测试图片")

# ---------------------- 2. HSL颜色空间过滤（白/黄线） ----------------------
def convert_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_hsl_white_yellow(image):
    hsl = convert_hsl(image)
    # 白色线：亮度高
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hsl, lower_white, upper_white)
    # 黄色线：色调+饱和度过滤
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(hsl, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# ---------------------- 3. 灰度化 ----------------------
def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# ---------------------- 4. 高斯滤波去噪 ----------------------
def apply_smoothing(image, ksize=15):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

# ---------------------- 5. Canny边缘检测 ----------------------
def detect_edges(image, low_thresh=50, high_thresh=150):
    return cv2.Canny(image, low_thresh, high_thresh)

# ---------------------- 6. 感兴趣区域（ROI）提取 ----------------------
def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])
    return cv2.bitwise_and(image, mask)

def select_region(image):
    rows, cols = image.shape[:2]
    # 定义梯形ROI顶点（根据图片视角微调）
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

# ---------------------- 7. 霍夫直线检测 ----------------------
def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20,
                           minLineLength=20, maxLineGap=300)

# ---------------------- 8. 车道线拟合 ----------------------
def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue # 忽略垂直线
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # 加权平均，权重为线段长度
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    """将(slope, intercept)格式的直线转换为图像坐标点"""
    if line is None:
        return None
    slope, intercept = line
    # 计算两端点的x坐标
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]  # 图像底部
    y2 = y1 * 0.6        # 车道线上端点
    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    return left_line, right_line

# ---------------------- 9. 绘制车道线（实验文档6.7.3函数） ----------------------
def draw_lane_lines(image, lines, color=(255, 0, 0), thickness=20):
    # 创建和原图大小相同的黑色图像，用于绘制车道线
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    # 将车道线图像与原图加权融合
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0)

# ---------------------- 10. 图片车道线检测并绘制（实验文档6.7.4） ----------------------
# 一次性完成所有处理步骤
processed_images = []
for image in test_images:
    filtered = select_hsl_white_yellow(image)
    gray = convert_gray_scale(filtered)
    blurred = apply_smoothing(gray)
    edges = detect_edges(blurred)
    roi = select_region(edges)
    lines = hough_lines(roi)
    if lines is not None:
        left_line, right_line = lane_lines(image, lines)
        processed_img = draw_lane_lines(image, [left_line, right_line])
        processed_images.append(processed_img)
    else:
        processed_images.append(image)

show_images(processed_images)

# ---------------------- 11. 视频车道线检测流程（实验文档6.8） ----------------------
def Laneline_process(image):
    # 完整的车道线检测流程
    filtered = select_hsl_white_yellow(image)
    gray = convert_gray_scale(filtered)
    blurred = apply_smoothing(gray)
    edges = detect_edges(blurred)
    roi = select_region(edges)
    lines = hough_lines(roi)
    if lines is not None:
        left_line, right_line = lane_lines(image, lines)
        return draw_lane_lines(image, [left_line, right_line])
    return image

# ---------------------- 12. 视频处理与输出（修改为你的路径） ----------------------
# 输入视频路径（和你的文件夹结构完全对应）
input_challenge = r"D:\计算机视觉周末课程设计\test_videos\challenge.mp4"
input_white = r"D:\计算机视觉周末课程设计\test_videos\solidWhiteRight.mp4"
input_yellow = r"D:\计算机视觉周末课程设计\test_videos\solidYellowLeft.mp4" # 你截图里没看到这个文件，先加上

# 输出视频路径（保存在和代码同目录）
output_1 = 'challenge_result.mp4'
output_2 = 'solidWhiteRight_result.mp4'
output_3 = 'solidYellowLeft_result.mp4'

# 处理视频（可选）
if __name__ == "__main__":
    # 处理第一个视频
    clip_1 = VideoFileClip(input_challenge)
    out_clip_1 = clip_1.image_transform(Laneline_process)
    out_clip_1.write_videofile(output_1, audio=False) # 部分视频无音频，audio=False更稳定

    # 处理第二个视频
    clip_2 = VideoFileClip(input_white)
    out_clip_2 = clip_2.image_transform(Laneline_process)
    out_clip_2.write_videofile(output_2, audio=False)

    # 处理第三个视频（如果文件存在）
    import os
    if os.path.exists(input_yellow):
        clip_3 = VideoFileClip(input_yellow)
        out_clip_3 = clip_3.image_transform(Laneline_process)
        out_clip_3.write_videofile(output_3, audio=False)