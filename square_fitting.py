import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects

from EvaluationCNN.detect import predict_image_quality
from keypoint.detect import key_detect
from mol_segment.detect import segmented_image

matplotlib.use('Agg')

def generate_square_vertices(center, side_length, angle):
    """
    生成正方形的四个顶点坐标。

    参数:
    center (tuple): 正方形中心点坐标 (x, y)。
    side_length (float): 正方形的边长。
    angle (float): 正方形的旋转角度（以度为单位）。

    返回:
    numpy.ndarray: 正方形的四个顶点坐标。
    """
    half_side = side_length / 2

    # 定义未旋转的正方形顶点
    vertices = np.array([
        [-half_side, -half_side],
        [half_side, -half_side],
        [half_side, half_side],
        [-half_side, half_side]
    ])

    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # 旋转顶点
    rotated_vertices = np.dot(vertices, rotation_matrix)

    # 平移顶点到中心点
    rotated_vertices += center

    return rotated_vertices

def calculate_orientation(mask):
    # 查找mask的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 获取最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    
    # 获取矩形的顶点
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # 计算角度
    angle = rect[-1]
    
    # 调整角度范围
    if angle < -45:
        angle += 90
    
    return angle


def process_reaction_points(image, 
                            key_point_result, 
                            img_scale=1000, 
                            mol_scale=200, 
                            Br_size=60, 
                            orientation_angle=0, 
                            site_model_path = "./model/site_model.pth"
                            ):
    image_pix_edge = image.shape[0]
    Br_site_edge = image_pix_edge / (img_scale / Br_size)
    mol_edge = image_pix_edge / (img_scale / mol_scale)
    
    # 1. Calculate the molecule's reaction points
    molecule_center = np.array(key_point_result[0][5:7]) * image_pix_edge
    half_size = mol_edge / 2
    reaction_points = [
        (molecule_center[0] - half_size, molecule_center[1] + half_size),
        (molecule_center[0] + half_size, molecule_center[1] + half_size),
        (molecule_center[0] + half_size, molecule_center[1] - half_size),
        (molecule_center[0] - half_size, molecule_center[1] - half_size)
    ]
    
    # 2. Rotate the reaction points according to the molecule's orientation
    orientation_angle = -1 * orientation_angle
    rotation_matrix = cv2.getRotationMatrix2D(molecule_center, orientation_angle, 1)
    rotated_reaction_points = []
    for pt in reaction_points:
        rotated_pt = cv2.transform(np.array([[pt]], dtype=np.float32), rotation_matrix)[0][0]
        rotated_reaction_points.append(rotated_pt)
    
    # 3. Prepare patches
    patch_size = round(Br_site_edge)
    patches = []
    Br_site_state_list = []
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    for i, pt in enumerate(rotated_reaction_points):
        # Calculate the square area around each reaction point
        x, y = int(pt[0]), int(pt[1])
        x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        x2, y2 = min(image.shape[1], x + patch_size // 2), min(image.shape[0], y + patch_size // 2)
        
        # Extract patch
        patch = image[y1:y2, x1:x2]
        # Predict the reaction site state
        predict = predict_image_quality(patch, site_model_path)
        Br_site_state_list.append(predict)
        
        patches.append((patch, (x1, y1, x2, y2), predict))
    
    return rotated_reaction_points, patches, Br_site_state_list, time_stamp


# def visualize_reaction_points(image, rotated_reaction_points, patches, save_dir="./reaction_patches", save_patch=False):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(image, cmap='gray')
    
#     # Draw molecule center
#     molecule_center = np.array(rotated_reaction_points[0])  # Assuming molecule center is the first reaction point
#     ax.scatter(molecule_center[0], molecule_center[1], color='red', label="Center", zorder=5)
    
#     # Draw reaction points
#     for pt in rotated_reaction_points:
#         ax.scatter(pt[0], pt[1], color='blue', label="Br_site", zorder=5)
    
#     # Draw the molecule's bounding box
#     molecule_square = plt.Polygon(rotated_reaction_points, fill=None, edgecolor='yellow', linewidth=2)
#     ax.add_patch(molecule_square)

#     ax.set_title("Molecule and Reaction Points Visualization")
#     ax.legend()

#     # Draw rectangles around the reaction points and save patches
#     for i, (patch, (x1, y1, x2, y2), predict) in enumerate(patches):
#         box_color = 'green' if predict < 0.3 else 'red'
#         rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, edgecolor=box_color, linewidth=1)
#         ax.text(x1, y1, f"{i+1}", fontsize=12, color=box_color)
#         ax.add_patch(rect)
        
#         if save_patch:
#             patch_filename = os.path.join(save_dir, f"reaction_point_{i+1}_{time_stamp}.png")
#             cv2.imwrite(patch_filename, patch)

#     time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
#     plt.savefig(save_dir + f"/reaction_point_{time_stamp}.png")
#     plt.clf()

def visualize_reaction_points(image, rotated_reaction_points, patches, save_dir="./reaction_patches", save_patch=False):
    # 创建一个彩色图像用于绘制
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 绘制分子中心
    molecule_center = np.array(rotated_reaction_points[0])  # Assuming molecule center is the first reaction point
    cv2.circle(vis_img, (int(molecule_center[0]), int(molecule_center[1])), 5, (0, 0, 255), -1)  # Red color
    
    # 绘制反应点
    for pt in rotated_reaction_points:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)  # Blue color
    
    # 绘制分子的边界框
    pts = np.array(rotated_reaction_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow color

    # 绘制反应点周围的矩形框并保存patches
    for i, (patch, (x1, y1, x2, y2), predict) in enumerate(patches):
        box_color = (0, 255, 0) if predict < 0.3 else (0, 0, 255)  # Green if predict < 0.3 else Red
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 1)
        cv2.putText(vis_img, f"{i+1}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
        
        if save_patch:
            patch_filename = os.path.join(save_dir, f"reaction_point_{i+1}_{time_stamp}.png")
            cv2.imwrite(patch_filename, patch)

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir, f"reaction_point_{time_stamp}.png"), vis_img)

def process_mol_orientation(mask, 
                            min_obj_size=100, 
                            hole_area_threshold=500, 
                            close_kernel_size=20, 
                            ):
    """
    This function processes the input mask and returns the calculated angle and relevant intermediate results.
    
    Parameters:
    mask               : Binary mask image (H, W), can be 0/1 or 0/255.
    min_obj_size       : Minimum size of objects to keep.
    hole_area_threshold: Threshold for filling small holes.
    close_kernel_size  : Kernel size for morphological closing.
    
    Returns:
    angle_degrees      : The calculated orientation angle (degrees).
    largest_mask       : The largest connected component mask.
    largest_contour    : The largest contour.
    box                 : The minimum enclosing rectangle around the largest contour.
    """
    
    # Convert mask to 0/1
    if mask.max() > 1:
        bin_mask = (mask > 0).astype(np.uint8)
    else:
        bin_mask = mask.astype(np.uint8)
    raw_mask = bin_mask.copy()

    # Remove small objects, fill holes, and perform morphological closing
    bool_mask = (bin_mask > 0)
    bool_removed_small = remove_small_objects(bool_mask, min_size=min_obj_size)
    bool_filled_holes = remove_small_holes(bool_removed_small, area_threshold=hole_area_threshold)
    pre_mask = bool_filled_holes.astype(np.uint8)

    if close_kernel_size > 1:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        pre_mask_closed = cv2.morphologyEx(pre_mask, cv2.MORPH_CLOSE, kernel)
    else:
        pre_mask_closed = pre_mask.copy()

    # Connected components analysis to keep the largest component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pre_mask_closed, connectivity=8)
    if num_labels <= 1:
        largest_mask = np.zeros_like(pre_mask_closed)
    else:
        max_area = 0
        max_label = 0
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = label_id
        largest_mask = (labels == max_label).astype(np.uint8)

    # Find contours and calculate the main orientation using minAreaRect
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        angle_degrees = None
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        angle = rect[-1]
        if angle < 0:
            angle += 90
        if angle >= 90:
            angle -= 90
        if angle == 0:
            angle = 0.01
        angle_degrees = angle

    return angle_degrees, largest_mask, largest_contour, box

def visualize_mol_orientation(mask, 
                            angle_degrees, 
                            largest_mask, 
                            largest_contour, 
                            box, 
                            key_point_result, 
                            img_scale="10n",
                            mol_scale="2n",
                            square_save_path="./",
                            ):
    """
    This function visualizes the mask processing and orientation results.

    Parameters:
    mask               : The binary mask image.
    angle_degrees      : The calculated orientation angle.
    largest_mask       : The largest connected component mask.
    largest_contour    : The largest contour found.
    box                 : The minimum enclosing rectangle around the largest contour.
    key_point_result   : Key points used for the molecular box.
    square_save_path   : Directory path for saving the visualizations.
    """
    
    # Visualization of the processing steps
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    # Display the raw mask
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('0) Raw Mask')
    axes[0].axis('off')

    # Display the largest connected component mask
    axes[4].imshow(largest_mask, cmap='gray')
    axes[4].set_title('4) Largest Component')
    axes[4].axis('off')

    # Visualization of the minimum enclosing rectangle and molecule box
    vis_img = cv2.cvtColor(largest_mask * 255, cv2.COLOR_GRAY2BGR)
    if angle_degrees is not None:
        # Draw the largest contour
        cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 0), 2)
        # Draw the enclosing rectangle
        cv2.drawContours(vis_img, [box], -1, (0, 0, 255), 2)
        # Draw the molecule box
        mask_edge_pix = mask.shape[0]
        mol_box_length = mask_edge_pix / (img_scale / mol_scale)
        mol_box = generate_square_vertices(tuple(np.array(key_point_result[0][5:7]) * mask_edge_pix), mol_box_length, -angle_degrees)
        mol_box = mol_box.astype(np.int32)
        cv2.drawContours(vis_img, [mol_box], -1, (255, 0, 0), 2)

        axes[5].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        axes[5].set_title(f'5) Orientation={angle_degrees:.2f}°')
        axes[5].axis('off')

        # Draw an arrow indicating the orientation angle
        (cx, cy), _, _ = cv2.minAreaRect(largest_contour)
        dx = np.cos(np.radians(angle_degrees))
        dy = np.sin(np.radians(angle_degrees))
        axes[5].arrow(cx, cy, dx * 20, dy * 20, color='r', width=2)
    else:
        axes[5].set_title('5) No valid contour')
        axes[5].axis('off')

    # Save the visualization
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not os.path.exists(square_save_path):
        os.makedirs(square_save_path)

    plt.tight_layout()
    plt.savefig(square_save_path + f"/{time_stamp}.png")
    plt.clf()



def mol_Br_site_detection(img, mask, key_point_result, img_scale, mol_scale, Br_scale, square_save_path,site_model_path,visualize=True):
    
    if len(mask.shape) == 3:
        mask = mask.squeeze().astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize the mask to (208,208)
    mask = cv2.resize(mask, (208, 208), interpolation=cv2.INTER_NEAREST)
    
    angle, largest_mask, largest_contour, box = process_mol_orientation(
                            mask,
                            min_obj_size=200,
                            hole_area_threshold=200,
                            close_kernel_size=10,
                            )
    if visualize:
        visualize_mol_orientation(mask, 
                                angle, 
                                largest_mask, 
                                largest_contour, 
                                box, 
                                key_point_result, 
                                img_scale=img_scale,
                                mol_scale=mol_scale,
                                square_save_path=square_save_path,
                                )
        # print(f"Estimated orientation angle = {angle}")

    rotated_reaction_points, patches, Br_site_state_list, time_stamp = process_reaction_points(img, 
                            key_point_result, 
                            img_scale=img_scale, 
                            mol_scale=mol_scale, 
                            Br_size=Br_scale, 
                            orientation_angle=angle, 
                            site_model_path = site_model_path
                            )
    if visualize:
        visualize_reaction_points(img, rotated_reaction_points, patches, save_dir = square_save_path, save_patch=False)

    
    return angle, Br_site_state_list, patches
# =======================
# 使用示例
if __name__ == "__main__":



    # image_path = './mol_segment/058_test/state_0/367_5.png'
    input_folder = './mol_segment/058_test/test'
    segment_output_folder = './mol_segment/results1'
    segment_model_path = './mol_segment/unet_model-zzw-Ni_V2.pth'
    keypoint_output_folder = './keypoint/results1'
    keypoint_model_path = './keypoint/best.pt'  
    site_model_path = './EvaluationCNN/CNN_Br_V2.pth'
    square_save_path = './single_mol_results'
    
    img_scale = 500  # 1000 → 10nm
    mol_scale = 160  # 180 → 1.8nm
    Br_scale = 180  # 180 → 1.8nm

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)

            img_key = cv2.imread(image_path)
            # img = Image.open(image_path).convert('RGB')
            img = img_key
            mask,_,_ = segmented_image(img, segment_output_folder,segment_model_path)
            key_point_result = key_detect(img_key, keypoint_model_path, keypoint_output_folder)
            # print(key_point_result)

            #convert the img to 1 channel
            if len(mask.shape) == 3:
                mask = mask.squeeze().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # resize the mask to (208,208)
            mask = cv2.resize(mask, (208, 208), interpolation=cv2.INTER_NEAREST)
            
            angle, Br_site_state_list, patches = mol_Br_site_detection(img, mask, key_point_result, img_scale, mol_scale, Br_scale, square_save_path, site_model_path)

            print(Br_site_state_list)



