import atexit
import logging
import math
import os
import random
import time

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.metrics import silhouette_score
from torch import nn


def weights_init_(m):
    """
    Initialize weights in torch.nn object

    Parameters
    ----------
    m: torch.nn.Linear

    Returns
    -------
    None
    """
    if isinstance(m, nn.Linear):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0)

# 定义二维正态分布
def normal_distribution_2d(x, y, mean_x, mean_y, std_dev_x, std_dev_y):
    """
    二维正态分布概率密度函数，归一化到(0, mean)为1
    """
    norm_factor = (1 / (std_dev_x * std_dev_y * 2 * np.pi))
    exponent = -0.5 * (((x - mean_x) / std_dev_x) ** 2 + ((y - mean_y) / std_dev_y) ** 2)
    normal_value = norm_factor * np.exp(exponent)
    normalization = norm_factor * np.exp(-0.5 * ((0 - mean_x) / std_dev_x) ** 2 + ((0 - mean_y) / std_dev_y) ** 2)
    return normal_value / normalization

# 定义ind_distribution, non_distribution和suc_distribution在二维上的公式
def base_distribution_2d(x, y):
    if x<=0 and y>=0:
        return 1 - normal_distribution_2d(x, 0, 0, 0, 1, 1)+normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    if x<0 and y<0:
        return (1 - normal_distribution_2d(x, 0, 0, 0, 1, 1))+normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    # if x<0 and y>0:
    #     return (1 - normal_distribution_2d(0, y, 0, 0, 1, 1))+normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    else:
        return normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02 


def suc(x, y):
    return normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.96

def ind(x, y):
    if x>=0 and y<=0:
        return 1 - normal_distribution_2d(x, 0, 0, 0, 1, 1) + normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02
    if x<0 and y<0:
        return normal_distribution_2d(x, y, 0, 0, 1, 1) * 0.02

    return 1-base_distribution_2d(x, y) - suc(x, y)

def non(x, y):
        return 1-suc(x, y) - ind(x, y)



def generate_triangle_grid(center, grid_size, side_length):
    """
    Generate a grid of points forming an equilateral triangle grid with the given center, grid size, and side length.

    :param center: Tuple representing the grid center (cx, cy)
    :param grid_size: Tuple (n, m) representing the number of rows (n) and columns (m) in the grid
    :param side_length: The side length of the equilateral triangles
    :return: List of tuples containing the coordinates of all grid points
    """
    cx, cy = center
    n, m = grid_size

    # Height of equilateral triangle, based on side length 'a'
    height = side_length * math.sqrt(3) / 2
    
    # List to store the grid points
    grid_points = []

    for row in range(n):
        for col in range(m):
            # Horizontal shift for odd rows
            if row % 2 == 1:
                x_offset = col * side_length + side_length / 2
            else:
                x_offset = col * side_length
            
            # Vertical offset between rows
            y_offset = row * height

            # Add the calculated coordinates to the grid
            grid_points.append((x_offset, y_offset))
    
    # Scale the grid points by sqrt(3)/2 for y-coordinates
    for i in range(len(grid_points)):
        x, y = grid_points[i]
        grid_points[i] = (x, y * math.sqrt(3) / 2)
    
    # Find the current center of the grid (average of all x and y coordinates)
    current_center_x = sum(x for x, _ in grid_points) / len(grid_points)
    current_center_y = sum(y for _, y in grid_points) / len(grid_points)
    
    # Calculate the translation needed to move the center to the target center
    translate_x = cx - current_center_x
    translate_y = cy - current_center_y

    # Apply the translation to each grid point
    for i in range(len(grid_points)):
        x, y = grid_points[i]
        grid_points[i] = (x + translate_x, y + translate_y)

    return grid_points

def triangle_norm_distribution(position = (0,0) ,value_points=(250,250), size=100, key_points_size_factor=0.6, center_std_devs = 18, center_weights = 1, edge_std_devs = 16, edge_weights = 0.5):

    cx, cy = position
    # the three vertexes of the triangle
    # draw the four key points of the triangle and the center, the key points are 0.8 times the edge length in sdie of the vertexes
    key_points = np.array([[cx, cy - math.sqrt(3) / 3 * size * key_points_size_factor], 
                           [cx - size / 2 * key_points_size_factor, cy + math.sqrt(3) / 6 * size * key_points_size_factor], 
                           [cx + size / 2 * key_points_size_factor, cy + math.sqrt(3) / 6 * size * key_points_size_factor], 
                           [cx, cy]])
    
    centers = key_points
    means = ([0, 0], [0, 0], [0, 0], [0, 0])
    std_devs = ([edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [edge_std_devs, edge_std_devs], [center_std_devs, center_std_devs])
    weights = [edge_weights, edge_weights, edge_weights, center_weights]

    x = np.linspace(-size*2, size*2, size*4)
    y = x
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    value = 0

        # 计算每个正态分布并叠加
    for center, mean, std_dev, weight in zip(centers, means, std_devs, weights):
        x0, y0 = center
        mean_x, mean_y = mean
        std_x, std_y = std_dev
        
        # 计算二维正态分布
        Z += (1 / (2 * np.pi * std_x * std_y)) * np.exp(
            -(((X - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
              ((Y - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
        )
    
        value += (1 / (2 * np.pi * std_x * std_y)) * np.exp(
            -(((value_points[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
              ((value_points[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
        )
    # Z = Z / np.max(Z)
    value = value / np.max(Z)

    return value


def calculate_motion_vector(image_t, image_t1, threshold=0.001, cluster_arg = 'DBSCAN', nOctaves=8, nOctaveLayers=8, max_k=20):
    """
    Calculates the motion vector between two images (at time t and t+1)
    using the A-KAZE feature detection algorithm and feature matching.
    
    :param image_t: Remote sensing image at time t
    :param image_t1: Remote sensing image at time t+1
    :param scale: The scale of the remote sensing image (default is 1:500)
    :return: Motion vector (dx, dy) in real-world units (meters)
    """
    # get the heigh and width of the image_t
    height, width = image_t.shape
    # initialize the good_matches
    good_matches = []
    threshold = threshold*10
    while len(good_matches) < 4:
        threshold = threshold/10
        # 1. Initialize the A-KAZE detector
        good_matches = []
        akaze = cv2.AKAZE_create(threshold=threshold, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers)
        
        # 2. Detect keypoints and descriptors in both images
        kp1, des1 = akaze.detectAndCompute(image_t, None)
        kp2, des2 = akaze.detectAndCompute(image_t1, None)
        
        # 3. Use a brute force matcher to match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 4. Apply Lowe's Ratio Test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        # 6. Extract the matched keypoints
        if len(good_matches) < 4:
            print("Not enough matches found!\nAuto adjust the threshold and try again!\nPleas wait a moment!")
            threshold = threshold/10
        else:
            break

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # 6. Combine the source and destination points into a single array
    displacement = dst_pts - src_pts
    
    # 7. Use K-means or DBSCAN to find the largest cluster of matched points and its centroid
    if cluster_arg == 'DBSCAN':
        # Adaptive EPS, taking one twentieth of the average length and width of the image
        eps = height/20
        max_cluster_points, index = find_largest_cluster_DBSCAN(displacement, eps=eps, min_samples=2)
    if cluster_arg == 'KMeans': 
        max_cluster_points, index = find_largest_cluster_K(displacement, max_k=max_k)

    # 8. filter the good_matches by the index
    cluster_matches = [good_matches[i] for i in index[0]]
    # 7. Estimate the motion vector using the average displacement
    centroid = np.mean(max_cluster_points, axis=0)

    good_matches_point = []
    for match in good_matches:
        src_pt = kp1[match.queryIdx].pt
        dst_pt = kp2[match.trainIdx].pt
        good_matches_point.append((dst_pt[0] - src_pt[0], dst_pt[1] - src_pt[1]))
    # 7. Convert pixel displacement to real-world units using the scale
    # 1 pixel represents 500 meters / pixel (scale = 1:500)
    # real_world_displacement = avg_displacement * scale
    
    # 8. Return the motion vector as a tuple (dx, dy) in meters, dx, dy is the persetage of the image size
    # dx = avg_displacement[0] / image_t.shape[1]
    # dy = avg_displacement[1] / image_t.shape[0]
    # dx, dy = avg_displacement[0], avg_displacement[1]
    return centroid, kp1, kp2, good_matches, good_matches_point, cluster_matches, max_cluster_points

def find_largest_cluster_K(points, max_k=10):
    """
    Finds the centroid of the largest cluster among a set of points using K-means.
    
    :param points: List of binary coordinates (list of tuples or a 2D array)
    :param max_k: Maximum number of clusters to try (default 10)
    :return: Centroid of the largest cluster as a tuple (x_centroid, y_centroid)
    """
    points = np.array(points)  # Convert points to numpy array
    
    # 1. Use the Elbow Method or Silhouette Score to estimate the best number of clusters (k)
    best_k = 1
    best_score = -1
    best_model = None
    
    # Try different values for k (from 2 to max_k)
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(points)
        
        # Calculate silhouette score (higher is better)
        score = silhouette_score(points, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
    
    # Now we have the best model with best_k clusters
    labels = best_model.labels_
    
    # 2. Identify the largest cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    
    # 3. Extract points that belong to the largest cluster
    cluster_points = points[labels == largest_cluster_label]
    # get the index of the cluster_points in the points
    index = np.where(labels == largest_cluster_label)

    # Return the cluster_points
    return cluster_points,index

def find_largest_cluster_DBSCAN(points, eps=1.0, min_samples=2):
    """
    Finds the centroid of the largest cluster among a set of points using DBSCAN.
    
    :param points: List of binary coordinates (list of tuples or a 2D array)
    :param eps: The maximum distance between two points to be considered in the same cluster (DBSCAN parameter)
    :param min_samples: The minimum number of points to form a cluster (DBSCAN parameter)
    :return: Centroid of the largest cluster as a tuple (x_centroid, y_centroid)
    """
    # Convert the list of points to a numpy array if it's not already
    points = np.array(points)
    
    # 1. Apply DBSCAN to cluster the points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    
    # 2. Find the labels for the clusters
    labels = clustering.labels_
    
    # -1 label means noise points (not part of any cluster)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    if len(unique_labels) == 0:
        print("No clusters found!")
        return None
    
    # 3. Identify the largest cluster
    largest_cluster_label = unique_labels[np.argmax(counts)]
    
    # 4. Extract points that belong to the largest cluster
    cluster_points = points[labels == largest_cluster_label]
    
    # get the index of the cluster_points in the points
    index = np.where(labels == largest_cluster_label)
    
    # Return the centroid as a tuple (x_centroid, y_centroid)
    return cluster_points,index

prefixes = {'a':-18, 'f':-15, 'p':-13, 'n': -9, 'μ': -6, 'm': -3, '': 0, 'k': 3, 'M': 6, 'G': 9}

def sci_to_unit(value):
    """Convert a value to a string with units"""
    exp = int(np.floor(np.log10(abs(value))))
    prefix = next((k for k,v in prefixes.items() if v <= exp), '')

    scaled = value * 10 ** (-exp)
    return f"{scaled:.7g}{prefix}"

def unit_to_sci(string):
    """Convert a string with units to a float value"""
    for prefix, exp in prefixes.items():
        if string.endswith(prefix):
            number = float(string[:-len(prefix)])
            return number * 10 ** exp
    
    return float(string)


# #a function to subtract a fitted plane from a 2D array
# def subtract_plane(arr):
#     # Fit plane to points
#     A = np.c_[arr[:,0], arr[:,1], np.ones(arr.shape[0])]
#     C, _, _, _ = lstsq(A, arr[:,2])   
    
#     # Evaluate plane on grid
#     ny, nx = arr.shape[:2]
#     x = np.arange(nx)
#     y = np.arange(ny)
#     xv, yv = np.meshgrid(x, y)
#     z = C[0]*xv + C[1]*yv + C[2]
    
#     # Subtract plane from original array
#     return arr - z




#def a function to fit the line scan data array as a line and return a array that subtract the line
def fit_line(data):
    # Fit a linear model y = m*x + b 
    # m = slope, b = intercept
    m, b = np.polyfit(np.arange(len(data)), data, 1)
    
    # Generate the fitted line
    fit = m*np.arange(len(data)) + b
    
    # Compute residuals
    residuals = data - fit
    
    return residuals


# Calculate similarity between two vectors
def vector_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cos_sim


# Calculate the gap in a 1D array
def linescan_max_min_check(arr):
    arr = np.array(arr)
    max_val = np.max(arr)
    min_val = np.min(arr)
    return max_val - min_val

# 

def linescan_similarity_check(vec1, vec2, threshold=0.9):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    similarity = vector_similarity(vec1, vec2)
    # print('linescan_similarity' + str(similarity))
    if similarity < threshold:
        # print('linescan_similarity' + str(similarity)+'bad')
        return 0    # 0 means the similarity is too low
    else:
        return 1    # 1 means the similarity is acceptable

# Calculate the phase difference between two signals   
def find_phase_diff(sig1, sig2):
    corr = correlate(sig1, sig2)
    max_index = np.argmax(corr)
    # print('phase_diff' + str(max_index - len(sig1) + 1))
    return max_index - len(sig1) + 1 

# def a function to calculate the angle between two points vector and the x axis, 90 degree is -Y axis 180 degree is -X axis
def angle_calculate(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    angle = -1*(np.math.atan2(dy, dx) * 180 / np.pi)
    
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
        
    return angle

# def a function to change the Hexadecimal color to BGR color, Hexadecimal color input is a string, BGR color output is a tuple
def Hex_to_BGR(Hex):
    Hex = Hex.strip('#')
    RGB = tuple(int(Hex[i:i+2], 16) for i in (0, 2, 4))
    BGR = (RGB[2], RGB[1], RGB[0]) # cv2 use BGR
    return BGR

def BGR_to_Hex(rgb_color):
    """ Convert RGB to hexadecimal. """
    return f"#{rgb_color[2]:02x}{rgb_color[1]:02x}{rgb_color[0]:02x}"

def interpolate_colors(color_a, color_b, t):
    """ Interpolate between two hexadecimal colors with a ratio t from 0 to 1, using BGR format. """
    if not (0 <= t <= 1):
        raise ValueError("The interpolation parameter t must be between 0 and 1.")

    bgr_a = Hex_to_BGR(color_a)
    bgr_b = Hex_to_BGR(color_b)

    # Calculate the interpolated BGR values
    interpolated_bgr = tuple(int(a + (b - a) * t) for a, b in zip(bgr_a, bgr_b))

    # Convert the BGR back to hexadecimal
    return interpolated_bgr

# def a function to transform the center egde to the left top point and right bottom point of the square
def center_to_square(center, edge):
    x, y = center
    x1 = round(x - 0.5*edge)
    y1 = round(y - 0.5*edge)
    x2 = round(x + 0.5*edge)
    y2 = round(y + 0.5*edge)
    return (x1, y1), (x2, y2)

# input: circle_i and circle_j are two tuple, (x, y, r), x, y is the center of the circle, r is the radius of the circle
def circle_intersection(circle_i, circle_j):
    
    if len(circle_i) == 4 or len(circle_j) == 4:
        x0, y0, r0, _ = circle_i
        x1, y1, r1, _ = circle_j
    elif len(circle_i) == 3 or len(circle_j) == 3:
        x0, y0, r0 = circle_i
        x1, y1, r1 = circle_j
    else:
        raise ValueError("circle_i and circle_j must be tuple of length 3 or 4")
    d = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    if d > r0 + r1:
        return None
    elif d < abs(r0 - r1):
        return None
    elif d == 0 and r0 == r1:
        return None
    else:
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = math.sqrt(r0**2 - a**2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d
        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d
        return (x3, y3), (x4, y4)

def closest_points_np(all_points, reference_point, X):
    # 将点列表和参考点转换为NumPy数组
    points_array = np.array(all_points)
    ref_point_array = np.array(reference_point)

    # 计算每个点与参考点的欧氏距离
    distances = np.sqrt(np.sum((points_array - ref_point_array) ** 2, axis=1))

    # 获取最近的X个点的索引
    closest_indices = np.argsort(distances)[:X]

    # 返回最近的X个点
    return points_array[closest_indices]

# def a function to check the point that is outside the circle in the circle_list
def point_out_circles(point, circles):
  (Xi, Yi) = point

  for x, y, r, _ in circles:
    distance = math.hypot(Xi - x, Yi - y)
    if distance <= r-1:
      return False
  return True

# def a function to check the point that is inside the circle in the 
def point_in_circles(point, circles):
  (Xi, Yi) = point

  for x, y, r, _ in circles:
    distance = math.hypot(Xi - x, Yi - y)
    if distance <= r-1:
      return True
  return False

def get_unique_coords(A, B):
    result = []
    for a in A:
        if not any(np.array_equal(a, b) for b in B):
            result.append(a)
    return np.array(result)
    # def a state function, Determine the Preselected circles at time tn+1 by virtue of its scan range at time t1, t2.....tn
    # input: circle_list is a list of tuple, (x, y, r), x, y is the center of the circle, r is the radius of the circle
    # output: Next point of tuple, (x, y), x, y is the center of the Next circle

def Next_inter(circle_list, plane_size=2000):
    # check if the circle_list is empty
    inter_closest = (round(plane_size/2), round(plane_size/2))
    X1, Y1 = inter_closest
    if not circle_list:
        return inter_closest
    # check if the circle_list has only one circle
    if len(circle_list) == 1:
        random_0to1 = random.random()
        X2 = round(plane_size/2 + circle_list[0][2] * math.cos(random_0to1 * 2 * math.pi))
        Y2 = round(plane_size/2 + circle_list[0][2] * math.sin(random_0to1 * 2 * math.pi))
        inter_closest = (X2, Y2)
        return inter_closest
    # check if the circle_list has more than two circles
    if len(circle_list) >= 2:
        # get the last circle in the circle_list
        circle_last = circle_list[-1]
        # get the lagrest radius in the circle_list
        R_now_max = max(circle_list, key=lambda x: x[2])[2]
        # get the list cicrle that there center is in the range of the +R_now_max pix -R_now_max pix of the last circle
        Preselected_circle_list = []
        for i in range(len(circle_list[:-1])):
            circle = circle_list[i]
            if circle_last[0] -circle[0] < 2*R_now_max < circle_last[0] + circle[0] and circle_last[1] -circle[1] < 2*R_now_max < circle_last[1] + circle[1]:
                Preselected_circle_list.append(circle)
        # cuclulate all the intersection point of the last circle and the circle in the Preselected_center_list
        intersection_list = []
        for i in range(len(Preselected_circle_list)):
            try:
                inter_1st, inter_2nd = circle_intersection(circle_last, Preselected_circle_list[i])
                # intersection_list.append(inter_1st)
                # intersection_list.append(inter_2nd)
                if point_out_circles(inter_1st, circle_list[:-1]):
                    intersection_list.append(inter_1st)
                if point_out_circles(inter_2nd, circle_list[:-1]):
                    intersection_list.append(inter_2nd)
            except:
                pass
        # keep the point that is outside the circle in the circle_list
        # get the last circle center
        (Xi_1 , Yi_1) = (circle_list[-1][0], circle_list[-1][1])        

        # if inter_list is not empty. 
        if intersection_list:
            # select the intersection point that is closest to the center of the image
            inter_closest = min(intersection_list, key=lambda x: (x[0]-X1)**2 + (x[1]-Y1)**2)
        else:
            #select the intersection point that most outside the last circle if intersection_list is empty
            last_distance =  math.sqrt((Xi_1-X1)**2 + (Yi_1-Y1)**2)
            R_last = circle_list[-1][2]
            inter_closest = (Xi_1 + ((Xi_1-X1)*R_last)/last_distance , Yi_1 + ((Yi_1-Y1)*R_last)/last_distance)
        return inter_closest

def Next_inter_line(circle_list, plane_size=2000, step=20, real_scan_factor = 0.8):
    # check if the circle_list is empty
    down = plane_size/2 * (1 - real_scan_factor)
    up = plane_size/2 * (1 + real_scan_factor)
    left = plane_size/2 * (1 - real_scan_factor)
    right = plane_size/2 * (1 + real_scan_factor)

    inter_closest = (round(plane_size/2), round(plane_size/2))
    X1, Y1 = inter_closest
    if not circle_list:
        return inter_closest
    # check if the circle_list has only one circle
    if len(circle_list) >= 1:
        my_number_X = random.choice((0, 1, 2, 3))
        if my_number_X == 0:
            if circle_list[-1][0] + step >= plane_size/2 * (1 + real_scan_factor): # if the next point is out of the plane, then the next point is the go back step
                X2 = circle_list[-1][0] - step
            else:
                X2 = round(circle_list[-1][0] + step)   
            Y2 = round(circle_list[-1][1])

        elif my_number_X == 1:
            if circle_list[-1][0] - step <= plane_size/2 * (1 - real_scan_factor):
                X2 = circle_list[-1][0] + step
            else:
                X2 = round(circle_list[-1][0] - step)
            Y2 = round(circle_list[-1][1])

        elif my_number_X == 2:
            if circle_list[-1][1] + step >= plane_size/2 * (1 + real_scan_factor):
                Y2 = circle_list[-1][1] - step
            else:
                Y2 = round(circle_list[-1][1] + step)
            X2 = round(circle_list[-1][0])

        elif my_number_X == 3:
            if circle_list[-1][1] - step <= plane_size/2 * (1 - real_scan_factor):
                Y2 = circle_list[-1][1] + step
            else:
                Y2 = round(circle_list[-1][1]- step)
            X2 = round(circle_list[-1][0])            
        inter_closest = (X2, Y2)
        return inter_closest


def initial_net_point(plane_size=2000, step=30, real_scan_factor = 0.8):
    inter_closest = (round(plane_size/2), round(plane_size/2))
    X1, Y1 = inter_closest
    net_x = np.arange(plane_size/2, X1 - plane_size/2 * real_scan_factor, -step)[::-1]
    net_x = np.append(net_x, np.arange(plane_size/2, X1 + plane_size/2 * real_scan_factor, step))
    net_y = np.arange(plane_size/2, Y1 - plane_size/2 * real_scan_factor, -step)[::-1]
    net_y = np.append(net_y, np.arange(plane_size/2, Y1 + plane_size/2 * real_scan_factor, step))
    # use the net_x as the x and net_y as y to create a net of points
    net = np.array(np.meshgrid(net_x, net_y)).T.reshape(-1, 2)

    return np.unique(net, axis=0)

def Next_inter_net(circle_list, net, plane_size=2000):

    # initialize the inter_closest
    if len(circle_list) == 0:
        return (round(plane_size/2), round(plane_size/2)), net

    # if circle_list is not empty
    (X1, Y1) = (round(plane_size/2), round(plane_size/2))
    point_in_circle = []
    # traversal every point in the net_outofcircle to find how many point that is in the circle_list
    for point in net:
        for circle in circle_list:
            if point_in_circles(point, [circle]):
                new_coordinate = np.array([point[0], point[1]])
                point_in_circle.append(new_coordinate)

    point_in_circle = np.array(point_in_circle)
            
    # print(len(point_in_circle))
    # remove the point_in_circle in net
    point_in_circle = np.unique(point_in_circle, axis=0)
    # print(point_in_circle)
    net_str = np.array([f"{x},{y}" for x, y in net])
    point_in_circle_str = np.array([f"{x},{y}" for x, y in point_in_circle])
    diff_str = np.setdiff1d(net_str, point_in_circle_str)
    net = np.array([list(map(float, item.split(','))) for item in diff_str])
    print('net_long',len(net))
    # net = np.delete(net, point_in_circle, axis=0)
    
    # print(net)
    # find the 5 most closest point near the last circle
    if len(net) <= 1:
        return (round(plane_size/2), round(plane_size/2)), []
    inter_closest_list = closest_points_np(net, circle_list[-1][0:2], 5)
    # print('next point list: ', inter_closest_list)
    # find the closest point in the (X1, Y1)
    inter_closest = min(inter_closest_list, key=lambda x: (x[0]-X1)**2 + (x[1]-Y1)**2)
    # print('next point: ', inter_closest)
    return inter_closest, net

    






#def a function to increase the radius of the circle. input: 1.if the circle radius should be increased. 2.the last radius 3. the initial radius. 4. the max radius. 5. the step of the radius
def increase_radius(scan_qulity, R_last, R_init, R_max, R_step):
    if not scan_qulity:
        if R_last < R_max:
            R = R_last + R_step
        else:
            R = R_max
    else:
        R = R_init
    return R

# def a function to convert the pix point to nanomite coordinate: example: (0, 0)-> (-1e-6, 1e-6), (2000, 2000) -> (1e-6, -1e-6), (1000, 1000) -> (0.0, 0.0)
def pix_to_nanocoordinate(pix_point, plane_size=2000, int_prosses=True):
    (X, Y) = pix_point
    if int_prosses:
        X =     round(X - plane_size/2)* 1e-9  #* (2000/plane_size)
        Y = (-1)*round(Y - plane_size/2)* 1e-9 #* (2000/plane_size)              # pix_point is matrix, but the nanomite coordinate is nomal coordinate, so the Y should be -Y
    else:
        X = (X - plane_size/2)* 1e-9 #* (2000/plane_size)
        Y = (-1)*(Y - plane_size/2)* 1e-9 #* (2000/plane_size)              # pix_point is matrix, but the nanomite coordinate is nomal coordinate, so the Y should be -Y
    return (X, Y)

# convert the pix in the image to nanomite coordinate in surface
def pix_to_nanomite(pix, plane_size=2000):
    nanomite =  pix * 1e-9 * (2000/plane_size)
    return nanomite

def normalize_2darray(arr):

    # Find min and max values across the entire 2D array
    min_val = np.min(arr)  
    max_val = np.max(arr)

    # Rescale all values in the 2D array 
    normalized = 255 * (arr - min_val) / (max_val - min_val)

    return normalized

def normalize_to_image(arr):

    # Normalize the 2D array 
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized = 255 * (arr - min_val) / (max_val - min_val)

    # Convert the normalized array to uint8 type
    normalized = normalized.astype(np.uint8)

    # Convert the normalized array to an image
    img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    return normalized

def linear_whole(matrix):
    rows, cols = matrix.shape
    Y, X = np.indices((rows, cols))  # Y is row indices, X is column indices
    X = X.ravel()  # Flatten X to a 1D array
    Y = Y.ravel()  # Flatten Y to a 1D array
    data = matrix.ravel()  # Flatten the matrix data to a 1D array

    # Prepare matrix A for linear fitting
    A = np.column_stack([X, Y, np.ones(rows * cols)])
    c, _, _, _ = lstsq(A, data)  # Perform linear regression

    # Calculate the fitted plane over the entire matrix
    fitted_plane = c[0] * X + c[1] * Y + c[2]
    fitted_plane = fitted_plane.reshape(rows, cols)  # Reshape to the original matrix shape

    # Subtract the fitted plane
    processed_matrix = matrix - fitted_plane

    return processed_matrix

def linear_normalize_whole(matrix):
    rows, cols = matrix.shape
    Y, X = np.indices((rows, cols))  # Y is row indices, X is column indices
    X = X.ravel()  # Flatten X to a 1D array
    Y = Y.ravel()  # Flatten Y to a 1D array
    data = matrix.ravel()  # Flatten the matrix data to a 1D array

    # Prepare matrix A for linear fitting
    A = np.column_stack([X, Y, np.ones(rows * cols)])
    c, _, _, _ = lstsq(A, data)  # Perform linear regression

    # Calculate the fitted plane over the entire matrix
    fitted_plane = c[0] * X + c[1] * Y + c[2]
    fitted_plane = fitted_plane.reshape(rows, cols)  # Reshape to the original matrix shape

    # Subtract the fitted plane
    processed_matrix = matrix - fitted_plane

    # Normalize to 0-255
    min_val = processed_matrix.min()
    max_val = processed_matrix.max()
    if min_val == max_val:  # if the matrix is a constant matrix
        return processed_matrix
    
    normalized_matrix = 255 * (processed_matrix - min_val) / (max_val - min_val)
    normalized_matrix = np.array(normalized_matrix, dtype=np.uint8)

    return normalized_matrix


def linear_background_and_normalize(matrix):
    # matrix = np.mat(matrix)
    rows, cols = matrix.shape
    X = np.arange(cols)

    # 初始化一个与原矩阵同样大小的矩阵来存储处理后的数据
    processed_matrix = np.zeros_like(matrix, dtype=float)

    for i in range(rows):
        # 对每一行进行线性拟合
        A = np.column_stack([X, np.ones(cols)])
        b = matrix[i, :]
        c, _, _, _ = lstsq(A, b)

        # 计算拟合出的线性趋势
        line_fit = c[0] * X + c[1]

        # 扣除线性背底
        processed_matrix[i, :] = matrix[i, :] - line_fit

    # 归一化到0-255
    min_val = processed_matrix.min()
    max_val = processed_matrix.max()
    if min_val == max_val: # if the matrix is a constant matrix
        return processed_matrix
    
    normalized_matrix = 255 * (processed_matrix - min_val) / (max_val - min_val)

    normalized_matrix = np.array(normalized_matrix, dtype=np.uint8)
    
    return normalized_matrix

def get_latest_checkpoint(parent_folder, checkpoint_name="checkpoint.json"):
    """Returns the path to the most recently created subfolder in the specified parent folder."""
    # Ensure the path is absolute
    parent_folder = os.path.abspath(parent_folder)
    
    # Get all entries in the directory that are directories themselves
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    # Find the subfolder with the latest creation time
    if not subfolders:
        return None  # Return None if there are no subfolders
    
    latest_subfolder = max(subfolders, key=os.path.getctime)
    
    return os.path.join(latest_subfolder,checkpoint_name)

def get_latest_filelist(parent_folder):
    """Returns the path to the most recently created subfolder in the specified parent folder."""
    # Ensure the path is absolute
    parent_folder = os.path.abspath(parent_folder)
    
    # Get all entries in the directory that are directories themselves
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    # Find the subfolder with the latest creation time
    if not subfolders:
        return None  # Return None if there are no subfolders
    #sort the subfolders by the time
    subfolders.sort(key=lambda x: os.path.getctime(x))
    
    return subfolders

def time_trajectory_list(parent_path, file_extension='.json'):
    """ Return a list of '.json' file paths sorted by creation time. """
    npy_files = []
    # Walk through all directories and files in the parent path
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                creation_time = os.path.getctime(file_path)
                npy_files.append((file_path, creation_time))

    # Sort files based on creation time
    npy_files_sorted = sorted(npy_files, key=lambda x: x[1])
    
    # Return only the file paths, sorted by creation time
    return [file[0] for file in npy_files_sorted]

def subtract_plane(arr):
    # Fit plane to points
    A = np.c_[arr[:,0], arr[:,1], np.ones(arr.shape[0])]
    C, _, _, _ = lstsq(A, arr[:,2])   
    
    # Evaluate plane on grid
    ny, nx = arr.shape[:2]
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y)
    z = C[0]*xv + C[1]*yv + C[2]
    
    # Subtract plane from original array
    return arr - z

def tip_in_boundary(inter_closest, plane_size, real_scan_factor):
    if inter_closest[0] <= plane_size/2 * (1 + real_scan_factor) and inter_closest[0] >= plane_size/2 * (1-real_scan_factor) and inter_closest[1] <= plane_size/2 * (1 + real_scan_factor) and inter_closest[1] >= plane_size/2 * (1-real_scan_factor):
        return True
    else:
        return False

def images_equalization(image, alpha=0.5):
    # Normalize the image
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Fully equalize the image
    equalized_image = cv2.equalizeHist(norm_image)
    
    # Blend the original normalized image and the equalized image based on alpha
    adjusted_image = cv2.addWeighted(norm_image, 1-alpha, equalized_image, alpha, 0)
    
    return adjusted_image

def filter_close_bboxes(molecular_registration_list, threshold=5):
    """
    过滤掉中心点距离小于阈值的bbox。

    参数:
    molecular_registration_list (list): 包含bbox的列表，每个bbox格式为 [class, x, y, w, h, key_x, key_y, ...]。
    threshold (float): 中心点距离的阈值，默认为5。

    返回:
    list: 过滤后的bbox列表。
    """
    # if the length of the molecular_registration_list is less than 2, return the original list
    if len(molecular_registration_list) < 2:
        return molecular_registration_list
    to_remove = set()
    for i in range(len(molecular_registration_list)):
        for j in range(i + 1, len(molecular_registration_list)):
            bbox1 = molecular_registration_list[i]
            bbox2 = molecular_registration_list[j]
            center1 = np.array([bbox1[1], bbox1[2]])
            center2 = np.array([bbox2[1], bbox2[2]])
            distance = np.linalg.norm(center1 - center2)
            if distance < threshold:
                to_remove.add(i)
                to_remove.add(j)

    # 从molecular_registration_list中移除距离小于阈值的点
    filtered_list = [key_point for idx, key_point in enumerate(molecular_registration_list) if idx not in to_remove]
    return filtered_list

def rotate_point(center, angle, point):
    """
    旋转坐标点
    :param center: 旋转中心坐标 (x, y)
    :param angle: 旋转角度（顺时针，单位：度）
    :param point: 被旋转坐标 (x, y)
    :return: 旋转后的坐标 (x', y')
    """
    angle_rad = math.radians(-angle)  # 将角度转换为弧度
    cx, cy = center
    px, py = point

    # 计算旋转后的坐标
    x_new = cx + math.cos(angle_rad) * (px - cx) - math.sin(angle_rad) * (py - cy)
    y_new = cy + math.sin(angle_rad) * (px - cx) + math.cos(angle_rad) * (py - cy)

    return x_new, y_new

# # 示例用法
# molecular_registration_list = [
#     [0, 10, 10, 5, 5, 12, 12],
#     [0, 12, 12, 5, 5, 14, 14],
#     [0, 50, 50, 5, 5, 52, 52]
# ]

# filtered_list = filter_close_bboxes(molecular_registration_list, threshold=5)
# print(filtered_list)
def cal_Br_pos(molecule_position, mol_scale, angle):
    """
    计算并旋转 Br 原子的坐标
    :param molecule_position: 分子中心坐标 (x, y)
    :param mol_scale: 分子尺度
    :param angle: 旋转角度（顺时针，单位：度）
    :return: 旋转后的 Br 原子坐标列表
    """
    Br_positions = [
        (molecule_position[0] - mol_scale * 0.5, molecule_position[1] - mol_scale * 0.5),
        (molecule_position[0] + mol_scale * 0.5, molecule_position[1] - mol_scale * 0.5),
        (molecule_position[0] + mol_scale * 0.5, molecule_position[1] + mol_scale * 0.5),
        (molecule_position[0] - mol_scale * 0.5, molecule_position[1] + mol_scale * 0.5)
    ]

    rotated_positions = [rotate_point(molecule_position, angle, pos) for pos in Br_positions]

    # 按照指定规则排序
    sorted_positions = [None] * 4
    sorted_positions[3] = max(rotated_positions, key=lambda pos: pos[1])  # 最上面的点
    sorted_positions[2] = max(rotated_positions, key=lambda pos: pos[0])  # 最右边的点
    sorted_positions[1] = min(rotated_positions, key=lambda pos: pos[1])  # 最下面的点
    sorted_positions[0] = min(rotated_positions, key=lambda pos: pos[0])  # 最左边的点

    return sorted_positions

def scale_coordinates(coords, scale_factor):
    """
    计算四个二维坐标的中心坐标，并返回以中心坐标为中心缩放后的坐标。

    参数:
    coords (list of tuple): 包含四个二维坐标的列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    scale_factor (float): 缩放参数

    返回:
    tuple: 中心坐标 (cx, cy)
    list of tuple: 缩放后的坐标 [(sx1, sy1), (sx2, sy2), (sx3, sy3), (sx4, sy4)]
    """
    # 计算中心坐标
    coords_array = np.array(coords)
    center = np.mean(coords_array, axis=0)
    
    # 计算缩放后的坐标
    scaled_coords = (coords_array - center) * scale_factor + center
    
    return tuple(center), [tuple(coord) for coord in scaled_coords]

def generate_grid(center, X, Y, L, W):
    """
    生成以指定坐标为中心的格点。
    
    参数:
    center (tuple): 中心坐标 (cx, cy)
    X (int): 行数
    Y (int): 列数
    L (float): 总长度
    W (float): 总宽度
    
    返回:
    np.ndarray: 格点坐标数组，形状为 (X, Y, 2)
    """
    cx, cy = center
    x_spacing = L / (X - 1)
    y_spacing = W / (Y - 1)
    
    grid_points = np.zeros((X, Y, 2))
    
    for i in range(X):
        for j in range(Y):
            grid_points[i, j, 0] = cx - L / 2 + i * x_spacing
            grid_points[i, j, 1] = cy - W / 2 + j * y_spacing
    
    return grid_points

def find_nearest_grid_point(grid_points, coord):
    """
    找到距离指定坐标最近的格点。
    
    参数:
    grid_points (np.ndarray): 格点坐标数组，形状为 (X, Y, 2)
    coord (tuple): 指定坐标 (x, y)
    
    返回:
    tuple: 最近格点的坐标 (x, y)
    """
    distances = np.sqrt((grid_points[..., 0] - coord[0])**2 + (grid_points[..., 1] - coord[1])**2)
    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    nearest_point = grid_points[min_index]
    
    return tuple(nearest_point)

def mouse_click_tip(image):
    """
    显示窗口并等待鼠标事件：
    - 左键：在点击位置画红点，展示1秒后关闭并返回 ( (x百分比, y百分比), image, voltage, current )
    - 右键：关闭窗口并返回 ( None, image, voltage, current )
    - 中键：要求输入电压、电流，多次点击每次可重新输入。若从未点过中键，voltage 与 current 为 None
    函数不会结束，直到左键或者右键点击才返回。
    """
    
    mouse_info = {
        "left_click": False,
        "right_click": False,
        "middle_click": False,
        "x": 0,
        "y": 0
    }
    bias = None
    current = None

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["left_click"] = True
            param["x"] = x
            param["y"] = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            param["right_click"] = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            param["middle_click"] = True

    cv2.namedWindow("MyWindow", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("MyWindow", image)
    cv2.setMouseCallback("MyWindow", mouse_callback, mouse_info)

    while True:
        cv2.waitKey(50)
        # 右键 -> 结束并返回
        if mouse_info["right_click"]:
            cv2.destroyWindow("MyWindow")
            return None, image, (bias, current)
        # 左键 -> 在点击处画点后返回
        if mouse_info["left_click"]:
            px, py = mouse_info["x"], mouse_info["y"]
            height, width = image.shape[:2]
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
            cv2.imshow("MyWindow", image)
            cv2.waitKey(1)
            time.sleep(1)
            cv2.destroyWindow("MyWindow")
            return (px / width, py / height), image, (bias, current)
        # 中键 -> 输入电压电流，覆盖上次输入值
        if mouse_info["middle_click"]:
            bias = input("Please input the bias (V): ")
            current = input("Please input the current(nA): ")
            bias = str(bias)
            current = str(current) + 'n'
            mouse_info["middle_click"] = False
        # 按下空格 输入分子状态

def mouse_click_tip(image, mode = 'default', state_num = 4): # mode = 'default' or 'sigle' or 'batch'
    """
    显示窗口并等待鼠标事件：
    - 左键：在点击位置画红点，展示1秒后关闭并返回 ( (x百分比, y百分比), image, voltage, current )
    - 右键：关闭窗口并返回 ( None, image, voltage, current )
    - 中键：要求输入电压、电流，多次点击每次可重新输入。若从未点过中键，voltage 与 current 为 None
    - 空格：输入一个四位数字(仅含'0'或'1')，若输入错误则重新输入，直到正确后转为 np.array([x,x,x,x])     1  non   2 suc   3 bad
    函数不会结束，直到左键或者右键点击才返回。
    """

    mouse_info = {
        "left_click": False,
        "right_click": False,
        "middle_click": False,
        "x": 0,
        "y": 0
    }

    bias = None
    current = None
    br_sites_array = None  # 存储用户空格输入的 4 位二值数组

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param["left_click"] = True
            param["x"] = x
            param["y"] = y
        elif event == cv2.EVENT_RBUTTONDOWN:
            param["right_click"] = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            param["middle_click"] = True

    cv2.namedWindow("MyWindow", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("MyWindow", image)
    cv2.setMouseCallback("MyWindow", mouse_callback, mouse_info)

    while True:
        key = cv2.waitKey(50)

        # 右键 -> 结束并返回
        if mouse_info["right_click"]:
            cv2.destroyWindow("MyWindow")
            return None, image, (bias, current), br_sites_array

        # 左键 -> 在点击处画点后返回
        if mouse_info["left_click"]:
            px, py = mouse_info["x"], mouse_info["y"]
            height, width = image.shape[:2]
            cv2.circle(image, (px, py), 3, (0, 0, 255), -1)
            cv2.imshow("MyWindow", image)
            cv2.waitKey(1)
            time.sleep(1)
            cv2.destroyWindow("MyWindow")
            return (px / width, py / height), image, (bias, current), br_sites_array

        # 中键 -> 输入电压电流，覆盖上次输入值，范围检查
        if mouse_info["middle_click"]:
            # 输入并检查 bias
            while True:
                try:
                    temp_bias = float(input("Please input the bias (V) [0.5~5]: "))
                    if 0.5 <= temp_bias <= 5:
                        bias = str(temp_bias)
                        break
                    else:
                        print("Invalid input. Please enter a value between 0.5 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value between 0.5 and 5.")

            # 输入并检查 current
            while True:
                try:
                    temp_current = float(input("Please input the current (nA) [0.01~5]: "))
                    if 0.01 <= temp_current <= 5:
                        current = str(temp_current) + 'n'
                        break
                    else:
                        print("Invalid input. Please enter a value between 0.01 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value between 0.01 and 5.")

            mouse_info["middle_click"] = False

        # 按下空格 -> 输入四位数字，仅含 '0' 或 '1'
        if key == 32:  # 空格的 ASCII 码
            while True:
                br_sites_input = input("Please enter digits: ")
                if mode == 'sigle':
                    if len(br_sites_input) == 1:
                        br_sites_input = br_sites_input * 4 #repit
                    if len(br_sites_input) == state_num and all(ch in {'1', '2', '3'} for ch in br_sites_input):
                        br_sites_array = np.array([int(ch) for ch in br_sites_input])
                        br_sites_array = np.array([br_sites_array[0]])
                        print(f"Br site states: {br_sites_array}")
                        break
                    else:
                        print("Invalid input. Must be a 4-digit string of 1 or 2 or 3.")    
                
                elif mode == 'default':
                    if len(br_sites_input) == state_num and all(ch in {'0', '1', '2',} for ch in br_sites_input):
                        br_sites_array = np.array([int(ch) for ch in br_sites_input])
                        print(f"Br site states: {br_sites_array}")
                        break
                    else:
                        print("Invalid input. Must be a 4-digit string of 0 or 1 or 2.")  

                elif mode == 'batch':
                    # while True:
                    # br_sites_input = input("Please enter 4 digits (1, 2, or 3): ")
                    if len(br_sites_input) == state_num and all(ch in {'1', '2', '3'} for ch in br_sites_input):
                        # 将输入转化为对应数量的 np.array
                        br_sites_array = [np.array([int(ch)]) for ch in br_sites_input]
                        print(f"Br site states: {br_sites_array}")
                        break
                    else:
                        print(f"Invalid input. Must be a {state_num}-digit string of 1, 2, or 3.")
                
        
def matrix_to_cartesian(px, py, center=(50, 40), side_length=7):
    """
    px, py: 图像中的相对位置(0~1)，(0,0)对应图像左上角，(1,1)对应图像右下角
    center: 笛卡尔坐标中图像中心点
    side_length: 图像边长
    返回：转换后的笛卡尔 x, y 坐标
    """
    cx, cy = center
    # 确保图像中心为(cx, cy)，并将矩阵坐标系转换为笛卡尔坐标系，y 轴向上
    x = cx + (px - 0.5) * side_length
    y = cy - (py - 0.5) * side_length  # 矩阵坐标和笛卡尔坐标的 y 方向相反
    return x, y

def point_in_rotated_rect_polar(
    result,
    image_shape,
    rect_center,
    rect_angle_deg,
    rect_side_length
):
    """
    将鼠标点击得到的矩阵相对坐标转换到旋转矩形坐标系下的极坐标。
    参数说明：
    1. result: (x_frac, y_frac)，矩阵相对坐标(0~1)，左上角(0,0)，右下角(1,1)。
    2. image_shape: 图像的(shape[0], shape[1])，通常是(height, width)。
    3. rect_center: (cx, cy)，旋转矩形在图像像素坐标系下的中心点，单位像素。
    4. rect_angle_deg: 旋转矩形的角度(度数)，0表示不旋转，正值代表逆时针旋转。
    5. rect_side_length: 旋转矩形边长的像素长度。

    返回： (r_frac, theta_deg)
    其中：
    - r_frac：半径相对于 rect_side_length 的归一化值 (0~1)。
    - theta_deg：极坐标角度 (0~360)，相对旋转矩形 x 轴方向(矩形未旋转时的边平行于图像 x 轴)。
    """

    # 1. 将相对坐标result转换为图像像素坐标(px, py)
    height, width = image_shape[:2]
    px = result[0] * width
    py = result[1] * height

    # 2. 将矩形中心作为原点进行平移
    dx = px - rect_center[0]
    dy = py - rect_center[1]

    # 3. “反转”旋转，把点旋转到矩形未旋转状态
    #    rect_angle_deg 为逆时针角度，所以此处用 -rect_angle_deg
    angle_rad = np.deg2rad(-rect_angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    # 4. 转换为极坐标
    r = np.sqrt(dx_rot**2 + dy_rot**2)
    # 以矩形未旋转时水平向右为 0°，逆时针为正
    theta_deg = np.degrees(np.arctan2(dy_rot, dx_rot))
    # 范围调整到 [0, 360)
    theta_deg = (theta_deg + 360) % 360

    # 5. 半径用 rect_side_length 归一化
    r_frac = r / rect_side_length

    dy_rot = -dy_rot  # 矩形坐标系 y 轴向上，而图像坐标系 y 轴向下

    return r_frac, theta_deg, dx_rot, dy_rot

def reverse_point_in_rotated_rect_polar(
        dx_rot, dy_rot, 
        image_shape, 
        rect_center, 
        rect_angle_deg, 
        rect_side_length, 
        zoom_in_100, 
        scan_square_Buffer_pix):
    """
    从旋转矩形坐标系下的偏移量计算出矩阵相对坐标（逆向变换）
    
    参数说明：
    1. dx_rot, dy_rot: 旋转矩形坐标系中的偏移量（已缩放）
    2. image_shape: 图像的(shape[0], shape[1])，通常是(height, width)
    3. rect_center: (cx, cy)，旋转矩形在图像像素坐标系下的中心点，单位像素
    4. rect_angle_deg: 旋转矩形的角度(度数)，0表示不旋转，正值代表逆时针旋转
    5. rect_side_length: 旋转矩形边长的像素长度
    6. zoom_in_100: 缩放因子
    7. scan_square_Buffer_pix: 扫描边界大小
    
    返回：result (x_frac, y_frac)，矩阵相对坐标(0~1)
    """
    height, width = image_shape[:2]
    
    # 1. 恢复 dx_rot 和 dy_rot 的原始尺度（去除缩放）
    dx_rot_original = dx_rot / (zoom_in_100 / scan_square_Buffer_pix)
    dy_rot_original = dy_rot / (zoom_in_100 / scan_square_Buffer_pix)
    
    # 2. 注意：在 point_in_rotated_rect_polar 中，返回前执行了 dy_rot = -dy_rot，所以这里需要反转
    dy_rot_original = -dy_rot_original
    
    # 3. 执行旋转矩阵的逆变换，将旋转后的坐标变回原始坐标系下的偏移量
    angle_rad = np.deg2rad(-rect_angle_deg)  # 与原函数保持一致，使用 -rect_angle_deg
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # 旋转矩阵的逆变换
    # 由于旋转矩阵是正交矩阵，其逆等于其转置
    dx = dx_rot_original * cos_a + dy_rot_original * sin_a
    dy = -dx_rot_original * sin_a + dy_rot_original * cos_a
    
    # 4. 将偏移量加上矩形中心，得到图像像素坐标
    px = rect_center[0] + dx
    py = rect_center[1] + dy
    
    # 5. 将像素坐标转换为相对坐标 (0~1)
    x_frac = px / width
    y_frac = py / height
    
    return (x_frac, y_frac)

def save_action_to_npy(action, file_path="actions.npy"):
    """
    将新的 action (numpy array) 追加保存到同一个 .npy 文件中。
    如果文件不存在，则直接创建并保存；若存在，则先加载再追加。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 首先看目标文件是否存在
    if os.path.exists(file_path):
        # 加载已经存在的动作数据
        existing_data = np.load(file_path)

        # 如果现有数据是一维，则说明可能是单个 action，需在第一维上扩展
        # 或者如果现有数据是多维，就可直接拼接
        if existing_data.ndim == 1:
            existing_data = existing_data.reshape(1, -1)

        # 同样保证新 action 也是 shape=(1, n)
        action = action.reshape(1, -1)

        # 拼接数据
        new_data = np.vstack([existing_data, action])
        np.save(file_path, new_data)
    else:
        # 文件不存在时，直接保存
        np.save(file_path, action)

def compute_triangular_orientation(points, visualize=False,neighbor_distance=1.5):
    """
    鲁棒版：支持格点随机缺失的取向计算（0-120度）。
    改进点：
        1. 动态邻居数量（基于局部密度）
        2. 向量长度严格过滤
        3. 双聚类验证（角度+长度）
    """
    points = np.array(points)
    n_points = len(points)
    angles = []
    lengths = []

    # 计算每个点的动态最近邻（3-6个，且距离不超过1.5倍理论间距）
    for i in range(n_points):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf  # 排除自身
        sorted_indices = np.argsort(dists)
        
        # 动态选择有效邻居（距离≤1.5且最多6个）
        valid_neighbors = []
        for j in sorted_indices:
            if dists[j] <= neighbor_distance and len(valid_neighbors) < 6:
                valid_neighbors.append(j)
            else:
                break
        
        # 记录向量角度和长度
        for j in valid_neighbors:
            dx, dy = points[j] - points[i]
            angle = np.degrees(np.arctan2(dy, dx)) % 120
            length = np.sqrt(dx**2 + dy**2)
            angles.append(angle)
            lengths.append(length)

    # 第一次聚类：基于角度（剔除明显离群点）
    if len(angles) == 0:
        return 0.0
    angles_array = np.array(angles).reshape(-1, 1)
    ms_angle = MeanShift(bandwidth=1, bin_seeding=True)
    ms_angle.fit(angles_array)
    angle_labels = ms_angle.labels_
    angle_centers = ms_angle.cluster_centers_.ravel()
    
    # 第二次聚类：基于向量长度（进一步过滤）
    lengths_array = np.array(lengths).reshape(-1, 1)
    ms_length = MeanShift(bandwidth=0.01, bin_seeding=True)
    ms_length.fit(lengths_array)
    length_labels = ms_length.labels_
    length_centers = ms_length.cluster_centers_.ravel()
    
    # 选择同时属于角度和长度主簇的向量
    main_angle_cluster = np.argmax(np.bincount(angle_labels))
    main_length_cluster = np.argmax(np.bincount(length_labels))
    valid_mask = (angle_labels == main_angle_cluster) & (length_labels == main_length_cluster)
    length_center = np.mean(lengths_array[length_labels == main_length_cluster])
    filtered_angles = angles_array[valid_mask]
    
    # 计算最终主方向
    dominant_angle = np.mean(filtered_angles) % 120 if len(filtered_angles) > 0 else 0.0
    if dominant_angle > 60:
        dominant_angle = dominant_angle - 60

    # 找到最靠近中心的点
    center = np.mean(points, axis=0)  # 计算中心点
    distances_to_center = np.linalg.norm(points - center, axis=1)
    closest_point_index = np.argmin(distances_to_center)
    closest_point = points[closest_point_index]

    if visualize:
        plt.figure(figsize=(15, 5))
        
        # 1. 格点及有效邻居向量
        plt.subplot(1, 3, 1)
        plt.scatter(points[:, 0], points[:, 1], c='b', s=30, alpha=0.6, label="Points")
        for i in range(min(100, n_points)):  # 仅显示部分点的向量
            dists = np.linalg.norm(points - points[i], axis=1)
            dists[i] = np.inf
            valid_neighbors = [j for j in np.argsort(dists)[:6] if dists[j] <= 1.5]
            for j in valid_neighbors:
                plt.arrow(points[i, 0], points[i, 1], points[j, 0]-points[i, 0], points[j, 1]-points[i, 1],
                          head_width=0.05, head_length=0.1, fc='r', ec='r', alpha=0.3)
        plt.title("Points with Valid Neighbors")
        plt.gca().set_aspect('equal')
        plt.legend()

        # 2. 角度分布与聚类
        plt.subplot(1, 3, 2)
        plt.hist(angles, bins=36, range=(0, 120), alpha=0.5, label="All Angles")
        plt.hist(filtered_angles, bins=36, range=(0, 120), alpha=0.8, label="Filtered Angles")
        plt.axvline(dominant_angle, color='g', linewidth=2, label=f"Dominant Angle: {dominant_angle:.1f}°")
        plt.xlabel("Angle (mod 80°)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Angle Clustering")

        # 3. 向量长度分布
        plt.subplot(1, 3, 3)
        plt.hist(lengths, bins=20, range=(0, 2), alpha=0.5, label="All Lengths")
        
        plt.axvline(length_center, color='r', 
                   linestyle='--', label="Main Length Cluster")
        plt.xlabel("Vector Length")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Length Filtering")

        plt.tight_layout()
        plt.show()

    return dominant_angle, closest_point, length_center

def find_triangle_group(points, n,rotation_angle=30, dist = 0.2):
    plt.figure(figsize=(8, 8))
    
    points = np.array(points)
    if len(points) == 0:
        return -1, None
    plt.scatter(points[:, 0], points[:, 1], c='b', s=30, alpha=0.6, label="Points")
    # 1. 旋转坐标系以对齐晶格方向
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    rotated_points = np.dot(points, rotation_matrix.T)

    plt.scatter(rotated_points[:, 0], rotated_points[:, 1], c='r', s=30, alpha=0.6, label="Rotated Points")
    plt.legend()
    # plt.show()
    # 2. 估计晶格间距d，使用旋转后的点间最小距离
    # dists = cdist(rotated_points, rotated_points)
    # np.fill_diagonal(dists, np.inf)
    d = dist
    
    # 3. 映射每个实际点到最近的完美晶格坐标(i,j)
    # 计算基向量在旋转后的坐标系中的表示
    v1 = np.array([d, 0.0])
    v2 = np.array([d/2, d * np.sqrt(3)/2])
    
    lattice = {}
    for idx, p in enumerate(rotated_points):
        # 解方程 p = i*v1 + j*v2
        A = np.array([v1, v2]).T
        try:
            ij = np.linalg.solve(A, p)
        except np.linalg.LinAlgError:
            continue  # 基向量共线，理论上不应发生
        i, j = np.round(ij).astype(int)
        lattice[(i, j)] = (points[idx], idx)  # 存储原始点

    # draw lattice points
    for (i, j), p in lattice.items():
        plt.scatter(p[0][0], p[0][1], c='g', s=30, alpha=0.6, label=f"Lattice ({i},{j})")
        plt.text(p[0][0], p[0][1], f"({i},{j})", fontsize=8, ha='right', va='bottom', color='g')

    plt.show()
    
    # 4. 寻找所有可能的三角形结构
    valid_groups = []
    for (i0, j0) in lattice.keys():
        valid = True
        group = []
        for di in range(n):
            for dj in range(n - di):
                current = (i0 + di, j0 + dj)
                if current not in lattice:
                    valid = False
                    break
                group.append(current)
            if not valid:
                break
        if valid and len(group) == n*(n+1)//2:
            valid_groups.append(group)
    
    if not valid_groups:
        return -1, None
    
    # 5. 计算整体中心并选择最靠近的组
    center = np.mean(points, axis=0)
    min_dist = float('inf')
    best_group = None
    for group in valid_groups:
        group_points = np.array([lattice[ij][0] for ij in group])
        group_center = np.mean(group_points, axis=0)
        dist = np.linalg.norm(group_center - center)
        if dist < min_dist:
            min_dist = dist
            best_group = group_points
            best_indices = [lattice[ij][1] for ij in group]
    
    return best_group, best_indices

def filter_lattice_points(points):
    # 找到最靠近中心的格点
    center_point = min(points, key=lambda p: abs(p[0]) + abs(p[1]))

    # 提取原点坐标
    i, j = center_point

    # 找出x坐标的最大值和最小值
    max_x = max(p[0] for p in points)
    min_x = min(p[0] for p in points)
    max_y = max(p[1] for p in points)
    min_y = min(p[1] for p in points)

    # 生成一级过滤点
    primary_filter = set()
    x = i
    while x <= max_x+2*(max_x-min_x):
        primary_filter.add((x, j))
        x += 3
    x = i
    while x >= min_x-2*(max_x-min_x):
        primary_filter.add((x, j))
        x -= 3

    # 生成二级过滤点
    secondary_filter = set()
    for px, py in primary_filter:
        x = px
        y = py
        step = 1
        while x + step <= max_x and y + step <= max_y:
            secondary_filter.add((x + step, y + step))
            secondary_filter.add((x - step, y - step))
            step += 1
        step = 1
        while x - step >= min_x and y - step >= min_y:
            secondary_filter.add((x + step, y + step))
            secondary_filter.add((x - step, y - step))
            step += 1

    # 合并一级和二级过滤点
    filter_points = primary_filter.union(secondary_filter)

    # 过滤给定格点
    filtered_points = [p for p in points if p not in filter_points]

    return filtered_points

def find_patten_group(points, rotation_angle=30, dist=0.2, pattern='kagome',visualize=False):
    if visualize:
        plt.figure(figsize=(8, 8))
    
    points = np.array(points)
    if len(points) == 0:
        return -1, None
    
    if visualize:
        plt.scatter(points[:, 0], points[:, 1], c='b', s=30, alpha=0.6, label="Points")
    
    # 1. 旋转坐标系以对齐晶格方向
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    rotated_points = np.dot(points, rotation_matrix.T)

    if visualize:
        plt.scatter(rotated_points[:, 0], rotated_points[:, 1], c='r', s=30, alpha=0.6, label="Rotated Points")
        plt.legend()
    
    # 2. 估计晶格间距d，使用旋转后的点间最小距离
    d = dist
    
    # 3. 映射每个实际点到最近的完美晶格坐标(i,j)
    v1 = np.array([d, 0.0])
    v2 = np.array([d/2, d * np.sqrt(3)/2])
    
    lattice = {}
    for idx, p in enumerate(rotated_points):
        A = np.array([v1, v2]).T
        try:
            ij = np.linalg.solve(A, p)
        except np.linalg.LinAlgError:
            continue
        i, j = np.round(ij).astype(int)
        lattice[(i, j)] = (points[idx], idx)

    # 绘制映射到的晶格点
    if visualize:
        for (i, j), p in lattice.items():
            plt.scatter(p[0][0], p[0][1], c='g', s=30, alpha=0.6, label=f"Lattice ({i},{j})")
            plt.text(p[0][0], p[0][1], f"({i},{j})", fontsize=8, ha='right', va='bottom', color='g')

        plt.show()
    if pattern == 'kagome_lattice':
        all_tri_points = []
        for (i0, j0) in lattice.keys():
            all_tri_points.append((i0, j0))
        group = [point for point in all_tri_points if point[0]% 2 == 0 or point[1] % 2 == 0]
        group_points = np.array([lattice[ij][0] for ij in group])
        best_indices = [lattice[ij][1] for ij in group]
        return group_points, best_indices
    
    elif pattern == 'honeycomb_lattice':
        all_tri_points = []
        for (i0, j0) in lattice.keys():
            all_tri_points.append((i0, j0))
        group = filter_lattice_points(all_tri_points)
        group_points = np.array([lattice[ij][0] for ij in group])
        best_indices = [lattice[ij][1] for ij in group]
        return group_points, best_indices

    
    elif pattern == 'kagome':
        
        # 4. 寻找 Kagome 格点组
        pattern_points = [
            [(0, 3)],
            [(1, 1), (1, 2), (1, 3), (1, 4)],
            [(2, 1), (2, 3)],
            [(3, 0), (3, 1), (3, 2), (3, 3)],
            [(4, 1)]
        ]
    
    elif pattern == 'honeycomb':
        pattern_points = [
            [(0, 1), (0, 2)],
            [(1, 0), (1, 2)],
            [(2, 0), (2, 1)]
        ]

    valid_groups = []
    
    for (i0, j0) in lattice.keys():
        valid = True
        group = []
        for row in pattern_points:
            for di, dj in row:
                current = (i0 + di, j0 + dj)
                if current not in lattice:
                    valid = False
                    break
                group.append(current)
            if not valid:
                break
        if valid:
            valid_groups.append(group)
    
    if not valid_groups:
        return -1, None
    
    # 5. 选择最靠近中心的 Kagome 组
    center = np.mean(points, axis=0)
    min_dist = float('inf')
    best_group = None
    
    for group in valid_groups:
        group_points = np.array([lattice[ij][0] for ij in group])
        group_center = np.mean(group_points, axis=0)
        dist = np.linalg.norm(group_center - center)
        if dist < min_dist:
            min_dist = dist
            best_group = group_points
            best_indices = [lattice[ij][1] for ij in group]
    
    return best_group, best_indices

def interactive_keypoint_selector(image , key_results, image_size=304, angle=35.154730569873465, side_length=20):
    """交互式关键点选择函数
    参数：
        key_results: 关键点识别结果列表
        image_size: 图像尺寸（默认304）
        angle: 旋转角度（默认35.15°）
    返回：
        选中的key_result列表（包含原始和伪造的）
    """
    # 初始化状态变量
    boxes = []
    history_stack = []
    dragging = False
    current_operation = []
    selected_results = None
    
    # 计算旋转矩形
    def calculate_rotated_square(center_x, center_y, angle_deg, side_length=side_length):
        angle_rad = math.radians(angle_deg)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        half = side_length // 2
        
        points = [(-half, -half), (-half, half),
                 (half, half), (half, -half)]
        
        rotated_points = []
        for (x, y) in points:
            rx = x * cos_theta - y * sin_theta
            ry = x * sin_theta + y * cos_theta
            rotated_points.append((int(center_x + rx), int(center_y + ry)))
        return rotated_points

    # 初始化boxes
    for res in key_results:
        kpt_x = res[5] * image_size
        kpt_y = res[6] * image_size
        poly = calculate_rotated_square(kpt_x, kpt_y, angle)
        boxes.append({
            'poly': poly,
            'key_result': res,
            'selected': False,
            'is_fake': False
        })

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, current_operation, boxes, history_stack
        
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            current_operation = []
            in_box = False
            
            # 检测点击现有框
            for i, box in enumerate(boxes):
                contour = np.array(box['poly'], dtype=np.int32).reshape((-1, 1, 2))
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    prev_state = box['selected']
                    box['selected'] = not prev_state
                    current_operation.append((i, prev_state))
                    in_box = True
            
            # 创建伪造框
            if not in_box:
                x_percent = x / image_size
                y_percent = y / image_size
                fake_key = (0.0, 0.5, 0.5, 0.1, 0.1, x_percent, y_percent)
                poly = calculate_rotated_square(x, y, angle)
                boxes.append({
                    'poly': poly,
                    'key_result': fake_key,
                    'selected': True,
                    'is_fake': True
                })
                current_operation.append((len(boxes)-1, False))
            
            if current_operation:
                history_stack.append(current_operation.copy())
                current_operation.clear()
                
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            # 拖拽选择
            for i, box in enumerate(boxes):
                contour = np.array(box['poly'], dtype=np.int32).reshape((-1, 1, 2))
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    if not box['selected']:
                        prev_state = box['selected']
                        box['selected'] = True
                        current_operation.append((i, prev_state))
                        
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            if current_operation:
                history_stack.append(current_operation.copy())
                current_operation.clear()
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 撤销操作
            if history_stack:
                last_op = history_stack.pop()
                for idx, prev_state in last_op:
                    boxes[idx]['selected'] = prev_state

    # 创建窗口
    cv2.namedWindow('Keypoint Selector')
    cv2.setMouseCallback('Keypoint Selector', mouse_callback)
    
    # 创建空白图像（假设单通道输入）
    # image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    while True:
        display_img = image.copy()
        
        # 绘制所有框
        for box in boxes:
            color = (0, 255, 0) if not box['selected'] else (0, 0, 255)
            thickness = 1 if not box['selected'] else 2
            pts = np.array(box['poly'], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, color, thickness)
        
        cv2.imshow('Keypoint Selector', display_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # 空格键确认选择
            selected_results = [box['key_result'] for box in boxes if box['selected']]
            cv2.destroyWindow("Keypoint Selector")
            return selected_results
        elif key == ord('r'):  # 重置选择
            for box in boxes:
                box['selected'] = False
            history_stack.clear()
        elif key == 27:  # ESC退出
            selected_results = []
            cv2.destroyWindow("Keypoint Selector")
            return selected_results
    
    # cv2.destroyAllWindows()
    # return selected_results

if __name__ == "__main__":
    # # Example usage
    # drift_image_folder = "./drift_image/"
    # # Load the two remote sensing images (replace with actual file paths or image arrays)
    # image_t = cv2.imread(drift_image_folder + "image_t.png", cv2.IMREAD_GRAYSCALE)
    # image_t1 = cv2.imread(drift_image_folder + "image_t1.png", cv2.IMREAD_GRAYSCALE)
    # # Pre-process the image
    # image_t_equ = cv2.equalizeHist(image_t)
    # image_t1_equ = cv2.equalizeHist(image_t1)
    # image_t = cv2.addWeighted(image_t_equ, 0.4, image_t, 0.6, 0)
    # image_t1 = cv2.addWeighted(image_t1_equ, 0.4, image_t1, 0.6, 0)

    # # Call the function to calculate the motion vector
    # motion_vector, kp1, kp2, good_matches,good_matches_point, cluster_matches, cluster_points= calculate_motion_vector(image_t, image_t1)
    # print(f"Motion vector: {motion_vector}")
    # # draw the motion vector, keypoints, and matches on both images
    # img_matches = cv2.drawMatches(image_t, kp1, image_t1, kp2, cluster_matches, None)
    # cv2.imshow("Motion Vector", img_matches)
    # cv2.waitKey(0)
    # # culculate the bias of all good_matches and plot the bias point. 
    # # for example, there are 2 good_matches (0.1, 0.2) → (0.15, 0.26), (0.2, 0.3) → (0.24, 0.37),so the bias is (0.05, 0.06) and (0.04, 0.07)
    # # plot all the bias points on a 2D plane
    # good_matches_point = np.array(good_matches_point)
    # cluster_points = np.array(cluster_points)
    # # print("bias:", bias)
    # # plot the bias points in blue and motion_vector in red and cluster_points in yellow
    # plt.scatter(good_matches_point[:, 0], good_matches_point[:, 1], color='b', label='bias')
    # plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color='y', label='cluster_points')
    # plt.scatter(motion_vector[0], motion_vector[1], color='r', label='motion_vector')    
    # plt.legend()
    # plt.show()



    # # 测试用例：生成旋转45°的三角格点（50个点，带扰动）
    # np.random.seed(42)
    # L = 10
    # rotation_angle = 15
    # points = []
    # for i in range(L):
    #     for j in range(L):
    #         x = (i + 0.5 * j) * np.cos(np.radians(rotation_angle)) - (j * np.sqrt(3)/2) * np.sin(np.radians(rotation_angle))
    #         y = (i + 0.5 * j) * np.sin(np.radians(rotation_angle)) + (j * np.sqrt(3)/2) * np.cos(np.radians(rotation_angle))
    #         x += np.random.uniform(-0.1, 0.1)
    #         y += np.random.uniform(-0.1, 0.1)
    #         points.append((x, y))

    # # 随机删除5个点
    # points = np.array(points)
    # points = np.delete(points, np.random.choice(len(points), 5, replace=False), axis=0)

    # angle = compute_triangular_orientation(points, visualize=True)
    # print(f"Calculated Orientation: {angle:.2f}° (Ground Truth: {rotation_angle}°)")






    # # 测试示例
    # np.random.seed(42)
    # L = 7
    # rotation_angle = 30  # 格点旋转角度
    # n = 3  # 寻找边长为3的三角形

    # # 生成旋转后的完整格点
    # points = []
    # for i in range(L):
    #     for j in range(L):
    #         x = (i + 0.5*j) * np.cos(np.radians(rotation_angle)) - (j * np.sqrt(3)/2) * np.sin(np.radians(rotation_angle))
    #         y = (i + 0.5*j) * np.sin(np.radians(rotation_angle)) + (j * np.sqrt(3)/2) * np.cos(np.radians(rotation_angle))
    #         x += np.random.uniform(-0.1, 0.1)
    #         y += np.random.uniform(-0.1, 0.1)
    #         points.append((x, y))

    # # 随机删除5个点并打乱顺序
    # points = np.array(points)
    # points = np.delete(points, np.random.choice(len(points), 5, replace=False), axis=0)

    # # 查找三角形
    # result = compute_triangular_orientation(points)
    # print(f"Calculated Orientation: {result:.2f}° (Ground Truth: {rotation_angle}°)")



    # # 生成 L=7、rotation_angle=30 的理想格点并加随机扰动
    # # 同时随机删除一些点，模拟缺失
    
    # L = 10
    # rotation_angle = 0
    # random.seed(42)  
    # np.random.seed(42)

    key_points_result = [(0.0, 0.903857409954071, 0.19411621987819672, 0.18417970836162567, 0.17373046278953552, 0.9007812142372131, 0.19033202528953552, 0.903124988079071, 0.18857420980930328, 
0.8687500357627869, 0.23710936307907104, 0.9593749642372131, 0.19746093451976776), (0.0, 0.4025940001010895, 0.4186156988143921, 0.15410156548023224, 0.14707030355930328, 0.39765623211860657, 0.4175781309604645, 0.4000000059604645, 0.4164062440395355, 0.3705078065395355, 0.458984375, 0.44921875, 0.42499998211860657), (0.0, 0.878326416015625, 0.3707519471645355, 0.1719726324081421, 0.16367186605930328, 0.871874988079071, 0.36328125, 0.875, 0.36152341961860657, 0.840624988079071, 0.40703123807907104, 0.9312500357627869, 0.3736328184604645), (0.0, 0.27906492352485657, 0.13454589247703552, 0.16718749701976776, 0.15898436307907104, 0.271484375, 0.13037109375, 0.27421873807907104, 0.13027343153953552, 0.26191404461860657, 0.1796875, 0.3255859315395355, 0.11513671278953552), (0.0, 0.565264880657196, 0.34388425946235657, 0.17841796576976776, 0.162109375, 0.561718761920929, 0.3359375, 0.5640624761581421, 0.33417966961860657, 0.5347656011581421, 0.3837890625, 0.6214843392372131, 0.3402343690395355), (0.0, 0.23160399496555328, 0.4970458745956421, 0.16640624403953552, 0.1640624701976776, 0.22480468451976776, 0.4898437261581421, 0.22773437201976776, 0.4886718690395355, 0.20703125, 0.5445312261581421, 0.28398436307907104, 0.48554685711860657), (0.0, 0.733349621295929, 0.26240232586860657, 0.17294923961162567, 0.16582031548023224, 0.720703125, 0.259765625, 0.721875011920929, 0.25117185711860657, 0.692187488079071, 0.3072265684604645, 0.7808593511581421, 0.2757812440395355), (0.0, 0.841564953327179, 0.9088500142097473, 0.16660158336162567, 0.16249997913837433, 0.839062511920929, 0.90625, 0.8414062261581421, 0.90625, 0.803906261920929, 0.949999988079071, 0.8976562023162842, 0.9242187142372131), (0.0, 0.5865722298622131, 0.16909180581569672, 0.17392580211162567, 0.177734375, 0.5757812261581421, 0.16787108778953552, 0.580859363079071, 0.16513670980930328, 0.587109386920929, 0.15664061903953552, 0.5679687261581421, 0.16035155951976776), (0.0, 0.522204577922821, 0.8871826529502869, 0.16962890326976776, 0.1558593511581421, 0.518750011920929, 0.8773437142372131, 0.51953125, 0.879687488079071, 0.47187498211860657, 0.9289062023162842, 0.5855468511581421, 0.9007812142372131), (0.0, 0.86474609375, 
0.5472167730331421, 0.16093750298023224, 0.16064453125, 0.8546874523162842, 0.542187511920929, 0.8578124642372131, 0.5390625, 0.811718761920929, 0.598437488079071, 0.9289062023162842, 0.564453125), (0.0, 0.19438475370407104, 0.8609252572059631, 0.16142578423023224, 0.16484370827674866, 0.19414062798023224, 0.85546875, 0.19667968153953552, 0.85546875, 0.16416014730930328, 0.9054687023162842, 0.25312498211860657, 0.8656249642372131), (0.0, 0.529217541217804, 0.7096313834190369, 0.17080076038837433, 0.15449218451976776, 0.5249999761581421, 0.701953113079071, 0.5277343392372131, 0.702343761920929, 0.4886718690395355, 0.747265636920929, 0.587109386920929, 0.719531238079071), (0.0, 0.7617919445037842, 0.0875244140625, 0.17871089279651642, 0.16445313394069672, 0.7554687261581421, 0.07939452677965164, 0.7582030892372131, 0.07773437350988388, 0.7378906011581421, 0.13554687798023224, 0.819531261920929, 0.08173827826976776), (0.0, 0.42673948407173157, 0.24526365101337433, 0.15410155057907104, 0.13974608480930328, 0.42070311307907104, 0.24257813394069672, 0.4234374761581421, 0.2431640625, 0.380859375, 0.27363279461860657, 0.4691406190395355, 0.2681640684604645), (0.0, 0.851367175579071, 0.7275390625, 0.15986327826976776, 0.16523437201976776, 0.8382812142372131, 0.72265625, 0.8414062261581421, 0.7085937261581421, 0.81640625, 0.776562511920929, 0.90234375, 0.7300781011581421), (0.0, 0.384033203125, 0.6023193001747131, 0.16250000894069672, 0.158203125, 0.3746093809604645, 0.602343738079071, 0.37031248211860657, 0.591015636920929, 0.34257811307907104, 0.649609386920929, 0.43437501788139343, 0.622265636920929), (0.0, 0.7144103646278381, 0.44677734375, 0.16171877086162567, 0.16757814586162567, 0.705859363079071, 0.4410156011581421, 0.708203136920929, 0.4410156011581421, 0.678906261920929, 0.4921875, 0.770312488079071, 0.44843748211860657), (0.0, 0.5456786751747131, 0.525683581829071, 0.16435547173023224, 0.14912109076976776, 0.538281261920929, 0.518359363079071, 0.537890613079071, 0.5160155892372131, 0.5160155892372131, 0.5718749761581421, 0.602343738079071, 0.5249999761581421), (0.0, 0.689160168170929, 0.807666003704071, 0.15986332297325134, 0.16777339577674866, 0.6753906011581421, 0.803906261920929, 0.67578125, 0.7914062142372131, 0.6429687142372131, 0.8546874523162842, 0.741015613079071, 0.8265624642372131), (0.0, 0.08608398586511612, 0.3877197206020355, 0.17099609971046448, 0.17851562798023224, 0.06884765625, 0.3832031190395355, 0.06918945163488388, 0.3773437440395355, 0.05195312574505806, 0.4222656190395355, 0.11757811903953552, 0.38984373211860657), (0.0, 0.6944336295127869, 0.6285644173622131, 0.1568359136581421, 0.15898437798023224, 0.680468738079071, 0.6253905892372131, 0.6800780892372131, 0.610546886920929, 0.653515636920929, 0.675000011920929, 0.7421875, 0.6351562142372131), (0.0, 0.11257323622703552, 0.21501463651657104, 0.19130858778953552, 0.1850585788488388, 0.10888671875, 0.20488281548023224, 0.11152344197034836, 0.20429687201976776, 0.07763671875, 0.2583984434604645, 0.171875, 0.22031249105930328), (0.0, 0.08403319865465164, 0.568194568157196, 0.16220702230930328, 0.1664062738418579, 0.071533203125, 0.557812511920929, 0.07480468600988388, 0.556640625, 0.03239746019244194, 
0.565234363079071, 0.09912109375, 0.592578113079071), (0.0, 0.22187499701976776, 0.676708996295929, 0.15937499701976776, 0.16015625, 0.20566405355930328, 0.674609363079071, 0.20175780355930328, 0.657421886920929, 0.1728515625, 0.721875011920929, 0.26835936307907104, 0.688281238079071), (0.0, 0.4483886659145355, 0.06669921427965164, 0.17949219048023224, 0.13232421875, 0.4359374940395355, 0.04584961012005806, 0.43828126788139343, 0.04060058668255806, 0.416015625, 0.09990233927965164, 0.5093749761581421, 0.06069335713982582), (0.0, 0.3669189512729645, 0.788256824016571, 0.17109374701976776, 0.16044917702674866, 0.3525390625, 0.783984363079071, 0.35273435711860657, 0.7757812142372131, 0.32499998807907104, 0.8382812142372131, 0.41796875, 0.8070312142372131), (0.0, 0.06845702975988388, 0.743603527545929, 0.13652344048023224, 0.18886716663837433, 0.0361328125, 0.7398437261581421, 0.036376953125, 0.737109363079071, 0.04702148213982582, 0.7874999642372131, 0.08461914211511612, 0.725390613079071), (0.0, 0.3454833924770355, 0.933837890625, 0.18544919788837433, 0.13271482288837433, 0.33574217557907104, 0.9507812857627869, 0.33515623211860657, 0.9507812857627869, 0.32246091961860657, 
0.9609375, 0.3609375059604645, 0.953906238079071), (0.0, 0.12835693359375, 0.055755615234375, 0.18115234375, 0.11123047024011612, 0.12353515625, 0.02519531175494194, 0.12412109225988388, 0.02326660044491291, 0.09990233927965164, 0.071044921875, 0.1796875, 0.04165038838982582), (0.0, 0.958984375, 0.822583019733429, 0.08212892711162567, 0.17099611461162567, 0.9906249642372131, 0.82421875, 0.992968738079071, 0.823437511920929, 0.9507812857627869, 0.859375, 1.0, 0.8578124642372131), (0.0, 0.6829833984375, 0.9509215950965881, 0.16328124701976776, 0.09682625532150269, 0.6753906011581421, 0.981249988079071, 0.677734375, 0.981249988079071, 0.671093761920929, 0.9859375357627869, 0.698046863079071, 0.973437488079071), (0.0, 0.964855968952179, 0.47016599774360657, 0.06611321866512299, 0.17197264730930328, 0.9906249642372131, 0.48359373211860657, 0.9867187142372131, 0.4769531190395355, 0.96875, 0.5078125, 0.9945312142372131, 0.5035156011581421), (0.0, 0.0632934495806694, 0.917755126953125, 0.12773437798023224, 0.15371087193489075, 0.02937011606991291, 0.9273437857627869, 0.03090820275247097, 0.9273437857627869, 0.02753906138241291, 0.9718749523162842, 0.0869140625, 0.926562488079071), (0.0, 0.965869128704071, 0.639355480670929, 0.07050785422325134, 0.16738279163837433, 0.9906249642372131, 0.63671875, 0.9937500357627869, 0.6351562142372131, 0.973437488079071, 0.643359363079071, 0.98828125, 0.645312488079071), (0.0, 0.27793577313423157, 0.3173767030239105, 0.14335937798023224, 0.15205079317092896, 0.27421873807907104, 
0.31660154461860657, 0.27734375, 0.3154296875, 0.2728515565395355, 0.3677734434604645, 0.32929685711860657, 0.30351561307907104), (0.0, 0.17766113579273224, 0.9733397960662842, 0.16621094942092896, 0.054931640625, 0.16962890326976776, 0.9945312142372131, 0.16396483778953552, 0.9984374642372131, 0.14707030355930328, 0.973437488079071, 0.14414061605930328, 0.996874988079071), (0.0, 0.91162109375, 0.03693847730755806, 0.17402347922325134, 0.07285156100988388, 0.922656238079071, 0.006335448939353228, 0.921875, 0.009069823659956455, 0.8976562023162842, 0.02778320200741291, 0.942187488079071, 0.04025878757238388), (0.0, 0.9635253548622131, 0.9568846821784973, 0.07705086469650269, 0.08583982288837433, 0.9796874523162842, 0.9789062142372131, 0.9796874523162842, 0.965624988079071, 0.9765625, 0.9585937857627869, 0.9718749523162842, 0.9585937857627869)]
    all_points = []
    
    for key_points in key_points_result: 
        # 检查key_points[5]和key_points[6]的数值是不是0.05到0.95之间
        if key_points[5] < 0.03 or key_points[5] > 0.97 or key_points[6] < 0.03 or key_points[6] > 0.97:
            continue
        all_points.append((key_points[5], 1-key_points[6]))

    all_points = np.array(all_points)
    # 随机删除若干点，模拟缺陷
    # delete_indices = np.random.choice(len(all_points), 5, replace=False)
    # real_points = np.delete(all_points, delete_indices, axis=0)
    real_points = all_points
    print(all_points)
    # 需求：寻找边长 n=3 的金字塔三角形，共需 1+2+3=6 个点
    result, center_point, length_center = compute_triangular_orientation(real_points,visualize=False ,neighbor_distance=0.3)

    # tri_group,best_indices = find_triangle_group(real_points,n=4,rotation_angle = result, dist = length_center)
    # tri_group,best_indices = find_kagome_group(real_points,n=4)
    tri_group,best_indices = find_patten_group(real_points,rotation_angle = result, dist = length_center, pattern='honeycomb_lattice')

    # print(f"Calculated Orientation: {result:.2f}° (Ground Truth: {rotation_angle}°)")


    if isinstance(tri_group, np.ndarray):
        print("找到满足要求的三角格点组，其对应的真实坐标如下：")
        print(tri_group)
        # plot the tri_group

        plt.figure(figsize=(8, 8))
        # plt.scatter(perfect_points[:, 0], perfect_points[:, 1], c='black', s=30, label="perfect_points")
        plt.scatter(real_points[:, 0], real_points[:, 1], c='blue', s=60, label="Real Points")
        plt.scatter(tri_group[:, 0], tri_group[:, 1], c='red', s=100, marker='*', label="Triangle Vertices") 

        # 保持 X 和 Y 轴的比例相同
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Triangle Lattice Group")
        # 图例
        plt.legend()
        plt.show()

    else:
        print("未能找到符合要求的完整三角形子格，返回 -1")

    pass
