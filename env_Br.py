import random
from math import cos, radians, sin, sqrt

import numpy as np
from gym import spaces

from utils import *

if __name__ == "__main__":  # pretend to overload the memory
    import pygame
    
class Molecule:
    def __init__(self, position=(0, 0), shape='square', size = 140, state=0, angle=0):
        self.position = position
        self.shape = shape
        self.size = size
        self.state = state
        self.angle = angle % 360
        self.delta_vector = (0, 0)
        self._key_points = []  # For interaction (tip)
        self._render_points = []  # For origin shape
        self.key_points = []  # For interaction (tip)
        self.render_points = []  # For rendering shape

        # 根据形状设置 key_points 和 render_points
        if shape == 'circle':
            self._key_points = [(self.position[0], self.position[1])]  # center as key point
            self._render_points = [(self.position[0], self.position[1])]  # center for rendering
            self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
            self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)
        
        elif shape == 'triangle':
            # interaction parameters
            self.key_points_size_factor = 0.6
            self.center_std_devs = 18
            self.center_weights = 1
            self.edge_std_devs = 15
            self.edge_weights = 0.6            
            # _key_points: center + 3 vertices (for interaction)
            self._key_points = [
                (self.position[0], self.position[1]),  # Center
                (self.position[0], self.position[1] - sqrt(3) * self.size / 3),  # top
                (self.position[0] - self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # left
                (self.position[0] + self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # right
            ]
            # _render_points: 3 vertices for triangle shape
            self._render_points = [
                (self.position[0], self.position[1] - sqrt(3) * self.size / 3),  # top
                (self.position[0] - self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # left
                (self.position[0] + self.size / 2, self.position[1] + sqrt(3) * self.size / 6),  # right
            ]
            self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
            self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)   

            self.mean = ([0, 0], [0, 0], [0, 0], [0, 0])
            self.std_devs = ([self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.center_std_devs, self.center_std_devs])
            self.weights = [self.edge_weights, self.edge_weights, self.edge_weights, self.center_weights]

            self.optimal_bias = 3.45
            self.optimal_current = 1.0



        elif shape == 'square':
            # 用制表符画一个正方形
            #  



            # interaction parameters
            self.key_points_size_factor = 0.6
            self.center_std_devs = 45
            self.center_weights = 1
            self.edge_std_devs = 40
            self.edge_weights = 0.6 
            # _key_points: center + 4 vertices
            
            half_size = self.size / 2
            self.state = np.array([0, 0, 0, 0])
            self.init_state = np.array([0, 0, 0, 0])
            self.reaction_path_num = 1 # 0 means 'shoulder' path, 1 means 'para' path
            self.reaction_num = 1 # 1 means 0→1 , 2 means 1→2, 3 means 1→3, 4 means 2→4, 5 means 3→4, 6 means 4→5 
            self.state_name = 'full'
            


            self._key_points = [
                (self.position[0], self.position[1]),  # Center
                (self.position[0] - half_size, self.position[1] - half_size),  # top left
                (self.position[0] + half_size, self.position[1] - half_size),  # top right
                (self.position[0] + half_size, self.position[1] + half_size),  # bottom right
                (self.position[0] - half_size, self.position[1] + half_size),  # bottom left
            ]
            # _render_points: 4 vertices for square shape
            self._render_points = [
                (self.position[0] - half_size, self.position[1] - half_size),  # top right
                (self.position[0] + half_size, self.position[1] - half_size),  # bottom right
                (self.position[0] + half_size, self.position[1] + half_size),  # bottom left
                (self.position[0] - half_size, self.position[1] + half_size),  # top left
            ]
        
            self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
            self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)

            """to be updated, it is better to use dictionary to store the all parameters"""
            self.optimal_bias_0000 = 3.45
            self.optimal_current_0000 = 1.0
            self.optimal_bias_0 = 3.45  
            self.optimal_current_0 = 1.0

            # Chemical Synthesis path ：0. 0→1→2→4→5 (shoulder)   1. 0→1→3→4→5 (para)  
            self.optimal_parameters = {
                'full':     {'bias': 2.8,   'current': 0.9 },
                'one':      {'bias': 2.9,   'current': 0.9 },
                'shoulder': {'bias': 3.0,   'current': 1.0  },
                'para':     {'bias': 3.2,   'current': 1.0 },
                'single':   {'bias': 3.2,   'current': 1.1  },
                'empty' :   {'bias': 3.5,  'current': 1.2  },
                'bad' :     {'bias': -1,  'current': -1.0  }
                # Add more states if needed
            }
            self._encode_molecular_state(self.state)

    # def _encode_molecular_state(self, state):
    #     """
    #     编码分子状态，基于反应位点的状态进行编码。
    #     输入：state（列表，包含四个反应位点的状态，0表示未反应，1表示已反应）
    #     输出：对应的编码值（整数）
    #     """
    #     # 检查是否有解体反应失败（bad）状态
    #     if 2 in state:

    #         self.molecular_state = 6

    #     # 统计反应位点的数量
    #     count_1 = np.count_nonzero(state == 1)
        
    #     # 对应状态的分类
    #     if count_1 == 0:
    #         self.state_name = 'full'
    #         self.reaction_num = 1   # to "one"
    #         # change the optimal bias and current
    #         self.optimal_bias_0 = self.optimal_parameters[self.state_name]['bias']
    #         self.optimal_current_0 = self.optimal_parameters[self.state_name]['current']
    #         self.molecular_state = 0  # 所有点位未反应（full）
    #     elif count_1 == 1:
    #         self.state_name = 'one'
    #         if self.reaction_path_num == 0:
    #             self.reaction_num = 2   # to "shoulder"
    #             # change the optimal bias and current
    #             self.optimal_bias_0 = self.optimal_parameters[self.state_name]['bias']
    #             self.optimal_current_0 = self.optimal_parameters[self.state_name]['current']
    #         elif self.reaction_path_num == 1:
    #             self.reaction_num = 3   # to "para"
    #             # change the optimal bias and current
    #             self.optimal_bias_0 = self.optimal_parameters[self.state_name]['bias']
    #             self.optimal_current_0 = self.optimal_parameters[self.state_name]['current']
    #         self.molecular_state = 1  # 只有一个点位反应（one）
    #     elif count_1 == 2:
    #         # 两个邻位反应（shoulder）
    #         if np.array_equal(state, [1, 1, 0, 0]) or np.array_equal(state, [0, 0, 1, 1]) or np.array_equal(state, [1, 0, 0, 1]) or np.array_equal(state, [0, 1, 1, 0]):
    #             self.state_name = 'shoulder'
    #             self.reaction_num = 4   # to "single"
    #             # change the optimal bias and current
    #             self.optimal_bias_0 = self.optimal_parameters[self.state_name]['bias']
    #             self.optimal_current_0 = self.optimal_parameters[self.state_name]['current']
    #             self.molecular_state = 2
    #         # 两个对位点反应（para）
    #         elif np.array_equal(state, [1, 0, 1, 0]) or np.array_equal(state, [0, 1, 0, 1]):
    #             self.state_name = 'para'
    #             self.reaction_num = 5   # to "single"
    #             # change the optimal bias and current
    #             self.optimal_bias_0 = self.optimal_parameters[self.state_name]['bias']
    #             self.optimal_current_0 = self.optimal_parameters[self.state_name]['current']
    #             self.molecular_state = 3
    #     elif count_1 == 3:
    #         self.state_name = 'single'
    #         self.reaction_num = 6   # to "empty"
    #         # change the optimal bias and current
    #         self.optimal_bias_0 = self.optimal_parameters[self.state_name]['bias']
    #         self.optimal_current_0 = self.optimal_parameters[self.state_name]['current']
    #         self.molecular_state = 4  # 三个点位反应（single）
    #     elif count_1 == 4:
    #         self.state_name = 'empty'
    #         self.reaction_num = -1  # for nothing
    #         # change the optimal bias and current
    #         self.optimal_bias_0 = self.optimal_parameters[self.state_name]['bias']
    #         self.optimal_current_0 = self.optimal_parameters[self.state_name]['current']
    #         self.molecular_state = 5  # 所有点位都反应（empty）

    #     self.molecular_state = -1  # 如果输入无效状态

    def _encode_molecular_state(self, state):
        def int_to_one_hot(index, length):
            """
            将整数 index 转换为长度为 length 的 one-hot 向量。
            """
            one_hot = [0] * length
            one_hot[index] = 1
            one_hot[length-1] = self.reaction_path_num
            return one_hot


        # 辅助函数用于更新分子信息
        def update_state(name, mol_state, reaction_num):
            self.state_name = name
            self.molecular_state = int_to_one_hot(mol_state, 7+1)
            self.state_and_path = np.append(self.state, self.reaction_path_num)
            self.reaction_num = reaction_num
            self.optimal_bias_0 = self.optimal_parameters[name]['bias']
            self.optimal_current_0 = self.optimal_parameters[name]['current']

        # 检查“坏”状态
        if 2 in state:
            update_state('bad', 6, -1)
            return self.molecular_state

        # 计算已反应位点数
        count_1 = np.count_nonzero(state == 1)

        if count_1 == 0:
            update_state('full', 0, 1)
        elif count_1 == 1:
            update_state('one', 1, 2 if self.reaction_path_num == 0 else 3)
        elif count_1 == 2:
            # 区分邻位（shoulder）或对位（para）
            if any(np.array_equal(state, arr)
                for arr in ([1,1,0,0],[0,0,1,1],[1,0,0,1],[0,1,1,0])):
                update_state('shoulder', 2, 4)
            elif any(np.array_equal(state, arr)
                    for arr in ([1,0,1,0],[0,1,0,1])):
                update_state('para', 3, 5)
            else:
                self.molecular_state = -1
        elif count_1 == 3:
            update_state('single', 4, 6)
        elif count_1 == 4:
            update_state('empty', 5, -1)
        else:
            self.molecular_state = -1

        return self.molecular_state

    def rotate(self, delta_angle):
        """
        Rotate all points (key_points and render_points) by delta_angle degrees.
        """
        self.angle = (self.angle + delta_angle) % 360
        self.key_points = self._update_points(self._key_points)  # Rotate key points (for tip interaction)
        self.render_points = self._update_points(self._render_points)  # Rotate render points (for drawing)



    def _update_points(self, points, scale_factor=1, translation_vector=(0, 0)):
        """
        Helper function to update points (rotate them).
        """
        # Convert angle from degrees to radians
        angle_rad = radians(self.angle)
        # Unpack the center of rotation and the translation vector
        cx, cy = self.position
        dx, dy = translation_vector

        transformed_points = []

        for (x, y) in points:
            # Step 1: Scale the point
            x_scaled = cx + scale_factor * (x - cx)
            y_scaled = cy + scale_factor * (y - cy)

            # Step 2: Rotate the scaled point
            x_rot = (x_scaled - cx) * cos(angle_rad) - (y_scaled - cy) * sin(angle_rad) + cx
            y_rot = (x_scaled - cx) * sin(angle_rad) + (y_scaled - cy) * cos(angle_rad) + cy

            # Step 3: Translate the rotated point
            x_translated = x_rot + dx
            y_translated = y_rot + dy

            # Append the transformed point to the result list
            transformed_points.append((x_translated, y_translated))
        return transformed_points


    def move(self, translation_vector=(0, 0)):
        """
        Move the molecule by dx and dy.
        """
        self.delta_vector = self.delta_vector + translation_vector
        self.position = (self.position[0] + translation_vector[0], self.position[1] + translation_vector[1])
        self.key_points = self._update_points(self._key_points, translation_vector=self.delta_vector)
        self.render_points = self._update_points(self._render_points, translation_vector=self.delta_vector)

    def interact_area(self, interact_position):
        if self.shape == 'triangle':
            centers = self.key_points
            means = ([0, 0], [0, 0], [0, 0], [0, 0])
            std_devs = ([self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.edge_std_devs, self.edge_std_devs], [self.center_std_devs, self.center_std_devs])
            weights = [self.edge_weights, self.edge_weights, self.edge_weights, self.center_weights]
            
            x = np.linspace(self.position[0]-self.size, self.position[1]+self.size, self.size)
            y = x
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            value = 0
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
                    -(((interact_position[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
                    ((interact_position[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
                )
            # Z = Z / np.max(Z)
            value = value / np.max(Z)

            return value

        elif self.shape == 'square':
            value_list = []
            for petal_site in range(len(self.state)):
                petal_points = np.array([self.render_points[petal_site], self.position])
                centers = petal_points
                means = ([0, 0], [0, 0])
                std_devs = ([self.edge_std_devs, self.edge_std_devs], [self.center_std_devs, self.center_std_devs])
                weights = [self.edge_weights, self.center_weights]

                x = np.linspace(self.position[0]-self.size, self.position[1]+self.size, self.size)
                y = x
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                value = 0
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
                        -(((interact_position[0] - x0 - mean_x) ** 2) / (2 * std_x ** 2) +
                        ((interact_position[1] - y0 - mean_y) ** 2) / (2 * std_y ** 2))*weight
                    )
                # Z = Z / np.max(Z)
                value = value / np.max(Z)
                value_list.append(value)

   
            return value_list
    
    def interact_parameters(self, interact_bias, interact_current):
        if self.shape == 'triangle':

            delta_bias = interact_bias - self.optimal_bias
            delta_current = interact_current - self.optimal_current
            scale_factor_bias = 3
            scale_factor_current = 12
            x = delta_bias*scale_factor_bias
            y = delta_current*scale_factor_current
            suc_rate = suc(x, y)
            ind_rate = ind(x, y)
            non_rate = 1-suc_rate-ind_rate
            return suc_rate, ind_rate, non_rate

        if self.shape == 'square':
            suc_rate_list = []
            ind_rate_list = []
            non_rate_list = []
            for petal_site in self.state:
                delta_bias = interact_bias - self.optimal_bias_0
                delta_current = interact_current - self.optimal_current_0
                scale_factor_bias = 3
                scale_factor_current = 12
                x = delta_bias*scale_factor_bias
                y = delta_current*scale_factor_current
                suc_rate = suc(x, y)
                ind_rate = ind(x, y)
                non_rate = 1-suc_rate-ind_rate
                suc_rate_list.append(suc_rate)
                ind_rate_list.append(ind_rate)
                non_rate_list.append(non_rate)
            return suc_rate_list, ind_rate_list, non_rate_list


    def change_state(self, new_state):
        self.state = new_state
        # update the molecular state
        self._encode_molecular_state(list(new_state))

class Surface:
    def __init__(self, size=(700, 700), angle=0):
        self.size = size
        self.angle = angle
        self.molecules = []
        self.molecules_distance = 138 # 138 → 1.38nm
        self.grid_size = (1, 1)
        self.center = (self.size[0] / 2, self.size[1] / 2)
        self._position_list = generate_triangle_grid(self.center, self.grid_size, self.molecules_distance)
        
        self.position_list = []

        self.position_list = self._update_points(self._position_list)

    def adsorb(self, position):      #size=100 → 1nm
        """
        Creates and absorbs a new molecule.
        """
        molecule = Molecule(position=position)
        self.molecules.append(molecule)
    
    def _update_points(self, points, scale_factor=1, translation_vector=(0, 0)):
        """
        Helper function to update points (rotate them).
        """
        # Convert angle from degrees to radians
        angle_rad = radians(self.angle)
        # Unpack the center of rotation and the translation vector
        cx, cy = self.center
        dx, dy = translation_vector

        transformed_points = []

        for (x, y) in points:
            # Step 1: Scale the point
            x_scaled = cx + scale_factor * (x - cx)
            y_scaled = cy + scale_factor * (y - cy)

            # Step 2: Rotate the scaled point
            x_rot = (x_scaled - cx) * cos(angle_rad) - (y_scaled - cy) * sin(angle_rad) + cx
            y_rot = (x_scaled - cx) * sin(angle_rad) + (y_scaled - cy) * cos(angle_rad) + cy

            # Step 3: Translate the rotated point
            x_translated = x_rot + dx
            y_translated = y_rot + dy

            # Append the transformed point to the result list
            transformed_points.append((x_translated, y_translated))
        return transformed_points
    
    # def a function to rotate whole surface
    def rotate(self, delta_angle):
        """
        Rotate all points (key_points and render_points) by delta_angle degrees.
        """
        self.angle = (self.angle + delta_angle) % 360
        self.position_list = self._update_points(self._position_list)

    def arrange_molecules(self):
        
        for position in self.position_list:
            self.adsorb(position)
        self.position_list = self._update_points(self._position_list)
        

class Tip:
    def __init__(self, position=(0, 0)):
        self.position = position
        self.reaction_time = 0
        self.bias_voltage = 3.45
        self.current = 1.0
        self.suc_factor = 0
        self.fail_factor = 0
        self.non_factor = 0
        
    def move(self, new_position):
        self.position = new_position
    
    def change_bias(self, delta_bias):
        self.bias_voltage += delta_bias

    def change_current(self, delta_current):
        self.current += delta_current

    def culculate_reaction_factor(self, molecule, weight_suc = 0.9, weight_fail = 0.05):

        distance = np.linalg.norm(np.array(self.position) - np.array(molecule.position))
        if distance > molecule.size * 2:

            return np.array([0,0,0,0]), np.array([0,0,0,0]), np.array([1,1,1,1])
        
        position_factor = molecule.interact_area(self.position)

        parameters_suc_factor, parameters_ind_factor, parameters_non_factor = molecule.interact_parameters(self.bias_voltage, self.current)
        # print("parameters_ind_factor:", parameters_ind_factor)

        # convert the position_factor, parameters_suc_factor, parameters_ind_factor to np.array
        position_factor = np.array(position_factor)
        parameters_suc_factor = np.array(parameters_suc_factor)
        parameters_ind_factor = np.array(parameters_ind_factor)

        suc_factor = position_factor*parameters_suc_factor*weight_suc
        fail_factor = position_factor*weight_fail + parameters_ind_factor
        position_non_factor = 1 - suc_factor - fail_factor
        
        self.suc_factor = suc_factor
        self.fail_factor = fail_factor
        self.non_factor = position_non_factor

        return suc_factor, fail_factor, position_non_factor

    def interact(self, molecule):
        bias_voltage = self.bias_voltage
        current = self.current

        distance = np.linalg.norm(np.array(self.position) - np.array(molecule.position))
        if distance > molecule.size * 2:
            self.reaction_time = 10
            return 0

        # 直接使用 culculate_reaction_factor 计算反应因子
        suc_factor, fail_factor, non_factor= self.culculate_reaction_factor(molecule)

        # 计算反应时长
        # self.reaction_time = max((non_factor + parameters_non_factor) * 6 + random.gauss(0, 1), 0.01)

        for i, site_state in enumerate(molecule.state):
            if site_state == 0:
                action_random = random.random()
                if action_random < suc_factor[i]:
                    molecule.state[i] = 1
                elif action_random > 1 - fail_factor[i]:
                    molecule.state[i] = 2
        molecule.molecular_state = molecule._encode_molecular_state(molecule.state)



        # if molecule.state == 0:
        #     action_random = random.random()
        #     if action_random < suc_factor:
        #         molecule.change_state(1)
        #     elif action_random > 1 - fail_factor:
        #         molecule.change_state(2)
            # else:
            #     self.reaction_time = random.choice([self.reaction_time, 8])


        # success_probability = max(0.8 - 0.05 * bias_deviation - 0.05 * current_deviation - 0.1 * distance_penalty, 0)
        # fail_probability = max(0.5 + 0.1 * bias_deviation + 0.1 * current_deviation - 0.1 * distance_penalty, 0)

class Env:
    def __init__(self,polar_space = False):
        self.surface = Surface()
        self.tip = Tip()
        # self.screen = pygame.display.set_mode((1000,1000))
        self.running = True
        self.polar_space = polar_space
        self.render_flag = 1

        # 新增：记录按键和鼠标按钮的按下时刻与上次触发时刻
        self.key_hold_data = {}      # {key: {"press_time": float, "last_trigger": float}}
        self.mouse_hold_data = {}    # {button: {"press_time": float, "last_trigger": float}}
        self.REPEAT_DELAY = 0.5      # 按住 0.5 秒后开始连续触发
        self.REPEAT_INTERVAL = 0.1   # 连续触发的间隔        
        xy_grid_level_1 = 5
        xy_grid_level_2 = 3

        v_grid_level = 25
        i_grid_level = 23

        if self.polar_space:
            action_low_x, action_high_x = 0, 70
            action_low_y, action_high_y = 0, 360
        else:
            # action_low_x, action_high_x = -200, 200
            # action_low_y, action_high_y = -200, 200
            action_low_x, action_high_x = -80, 80
            action_low_y, action_high_y = -80, 80

        self.action_space = spaces.Box(
            low=np.array([action_low_x, action_low_y, 3.2, 0.3]),       #low bias, low current  2.4, 0.3
            high=np.array([action_high_x, action_high_y, 3.6, 1.5]),    #high bias, high current  4.2, 2.0
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([0]),
            high=np.array([2]),
            dtype=np.float32,
        )

        self.action_space_tip = spaces.Box(
            low=np.array([ 2, 0.1]),
            high=np.array([ 4, 1.5]),
            dtype=np.float32,
        )

        # xy_center = ((self.action_space.low[0] + self.action_space.high[0]) / 2,(self.action_space.low[1] + self.action_space.high[1]) / 2)
        # # self.xy_grid_points = generate_grid(xy_center, xy_grid_level_1 , xy_grid_level_1, self.action_space.high[0] - self.action_space.low[0], self.action_space.high[1] - self.action_space.low[1])
        # self.xy_grid_points_1 = generate_grid(xy_center, xy_grid_level_1 , xy_grid_level_2, self.action_space.high[0] - self.action_space.low[0], self.action_space.high[1] - self.action_space.low[1])
        # # print("xy_grid_12:", self.xy_grid_points_1[0][1])


        
        # self.xy_grid_points_1[1][0] = self.xy_grid_points_1[1][0] + (self.action_space.high[0] - self.action_space.low[0])/(xy_grid_level_2+1)
        # self.xy_grid_points_1[3][0] = self.xy_grid_points_1[3][0] + (self.action_space.high[0] - self.action_space.low[0])/(xy_grid_level_2+1)
        # self.xy_grid_points_1[1][1] = self.xy_grid_points_1[1][1] + (self.action_space.high[0] - self.action_space.low[0])/(xy_grid_level_2+1)
        # self.xy_grid_points_1[3][1] = self.xy_grid_points_1[3][1] + (self.action_space.high[0] - self.action_space.low[0])/(xy_grid_level_2+1)
        # self.xy_grid_points_1[1][2] = np.array([0, 0])
        # self.xy_grid_points_1[3][2] = np.array([0, 0])
        
        # self.xy_grid_points = self.xy_grid_points_1
        self.xy_grid_points = np.array([[[-70,70] , [0,70]  , [70,70]],
                                        [[-35,-35], [35,-35], [0,0]],
                                        [[-70,0],   [0,0],    [70,0]],
                                        [[-35,35],  [35,35],  [0,0]],
                                        [[-70,-70], [0,-70],  [70,-70]]])
        self.H_xy_grid_points = np.array([[[-70,-40.4] , [70,-40.4]  , [0,80.8]], # out top
                                        [[-35,20.2], [35,20.2], [0,40.4]], # out edge  
                                        [[0,40.4],   [-35,20.2],    [35,20.2]], #inter mid
                                        [[0,0],  [0,0],  [0,0]]])
        # print(self.xy_grid_points)

        vi_center = ((self.action_space.low[2] + self.action_space.high[2]) / 2,(self.action_space.low[3] + self.action_space.high[3]) / 2)
        self.vi_grid_points = generate_grid(vi_center, v_grid_level , i_grid_level, self.action_space.high[2] - self.action_space.low[2], self.action_space.high[3] - self.action_space.low[3])


    def render(self):
        # if self.render_flag == 1:# initialize the pygame
        #     pygame.init()
        #     self.screen = pygame.display.set_mode(self.surface.size, pygame.RESIZABLE)
        #     pygame.display.set_caption("Interactive STM Environment")
        #     self.clock = pygame.time.Clock()
        #     self.render_flag = 0  # set the flag to 0 to avoid initialize the pygame again
        self.screen.fill((255, 255, 255))
        for molecule in self.surface.molecules:
            if molecule.shape == 'triangle':
                color = (0, 0, 255) if molecule.state == 0 else (0, 255, 0) if molecule.state == 1 else (255, 0, 0)
                pygame.draw.polygon(self.screen, color, molecule.render_points)
                # draw the key points
                for point in molecule.key_points:
                    pygame.draw.circle(self.screen, (0, 0, 0), (int(point[0]), int(point[1])), 3)
            
            elif molecule.shape == 'square':
                body_color = (0, 0, 255) 
                sub_body_color = (0, 0, 128)
                # draw the body
                pygame.draw.polygon(self.screen, body_color, molecule.render_points)
                # draw the Br sites
                for i, point in enumerate(molecule.render_points):
                    if molecule.state[i] == 0:
                        sub_body_color = (0, 0, 128) 
                    elif molecule.state[i] == 1:
                        sub_body_color = (0, 255, 0)
                    elif molecule.state[i] == 2:
                        sub_body_color = (255, 0, 0)
                    
                    pygame.draw.circle(self.screen, sub_body_color, (int(point[0]), int(point[1])), 30)
                # draw the key points
                for point in molecule.key_points:
                    pygame.draw.circle(self.screen, (0, 0, 0), (int(point[0]), int(point[1])), 3)

        # draw the tip
        pygame.draw.circle(self.screen, (0, 255, 0), self.tip.position, 5)
        # show the tip position number at the self.tip.position
        font = pygame.font.SysFont("arial", 12)
        text = font.render("Tip Position: ({:.2f}, {:.2f})".format(self.tip.position[0], self.tip.position[1]), True, (0, 0, 0))
        self.screen.blit(text, (self.tip.position[0], self.tip.position[1]))
        # show the bias voltage and current at the top left corner
        font = pygame.font.SysFont("arial", 12)
        text = font.render("Tip Bias : {:.2f}V  Tip Current: {:.2f}nA\nOptimal bias: {:.2f}V  Optimal Current: {:.2f}nA".format(self.tip.bias_voltage, self.tip.current, self.surface.molecules[0].optimal_bias_0, self.surface.molecules[0].optimal_current_0), True, (0, 0, 0))
        self.screen.blit(text, (10, 20))
        # show the success rate, indeterminate rate and failure rate at the bottom left corner
        start_y = 300
        line_height = 15
        for i in range(len(self.tip.suc_factor)):
            line_text = f"{i+1}  Suc: {self.tip.suc_factor[i]:.2f}  Non: {self.tip.non_factor[i]:.2f}  Fail: {self.tip.fail_factor[i]:.2f}"
            text_line = font.render(line_text, True, (0, 0, 0))
            self.screen.blit(text_line, (10, start_y + i * line_height))
        # show the molecule state at the bottom of the tip.suc_factor
        start_y = 300 + len(self.tip.suc_factor) * line_height
        line_molecule_state_text = f"Molecule State: {self.surface.molecules[0].molecular_state}"
        text_molecule_state_line = font.render(line_molecule_state_text, True, (0, 0, 0))
        self.screen.blit(text_molecule_state_line, (10, start_y))
        pygame.display.flip()

    def reset(self):
        """
        重置环境到初始状态，并返回初始观测值。
        """
        self.surface.molecules = []
        self.surface.arrange_molecules()
        for molecule in self.surface.molecules:
            molecule.change_state(np.array([0, 0, 0, 0]))
        self.tip.move((self.surface.center[0], self.surface.center[1]))
        molecule = self.surface.molecules[0]  # 默认选择第一个分子
        # self.current_molecule = molecule
        # return np.array(molecule.molecular_state)
        return np.array(molecule.state_and_path)
    
    # TODO: redef the reward according to the reaction path
    # culculate the reward
    # def reward_culculate(self,state_init, state, reaction_time):
    #     if state[6] == 1:  # bad
    #         info ={"reaction" : "bad", "reaction_time" : reaction_time}
    #         return -10 - 0.2 * reaction_time, info
    #     elif state_init[0] == 1 and state[1] == 1: #mol 0→1
    #         info ={"reaction" : "mol 0→1", "reaction_time" : reaction_time}
    #         return 2 - 0.2 * reaction_time, info
    #     elif state_init[1] == 1 and state[2] == 1: #mol 1→2
    #         if state_init[-1] ==0:  # 'shoulder'
    #             info ={"reaction" : "mol 1→2", "reaction_time" : reaction_time}
    #             return 4 - 0.2 * reaction_time, info
    #         else:                   # 'para'
    #             info ={"reaction" : "mol 1→2 error", "reaction_time" : reaction_time}
    #             return -4 - 0.2 * reaction_time, info
    #     elif state_init[1] == 1 and state[3] == 1: #mol 1→3
    #         if state_init[-1] ==0:  # 'shoulder'
    #             info ={"reaction" : "mol 1→3 error", "reaction_time" : reaction_time}
    #             return -4 - 0.2 * reaction_time, info
    #         else:                   # 'para'
    #             info ={"reaction" : "mol 1→3", "reaction_time" : reaction_time}
    #             return 4 - 0.2 * reaction_time, info
    #     elif state_init[2] == 1 and state[4] == 1: #mol 2→4
    #         info ={"reaction" : "mol 2→4", "reaction_time" : reaction_time}
    #         return 6 - 0.2 * reaction_time, info
    #     elif state_init[3] == 1 and state[4] == 1: #mol 3→4
    #         info ={"reaction" : "mol 3→4", "reaction_time" : reaction_time}
    #         return 6 - 0.2 * reaction_time, info
    #     elif state_init[4] == 1 and state[5] == 1: #mol 4→5
    #         info ={"reaction" : "mol 4→5", "reaction_time" : reaction_time}
    #         return 8 - 0.2 * reaction_time, info
    #     elif state_init[5] == 1 and state[6] == 1: #mol 5→6
    #         info ={"reaction" : "mol 5→6", "reaction_time" : reaction_time}
    #         return 10 - 0.2 * reaction_time, info
    #     else:
    #         info ={"reaction" : "nothing", "reaction_time" : reaction_time}
    #         return -1, info
    
    def reward_culculate(self,state_init, state, reaction_time):
        ligal_state = [[np.array([0,0,0,0,0]), np.array([1,0,0,0,0]), np.array([1,1,0,0,0]), np.array([1,1,1,0,0]), np.array([1,1,1,1,0])], 
                       [np.array([0,0,0,0,1]), np.array([1,0,0,0,1]), np.array([1,0,1,0,1]), np.array([1,1,1,0,1]), np.array([1,1,1,1,1])]]

        if 2 in state:                                                                                  # bad
            info ={"reaction" : "bad", "reaction_time" : reaction_time}
            return -4 - 0.2 * reaction_time, info
        if not any(np.array_equal(state, ls) for ls in ligal_state[0]) and not any(np.array_equal(state, ls) for ls in ligal_state[1]):  # wrong
            info = {"reaction": "wrong", "reaction_time": reaction_time}
            # get the count of 1 in the state
            count_1 = np.count_nonzero(state == 1)
            if count_1 == 0:
                return -2 - 0.2 * reaction_time, info
            if count_1 == 1:
                return -3 - 0.2 * reaction_time, info
            if count_1 == 2:
                return -4 - 0.2 * reaction_time, info
        if np.array_equal(state, state_init):                                                           # non
            info = {"reaction": "non", "reaction_time": reaction_time}
            return -1 - 0.2 * reaction_time, info
        
        if np.array_equal(state_init, ligal_state[0][0]) and np.array_equal(state, ligal_state[0][1]):  # mol 0→1
            info = {"reaction": "mol 0→1", "reaction_time": reaction_time}
            return 4 - 0.2 * reaction_time, info

        if np.array_equal(state_init, ligal_state[0][1]) and np.array_equal(state, ligal_state[0][2]):  # mol 1→2
            info = {"reaction": "mol 1→2", "reaction_time": reaction_time}
            return 6 - 0.2 * reaction_time, info

        if np.array_equal(state_init, ligal_state[0][2]) and np.array_equal(state, ligal_state[0][3]):  # mol 2→4
            info = {"reaction": "mol 2→4", "reaction_time": reaction_time}
            return 8 - 0.2 * reaction_time, info

        if np.array_equal(state_init, ligal_state[0][3]) and np.array_equal(state, ligal_state[0][4]):  # mol 4→5
            info = {"reaction": "mol 4→5", "reaction_time": reaction_time}
            return 8 - 0.2 * reaction_time, info

        if np.array_equal(state_init, ligal_state[1][0]) and np.array_equal(state, ligal_state[1][1]):  # mol 0→1
            info = {"reaction": "mol 0→1", "reaction_time": reaction_time}
            return 4 - 0.2 * reaction_time, info

        if np.array_equal(state_init, ligal_state[1][1]) and np.array_equal(state, ligal_state[1][2]):  # mol 1→3
            info = {"reaction": "mol 1→3", "reaction_time": reaction_time}
            return 6 - 0.2 * reaction_time, info

        if np.array_equal(state_init, ligal_state[1][2]) and np.array_equal(state, ligal_state[1][3]):  # mol 3→4
            info = {"reaction": "mol 3→4", "reaction_time": reaction_time}
            return 8 - 0.2 * reaction_time, info

        if np.array_equal(state_init, ligal_state[1][3]) and np.array_equal(state, ligal_state[1][4]):  # mol 4→5
            info = {"reaction": "mol 4→5", "reaction_time": reaction_time}
            return 8 - 0.2 * reaction_time, info

        info = {"reaction": "skip", "reaction_time": reaction_time}  # state skip
        return -5, info
        
        
    def H_reward_culculate(self,state_init, state, reaction_time):
        if state_init[0]==1 and state[0] == 2:                # success
            info ={"reaction" : "success", "reaction_time" : reaction_time}
            return 10 - 0.2 * reaction_time, info
            
        elif state_init[0]==1 and state[0] == 1:              # none
            info ={"reaction" : "none", "reaction_time" : reaction_time}
            return -0.1 - 0.2 * reaction_time, info
        
        elif state_init[0]==1 and state[0] == 3:              # failure
            info ={"reaction" : "bad", "reaction_time" : reaction_time}
            return -5 - 0.2 * reaction_time, info
        else:
            return -10, {"reaction" : "NO STATE ERROR!"}

        
    # def reward_culculate(self,state_init, state, reaction_time):


    def step(self, action):
        """
        执行动作，更新环境状态并返回:
        - 观测值
        - 奖励
        - 是否结束
        - 附加信息
        """
        done = 0
        if self.polar_space:
            l, theta, v, a = action # theta in 0-360
            x = l * np.cos(np.radians(theta))
            y = l * np.sin(np.radians(theta))
        else:
            x, y, v, a = action                 # x,y is the relative position to the center of the molecule
        # self.tip.move((x, y))
        (x,y) =  find_nearest_grid_point(self.xy_grid_points, (x,y))
        (v,a) =  find_nearest_grid_point(self.vi_grid_points, (v,a))

        for molecule in self.surface.molecules[0:]:
            state_init = molecule.state_and_path
            self.tip.move((molecule.position[0]+x, molecule.position[1]+y))
            self.tip.bias_voltage = v
            self.tip.current = a
            self.tip.interact(molecule)

        # 获取反应结果
        # state = molecule.molecular_state
        state = molecule.state_and_path
        reaction_time = self.tip.reaction_time
        reaction_time = 0

        reward, info = self.reward_culculate(state_init, state, reaction_time)

        # 是否结束， 如果state中有2，则结束, 或者state中的前4个元素都是1，则结束
        if 2 in state or all(state[:4]) or info["reaction"] in ["bad", "wrong"]:  # broken or all Br sites are occupied or wrong reaction
            done = 1

        # 返回新的状态和奖励
        return np.array(state, dtype=np.float32), reward, done, info
    
    def run(self):
        self.surface.arrange_molecules()
        pygame.init()
        self.screen = pygame.display.set_mode(self.surface.size, pygame.RESIZABLE)
        pygame.display.set_caption("Interactive STM Environment")
        self.clock = pygame.time.Clock()

        while self.running:
            # 处理事件：keydown、keyup、mousedown、mouseup等
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    # 记录按下时刻，避免重复覆盖
                    if event.key not in self.key_hold_data:
                        now = time.time()
                        self.key_hold_data[event.key] = {
                            "press_time": now, 
                            "last_trigger": now
                        }
                    # 初次按下立即触发一次
                    self._handle_key_once(event.key)
                elif event.type == pygame.KEYUP:
                    # 松开后移除记录
                    if event.key in self.key_hold_data:
                        del self.key_hold_data[event.key]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # 记录按下时刻
                    if event.button not in self.mouse_hold_data:
                        now = time.time()
                        self.mouse_hold_data[event.button] = {
                            "press_time": now,
                            "last_trigger": now
                        }
                    # 初次按下立即触发一次
                    self._handle_mouse_once(event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in self.mouse_hold_data:
                        del self.mouse_hold_data[event.button]
               
                elif event.type == pygame.MOUSEMOTION:
                    self.tip.move(event.pos)
                    for molecule in self.surface.molecules:
                        self.tip.culculate_reaction_factor(molecule)

            # 每帧检查是否有按住超 0.5 秒的情况，若是则间歇性重复触发
            self._process_held_keys()
            self._process_held_mouse()

            self.render()
            self.clock.tick(60)
        pygame.quit()

    def _process_held_keys(self):
        now = time.time()
        for key, data in self.key_hold_data.items():
            if now - data["press_time"] >= self.REPEAT_DELAY:
                if now - data["last_trigger"] >= self.REPEAT_INTERVAL:
                    data["last_trigger"] = now
                    self._handle_key_once(key)  # 重复触发

    def _process_held_mouse(self):
        now = time.time()
        for button, data in self.mouse_hold_data.items():
            if now - data["press_time"] >= self.REPEAT_DELAY:
                if now - data["last_trigger"] >= self.REPEAT_INTERVAL:
                    data["last_trigger"] = now
                    self._handle_mouse_once(button)  # 重复触发

    def _handle_key_once(self, key):
        """ 根据具体按键，执行原先一次性触发的逻辑 """
        if key == pygame.K_r:
            for molecule in self.surface.molecules:
                molecule.change_state(np.array([0, 0, 0, 0]))
        elif key == pygame.K_SPACE:
            for molecule in self.surface.molecules:
                print(molecule.render_points)
        elif key == pygame.K_RIGHT:
            for molecule in self.surface.molecules:
                molecule.rotate(1)
            print(molecule.angle)
        elif key == pygame.K_LEFT:
            for molecule in self.surface.molecules:
                molecule.rotate(-1)
            print(molecule.angle)
        elif key == pygame.K_w:
            self.tip.change_bias(0.05)
            print("bias:", self.tip.bias_voltage)
        elif key == pygame.K_s:
            self.tip.change_bias(-0.05)
            print("bias:", self.tip.bias_voltage)
        elif key == pygame.K_a:
            self.tip.change_current(-0.05)
            print("current:", self.tip.current)
        elif key == pygame.K_d:
            self.tip.change_current(0.05)
            print("current:", self.tip.current)

    def _handle_mouse_once(self, button):
        """ 根据具体鼠标按钮，执行原先一次性触发的逻辑 """
        # 这里只演示左键(1)，可以扩展到其他按钮
        if button == 1:  # 左键
            for molecule in self.surface.molecules:
                self.tip.interact(molecule)

    def close(self):
        """
        关闭环境。
        """
        self.running = False
        pygame.quit()

if __name__ == "__main__":

    env = Env()
    env.run()

    # R, info = env.reward_culculate([1,0,1,0,1], [1,1,1,0,1], 0)
    # print(R, info)



