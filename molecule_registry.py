import logging
import os
import time

import numpy as np

# 获得当前 时间戳
now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
#log save path check if the path exists
log_path = 'log/' +'molecule_log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)

# 配置日志记录
logging.basicConfig(filename=log_path + now + '.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Molecule:
    def __init__(self, position, key_points, site_states = np.array([0,0,0,0]),Br_postion =  np.array([(0,0),(0,0),(0,0),(0,0)]), orientation=0, status=0, operated=False, operated_time=0):
        self.position = position  # 分子位置 (x, y)
        self.key_points = key_points  # 分子关键点列表
        self.site_states = site_states  # 分子各个Br原子的状态
        self.Br_postion = Br_postion        # Br原子位置
        self.orientation = orientation  # 分子取向，默认为0
        self.registration_time = time.time()  # 登记时间，使用当前时间
        self.status = status  # 分子状态，默认为0
        self.operated = operated  # 是否被操作，默认为未操作 (False)
        self.operated_time = operated_time  # 被操作时间，默认为0

    def __repr__(self):
        return (f"Molecule(position={self.position}, key_points={self.key_points}, "
                f"orientation={self.orientation}, registration_time={self.registration_time}, "
                f"status={self.status}, operated={self.operated}, operated_time={self.operated_time})")

class Registry:
    def __init__(self):
        self.molecules = []
        self.threshold = 0.7  # 设置距离阈值   Br 0.5

    def register_molecule(self, position, key_points, site_states = np.array([0,0,0,0]), Br_postion =  np.array([(0,0),(0,0),(0,0),(0,0)]), orientation=0, status=0, operated=False, operated_time=0,fresh_all_position = True):
        new_molecule = Molecule(position, key_points, site_states, Br_postion, orientation, status, operated, operated_time)
        
        # 查找是否有相同的分子
        drift_x_list = []
        drift_y_list = []

        for index, molecule in enumerate(self.molecules):
            if molecule.status == 0:
                distance = np.linalg.norm(np.array(molecule.position)*10**9 - np.array(position)*10**9)
                if distance < self.threshold:
                    # 计算漂移量
                    drift_x = position[0] - molecule.position[0]
                    drift_y = position[1] - molecule.position[1]
                    drift_x_list.append(drift_x)
                    drift_y_list.append(drift_y)
        if drift_x_list and drift_y_list:
            median_drift_x = np.median(drift_x_list)
            median_drift_y = np.median(drift_y_list)            
                    # 修正所有分子的位置
            if fresh_all_position:
                self.move_all_molecules(median_drift_x, median_drift_y)
                logging.info(f"Corrected all molecules by drift ({drift_x}, {drift_y})")

            # 替换旧分子
            for index, molecule in enumerate(self.molecules):
                if molecule.status == 0:
                    distance = np.linalg.norm(np.array(molecule.position) * 10**9 - np.array(position) * 10**9)
                    if distance < self.threshold:
                        self.molecules[index] = new_molecule
                        logging.info(f"Replaced molecule at index {index} with new molecule: {new_molecule}")
                        return

        # 如果没有找到相同的分子，则注册新分子
        self.molecules.append(new_molecule)
        logging.info(f"Registered new molecule: {new_molecule}")

    def get_all_molecules(self):
        return self.molecules

    def move_all_molecules(self, delta_x, delta_y):
        """
        移动所有分子的位置和关键点。
        """
        for molecule in self.molecules:
            molecule.position = (molecule.position[0] + delta_x, molecule.position[1] + delta_y)
            molecule.key_points = [(x + delta_x, y + delta_y) for x, y in molecule.key_points]
        logging.info(f"Moved all molecules by ({delta_x}, {delta_y})")

    def update_molecule(self, index, **kwargs):
        """
        更新指定分子的属性。
        """
        molecule = self.molecules[index]
        for key, value in kwargs.items():
            if hasattr(molecule, key):
                setattr(molecule, key, value)
        logging.info(f"Updated molecule at index {index}: {kwargs}")

    def find_closest_molecule(self, coord):
        """
        找到距离指定坐标最近的分子(状态为0),并返回该分子及其索引。
        """
        if not self.molecules:
            logging.warning("No molecules registered.")
            return None, -1

        min_distance = float('inf')
        closest_molecule = None
        closest_index = -1
        for index, molecule in enumerate(self.molecules):
            # if molecule.status is 0 or molecule.status is np.array([0,0,0,0]), then it should be considered as a candidate
            if molecule.status == 0 or (isinstance(molecule.status, (list, np.ndarray)) and np.array_equal(np.array(molecule.status), np.array([0, 0, 0, 0]))):
                distance = np.linalg.norm(np.array(molecule.position) - np.array(coord))
                if distance < min_distance:
                    min_distance = distance
                    closest_molecule = molecule
                    closest_index = index

        if closest_molecule is None:
            logging.warning(f"No molecule with status 0 found near {coord}.")
            return None, -1

        logging.info(f"Found closest molecule to {coord}: {closest_molecule} at index {closest_index}")
        return closest_molecule, closest_index
    
    def find_closest_molecule_all(self, coord):
        """
        找到距离指定坐标最近的分子(状态为0),并返回该分子及其索引。
        """
        if not self.molecules:
            logging.warning("No molecules registered.")
            return None, -1

        min_distance = float('inf')
        closest_molecule = None
        closest_index = -1
        for index, molecule in enumerate(self.molecules):
            # if molecule.status is 0 or molecule.status is np.array([0,0,0,0]), then it should be considered as a candidate
            # if molecule.status == 0 or (isinstance(molecule.status, (list, np.ndarray)) and np.array_equal(np.array(molecule.status), np.array([0, 0, 0, 0]))):
            distance = np.linalg.norm(np.array(molecule.position) - np.array(coord))
            if distance < min_distance:
                min_distance = distance
                closest_molecule = molecule
                closest_index = index

        if closest_molecule is None:
            logging.warning(f"No molecule with status 0 found near {coord}.")
            return None, -1

        logging.info(f"Found closest molecule to {coord}: {closest_molecule} at index {closest_index}")
        return closest_molecule, closest_index

    def find_first_unreacted_molecule(self):
        """
        找到索引靠前的未反应分子(状态为0或[0, 0, 0, 0])，并返回该分子及其索引。
        """
        if not self.molecules:
            logging.warning("No molecules registered.")
            return None, -1

        for index, molecule in enumerate(self.molecules):
            # 判断分子是否未反应
            if molecule.operated == False:
                logging.info(f"Found first unreacted molecule: {molecule} at index {index}")
                return molecule, index

        logging.warning("No unreacted molecule found.")
        return None, -1
    
    def clear_the_registry(self):
        """
        清空分子登记处。
        """
        self.molecules.clear()
        logging.info("Cleared the registry.")

    def __repr__(self):
        return f"Registry(molecules={self.molecules})"
    

# 示例用法
if __name__ == "__main__":
    registry = Registry()
    registry.register_molecule((10, 20), [(12, 22), (14, 24)], orientation=45)
    registry.register_molecule((10, 20), [(12, 22), (14, 24)], orientation=45)
    registry.register_molecule((30, 40), [(32, 42), (34, 44)], status=1, operated=True, operated_time=time.time())

    print("所有分子:")
    print(registry.get_all_molecules())

    # 移动所有分子
    registry.move_all_molecules(5, 5)
    print("移动后的所有分子:")
    print(registry.get_all_molecules())

    # 更新分子属性
    registry.update_molecule(0, status=1, operated=True, operated_time=time.time())
    print("更新后的分子:")
    print(registry.get_all_molecules())

    # 找到距离指定坐标最近的分子
    closest_molecule, closest_index = registry.find_closest_molecule((15, 25))
    print("距离(15, 25)最近的分子:")
    print(closest_molecule)
    print("最近分子的索引:")
    print(closest_index)