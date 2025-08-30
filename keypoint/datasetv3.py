import sys
import os
import json
import shutil
import random
from PyQt5.QtWidgets import (QApplication, QGroupBox, QWidget, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QProgressBar, QCheckBox, QHBoxLayout, QMessageBox, QSpinBox, QDialog, QFormLayout, QDialogButtonBox, QLabel, QStackedWidget)
from PyQt5.QtCore import Qt
import imageio.v2 as imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
from PIL import Image
import io
import base64
import copy
import numpy as np
import yaml
from PyQt5.QtGui import QFont
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

# 定义一个递归函数来处理和转换所有的 float32 值
def convert_float32(obj):
    if isinstance(obj, list):
        return [convert_float32(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_float32(v) for k, v in obj.items()}
    elif isinstance(obj, np.float32):
        return float(obj)
    return obj

def bbox_to_xywh(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    return x / img_w, y / img_h, w / img_w, h / img_h

class ClassDialog(QDialog):
    def __init__(self, class_labels, parent=None):
        super(ClassDialog, self).__init__(parent)
        self.class_labels = class_labels
        self.class_info = []
        self.setWindowTitle('输入类别信息')

        layout = QVBoxLayout()
        self.form_layout = QFormLayout()
        self.class_id_inputs = []

        for i, label in enumerate(class_labels):
            class_id_input = QLineEdit(self)
            self.form_layout.addRow(f'类别 "{label}" 的 ID:', class_id_input)
            self.class_id_inputs.append(class_id_input)

        layout.addLayout(self.form_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self):
        try:
            for class_id_input, label in zip(self.class_id_inputs, self.class_labels):
                class_id = class_id_input.text().strip()
                if class_id == '':
                    QMessageBox.warning(self, '输入错误', f'类别 "{label}" 的 ID 不能为空。')
                    return
                self.class_info.append((label, int(class_id)))
            super(ClassDialog, self).accept()
        except ValueError:
            QMessageBox.warning(self, '输入错误', '请输入有效的整数类别 ID。')

class CombinedApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图像增强、JSON 转 TXT 转换和数据集划分')
        self.stacked_widget = QStackedWidget()
        self.expert_app = ExpertApp()
        self.basic_app = BasicApp(self.expert_app)
        
        self.stacked_widget.addWidget(self.basic_app)
        self.stacked_widget.addWidget(self.expert_app)
        
        self.switch_button = QPushButton('切换到专家模式', self)
        self.switch_button.clicked.connect(self.switch_mode)
        self.update_switch_button_text()
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.switch_button)
        main_layout.addWidget(self.stacked_widget)
        
        self.setLayout(main_layout)
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid gray;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QProgressBar {
                height: 20px;
                text-align: center;
            }
        """)

    def update_switch_button_text(self):
        if self.stacked_widget.currentIndex() == 0:
            self.switch_button.setText('切换到专家模式')
        else:
            self.switch_button.setText('切换到基础模式')

    def switch_mode(self):
        if self.stacked_widget.currentIndex() == 0:
            self.stacked_widget.setCurrentIndex(1)
        else:
            self.stacked_widget.setCurrentIndex(0)
        self.update_switch_button_text()

class ExpertApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # 增强部分
        aug_layout = QHBoxLayout()
        aug_left_layout = QVBoxLayout()
        aug_right_layout = QVBoxLayout()

        # JSON 文件组
        json_group = QGroupBox("JSON 文件")
        json_layout = QHBoxLayout()
        self.filename_input = QLineEdit(self)
        self.filename_button = QPushButton('浏览', self)
        self.filename_button.clicked.connect(self.browse_filename)
        json_layout.addWidget(self.filename_input)
        json_layout.addWidget(self.filename_button)
        json_group.setLayout(json_layout)

        # 输出路径组
        output_group = QGroupBox("输出路径")
        output_layout = QHBoxLayout()
        self.outputpath_input = QLineEdit(self)
        self.outputpath_button = QPushButton('浏览', self)
        self.outputpath_button.clicked.connect(self.browse_outputpath)
        output_layout.addWidget(self.outputpath_input)
        output_layout.addWidget(self.outputpath_button)
        output_group.setLayout(output_layout)

        # 增强次数组
        aug_times_group = QGroupBox("增强次数")
        aug_times_layout = QHBoxLayout()
        self.aug_times_input = QSpinBox(self)
        self.aug_times_input.setMinimum(1)
        aug_times_layout.addWidget(self.aug_times_input)
        aug_times_group.setLayout(aug_times_layout)

        # 将组添加到左侧布局
        aug_left_layout.addWidget(json_group)
        aug_left_layout.addWidget(output_group)
        aug_left_layout.addWidget(aug_times_group)

        # 增强选项
        self.dropout_checkbox = QCheckBox("Dropout")
        self.elastic_transform_checkbox = QCheckBox("弹性变换")
        self.gaussian_blur_checkbox = QCheckBox("高斯模糊")
        self.multiply_brightness_checkbox = QCheckBox("调整亮度")
        self.multiply_hue_saturation_checkbox = QCheckBox("调整色相和饱和度")
        self.add_hue_checkbox = QCheckBox("增加色相")
        self.add_saturation_checkbox = QCheckBox("增加饱和度")
        self.horizontal_jitter_checkbox = QCheckBox("水平抖动")

        aug_right_layout.addWidget(self.dropout_checkbox)
        aug_right_layout.addWidget(self.elastic_transform_checkbox)
        aug_right_layout.addWidget(self.gaussian_blur_checkbox)
        aug_right_layout.addWidget(self.multiply_brightness_checkbox)
        aug_right_layout.addWidget(self.multiply_hue_saturation_checkbox)
        aug_right_layout.addWidget(self.add_hue_checkbox)
        aug_right_layout.addWidget(self.add_saturation_checkbox)
        aug_right_layout.addWidget(self.horizontal_jitter_checkbox)

        # 进度条和开始按钮
        self.start_button = QPushButton('开始增强', self)
        self.start_button.clicked.connect(self.start_augmentation)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)

        aug_right_layout.addWidget(self.start_button)
        aug_right_layout.addWidget(self.progress_bar)

        # 将左侧和右侧布局添加到主布局
        aug_layout.addLayout(aug_left_layout)
        aug_layout.addLayout(aug_right_layout)

        main_layout.addLayout(aug_layout)

        # 转换部分
        conversion_group = QGroupBox("转换部分")
        conversion_layout = QVBoxLayout()

        # JSON 文件夹组
        json_folder_group = QGroupBox("JSON 文件夹路径")
        json_folder_layout = QHBoxLayout()
        self.json_folder_input = QLineEdit(self)
        self.json_folder_button = QPushButton('浏览', self)
        self.json_folder_button.clicked.connect(self.browse_json_folder)
        json_folder_layout.addWidget(self.json_folder_input)
        json_folder_layout.addWidget(self.json_folder_button)
        json_folder_group.setLayout(json_folder_layout)
        
        # 输出文件夹组
        output_folder_group = QGroupBox("输出文件夹路径")
        output_folder_layout = QHBoxLayout()
        self.output_folder_input = QLineEdit(self)
        self.output_folder_button = QPushButton('浏览', self)
        self.output_folder_button.clicked.connect(self.browse_output_folder)
        output_folder_layout.addWidget(self.output_folder_input)
        output_folder_layout.addWidget(self.output_folder_button)
        output_folder_group.setLayout(output_folder_layout)
        
        # 进度条和开始按钮
        self.class_info_button = QPushButton('输入类别信息', self)
        self.class_info_button.clicked.connect(self.enter_class_info)
        self.convert_button = QPushButton('开始转换', self)
        self.convert_button.clicked.connect(self.start_conversion)
        self.convert_progress_bar = QProgressBar(self)
        self.convert_progress_bar.setAlignment(Qt.AlignCenter)
        self.convert_progress_bar.setMinimum(0)
        self.convert_progress_bar.setMaximum(100)

        # 将小部件添加到转换布局
        conversion_layout.addWidget(json_folder_group)
        conversion_layout.addWidget(output_folder_group)
        conversion_layout.addWidget(self.class_info_button)
        conversion_layout.addWidget(self.convert_button)
        conversion_layout.addWidget(self.convert_progress_bar)

        conversion_group.setLayout(conversion_layout)

        main_layout.addWidget(conversion_group)

        # 数据集划分部分
        splitter_group = QGroupBox("数据集划分器")
        splitter_layout = QVBoxLayout()

        # 根路径组
        root_path_group = QGroupBox("根路径")
        root_path_layout = QHBoxLayout()
        self.root_path_input = QLineEdit()
        self.root_path_button = QPushButton('浏览')
        self.root_path_button.clicked.connect(self.browse_root_path)
        root_path_layout.addWidget(self.root_path_input)
        root_path_layout.addWidget(self.root_path_button)
        root_path_group.setLayout(root_path_layout)
        
        # 训练/测试/验证百分比组
        percent_group = QGroupBox("训练/测试/验证百分比")
        percent_layout = QVBoxLayout()
        
        self.train_test_percent_label = QLabel('训练/测试百分比 (0-100):')
        self.train_test_percent_input = QSpinBox()
        self.train_test_percent_input.setRange(0, 100)
        self.train_test_percent_input.setValue(90)

        self.train_valid_percent_label = QLabel('训练/验证百分比 (0-100):')
        self.train_valid_percent_input = QSpinBox()
        self.train_valid_percent_input.setRange(0, 100)
        self.train_valid_percent_input.setValue(90)

        percent_layout.addWidget(self.train_test_percent_label)
        percent_layout.addWidget(self.train_test_percent_input)
        percent_layout.addWidget(self.train_valid_percent_label)
        percent_layout.addWidget(self.train_valid_percent_input)
        percent_group.setLayout(percent_layout)

        # 划分按钮
        self.split_button = QPushButton('划分和组织数据集')
        self.split_button.clicked.connect(self.split_dataset)

        # 将小部件添加到划分布局
        splitter_layout.addWidget(root_path_group)
        splitter_layout.addWidget(percent_group)
        splitter_layout.addWidget(self.split_button)

        splitter_group.setLayout(splitter_layout)
        main_layout.addWidget(splitter_group)

        self.setLayout(main_layout)
        self.class_info = []
        self.json_folder_path = ""

    def browse_filename(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, '选择 JSON 文件', '', 'JSON 文件 (*.json)', options=options)
        if filename:
            self.filename_input.setText(filename)

    def browse_outputpath(self):
        options = QFileDialog.Options()
        outputpath = QFileDialog.getExistingDirectory(self, '选择输出目录', options=options)
        if outputpath:
            self.outputpath_input.setText(outputpath)

    def start_augmentation(self):
        filename = self.filename_input.text()
        output_path = self.outputpath_input.text()
        aug_times = self.aug_times_input.value()

        if not filename or not output_path:
            QMessageBox.warning(self, '输入错误', '请提供所有必需的输入。')
            return

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(filename, 'r', encoding='utf-8') as f:
            original_squad_box = json.load(f)

        image_data = base64.b64decode(original_squad_box['imageData'])
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)

        rotation_list = list(range(0, 360, 15))  # 旋转角度

        ia.seed(42)

        # 增强序列
        rotation_aug = iaa.Sequential([iaa.Affine(rotate=(0), fit_output=True)])  # 占位符
        sometimes = lambda aug: iaa.Sometimes(1, aug)
        augments = []

        # 添加增强（不包括旋转）
        if self.dropout_checkbox.isChecked():
            augments.append(sometimes(iaa.Dropout((0.01, 0.1), per_channel=0.5)))
        if self.elastic_transform_checkbox.isChecked():
            augments.append(sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)))
        if self.gaussian_blur_checkbox.isChecked():
            augments.append(sometimes(iaa.GaussianBlur(sigma=(0.0, 3.0))))
        if self.multiply_brightness_checkbox.isChecked():
            augments.append(sometimes(iaa.MultiplyBrightness(mul=(0.7, 1.3))))
        if self.multiply_hue_saturation_checkbox.isChecked():
            augments.append(sometimes(iaa.MultiplyHueAndSaturation(mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2))))
        if self.add_hue_checkbox.isChecked():
            augments.append(sometimes(iaa.AddToHue(value=(-10, 10))))
        if self.add_saturation_checkbox.isChecked():
            augments.append(sometimes(iaa.AddToSaturation(value=(-10, 10))))
        if self.horizontal_jitter_checkbox.isChecked():
            augments.append(sometimes(iaa.Affine(translate_percent={"x": (-0.05, 0.05)})))  # 水平位移 -5% 到 5%

        # 组合其他增强
        augment_seq = iaa.Sequential(augments, random_order=True)

        total_operations = len(rotation_list) * aug_times
        progress = 0

        self.progress_bar.setValue(0)
        jpegimages_path = os.path.join(output_path, 'dataset/voc/JPEGImages')
        worktxt_path = os.path.join(output_path, 'dataset/voc/worktxt')

        if not os.path.exists(jpegimages_path):
            os.makedirs(jpegimages_path)
        if not os.path.exists(worktxt_path):
            os.makedirs(worktxt_path)

        # 处理图像和注释
        for theta in rotation_list:
            # 应用旋转到图像和注释
            rotation_aug = iaa.Affine(rotate=theta, fit_output=True)
            squad_box = copy.deepcopy(original_squad_box)

            # 处理形状并将矩形与其关键点分组
            objects = []
            current_object = None

            for shape in squad_box['shapes']:
                if shape['shape_type'] == 'rectangle':
                    # 开始一个新对象
                    current_object = {
                        'rectangle': shape,
                        'keypoints': []
                    }
                    objects.append(current_object)
                elif shape['shape_type'] == 'point':
                    if current_object is not None:
                        current_object['keypoints'].append(shape)
                    else:
                        # 没有当前的矩形可以关联
                        print(f"警告：找到未关联矩形的关键点：{shape}")

            # 准备增强列表
            bbs_list = []
            keypoints_list = []
            keypoints_object_indices = []

            for obj_idx, obj in enumerate(objects):
                # 矩形
                bbox = obj['rectangle']['points']
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                bbs_list.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
                # 关键点
                for kp_shape in obj['keypoints']:
                    point = kp_shape['points'][0]
                    keypoint = Keypoint(x=point[0], y=point[1])
                    keypoints_list.append(keypoint)
                    keypoints_object_indices.append(obj_idx)

            # 创建可增强对象
            bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
            keypoints_on_image = KeypointsOnImage(keypoints_list, shape=image.shape)

            # 应用旋转
            rotated = rotation_aug(image=image, bounding_boxes=bbs, keypoints=keypoints_on_image)
            image_rotated = rotated[0]
            bbs_rotated = rotated[1]
            keypoints_rotated = rotated[2]

            # 对每个旋转后的图像，应用多次增强
            for i in range(aug_times):
                # 应用其他增强
                augmented = augment_seq(image=image_rotated, bounding_boxes=bbs_rotated, keypoints=keypoints_rotated)
                image_aug = augmented[0]
                bbs_aug = augmented[1]
                keypoints_aug = augmented[2]

                # 更新注释
                new_objects = []
                kp_idx = 0
                for obj_idx, obj in enumerate(objects):
                    # 更新矩形
                    bb_aug = bbs_aug.bounding_boxes[obj_idx]
                    x1_new, y1_new = bb_aug.x1, bb_aug.y1
                    x2_new, y2_new = bb_aug.x2, bb_aug.y2
                    new_rectangle = {
                        'label': obj['rectangle']['label'],
                        'points': [[x1_new, y1_new], [x2_new, y2_new]],
                        'group_id': obj['rectangle'].get('group_id'),
                        'shape_type': 'rectangle',
                        'flags': obj['rectangle'].get('flags', {})
                    }
                    # 更新关键点
                    new_keypoints = []
                    for kp_shape in obj['keypoints']:
                        kp_aug = keypoints_aug.keypoints[kp_idx]
                        kp_idx += 1
                        new_kp_shape = {
                            'label': kp_shape['label'],
                            'points': [[kp_aug.x, kp_aug.y]],
                            'group_id': kp_shape.get('group_id'),
                            'shape_type': 'point',
                            'flags': kp_shape.get('flags', {})
                        }
                        new_keypoints.append(new_kp_shape)
                    new_objects.append({
                        'rectangle': new_rectangle,
                        'keypoints': new_keypoints
                    })

                # 更新 squad_box['shapes']
                new_shapes = []
                for obj in new_objects:
                    new_shapes.append(obj['rectangle'])
                    new_shapes.extend(obj['keypoints'])
                squad_box['shapes'] = new_shapes

                # 更新 JSON 文件中的图像和注释
                file_prefix = os.path.splitext(os.path.basename(filename))[0]
                new_image_name = f"{file_prefix}_{theta}_aug_{i}.jpg"
                new_image_path = os.path.join(output_path, new_image_name)

                squad_box['imagePath'] = new_image_name
                # 将增强后的图像转换为 JPEG 并保存为 base64
                pil_img = Image.fromarray(image_aug)
                buff = io.BytesIO()
                pil_img.save(buff, format="JPEG")
                base64_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                squad_box['imageData'] = base64_string
                squad_box['imageHeight'], squad_box['imageWidth'] = image_aug.shape[0:2]
                # 确保将 np.float32 转换为标准 float
                squad_box = convert_float32(squad_box)

                # 保存新的 JSON 文件
                json_name = os.path.splitext(new_image_name)[0] + '.json'
                json_path = os.path.join(worktxt_path, json_name)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(squad_box, f, indent=4, ensure_ascii=False)

                # 保存增强后的图像
                imageio.imwrite(new_image_path, image_aug)
                # 将图像复制到 JPEGImages 文件夹
                shutil.copy(new_image_path, jpegimages_path)

                progress += 1
                self.progress_bar.setValue(int((progress / total_operations) * 100))

        QMessageBox.information(self, '成功', '增强已成功完成。')

    def browse_json_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, '选择 JSON 文件夹', options=options)
        if folder_path:
            self.json_folder_input.setText(folder_path)
            self.json_folder_path = folder_path

    def browse_output_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, '选择输出文件夹', options=options)
        if folder_path:
            self.output_folder_input.setText(folder_path)
 
    def enter_class_info(self):
        if not self.json_folder_path:
            QMessageBox.warning(self, '输入错误', '请先选择 JSON 文件夹路径。')
            return

        class_labels = set()
        for filename in os.listdir(self.json_folder_path):
            if filename.endswith('.json'):
                json_path = os.path.join(self.json_folder_path, filename)
                print(f"处理文件: {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for shape in data['shapes']:
                        if shape['shape_type'] == 'rectangle':
                            label = shape['label']
                            class_labels.add(label)
                            print(f"找到标签: {label}")

        if class_labels:
            print(f"找到的类别标签: {class_labels}")
            dialog = ClassDialog(class_labels=list(class_labels), parent=self)
            if dialog.exec_() == QDialog.Accepted:
                self.class_info = dialog.class_info
                print(f"类别信息: {self.class_info}")
            else:
                QMessageBox.warning(self, '输入错误', '类别信息输入已取消。')
        else:
            QMessageBox.warning(self, '输入错误', '在选定的 JSON 文件中未找到任何类别标签。')

    def start_conversion(self):
        json_folder_path = self.json_folder_input.text()
        output_folder_path = self.output_folder_input.text()

        if not self.class_info:
            QMessageBox.warning(self, '错误', '请在开始转换之前输入类别信息。')
            return

        # 确保输出文件夹存在
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # 函数：归一化坐标
        def normalize_coordinates(coords, img_w, img_h):
            return [(x / img_w, y / img_h) for x, y in coords]

        # 先扫描所有 JSON 文件，确定每个类别的最大关键点数量
        class_max_kp = {class_id: 0 for _, class_id in self.class_info}
        global_max_kp = 0  # 初始化全局最大关键点数量

        for filename in os.listdir(json_folder_path):
            if filename.endswith(".json"):
                json_path = os.path.join(json_folder_path, filename)
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"警告：无法解析 JSON 文件 {json_path}")
                        continue

                    # 处理形状并将矩形与其关键点分组
                    objects = []
                    current_object = None

                    for shape in data.get('shapes', []):
                        if shape.get('shape_type') == 'rectangle':
                            # 开始一个新对象
                            current_object = {
                                'rectangle': shape,
                                'keypoints': []
                            }
                            objects.append(current_object)
                        elif shape.get('shape_type') == 'point':
                            if current_object is not None:
                                current_object['keypoints'].append(shape)
                            else:
                                # 没有当前的矩形可以关联
                                print(f"警告：在文件 {filename} 中找到未关联矩形的关键点：{shape}")

                    for obj in objects:
                        label = obj['rectangle'].get('label')
                        if label is None:
                            continue
                        class_id = next((cid for lbl, cid in self.class_info if lbl == label), None)
                        if class_id is not None:
                            num_kp = len(obj['keypoints'])
                            if num_kp > class_max_kp[class_id]:
                                class_max_kp[class_id] = num_kp
                            if num_kp > global_max_kp:
                                global_max_kp = num_kp

        # Debug: 打印每个类别的最大关键点数量和全局最大关键点数量
        print("每个类别的最大关键点数量:", class_max_kp)
        print("全局最大关键点数量:", global_max_kp)

        # 遍历文件夹中的所有 JSON 文件
        total_files = len([f for f in os.listdir(json_folder_path) if f.endswith(".json")])
        processed_files = 0
        self.convert_progress_bar.setValue(0)

        for filename in os.listdir(json_folder_path):
            if filename.endswith(".json"):
                json_path = os.path.join(json_folder_path, filename)
                output_txt_name = filename.replace('.json', '.txt')
                output_txt_path = os.path.join(output_folder_path, output_txt_name)

                # 读取 JSON 文件
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"警告：无法解析 JSON 文件 {json_path}")
                        continue

                image_width = data.get('imageWidth')
                image_height = data.get('imageHeight')

                if image_width is None or image_height is None:
                    print(f"警告：文件 {filename} 缺少图像宽度或高度信息。")
                    continue

                # 处理形状并将矩形与其关键点分组
                objects = []
                current_object = None

                for shape in data.get('shapes', []):
                    if shape.get('shape_type') == 'rectangle':
                        # 开始一个新对象
                        current_object = {
                            'rectangle': shape,
                            'keypoints': []
                        }
                        objects.append(current_object)
                    elif shape.get('shape_type') == 'point':
                        if current_object is not None:
                            current_object['keypoints'].append(shape)
                        else:
                            # 没有当前的矩形可以关联
                            print(f"警告：在文件 {filename} 中找到未关联矩形的关键点：{shape}")

                # 处理每个对象并写入 TXT 文件
                with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                    for obj in objects:
                        rect = obj['rectangle']
                        label = rect.get('label')
                        if label is None:
                            continue
                        class_id = next((class_id for lbl, class_id in self.class_info if lbl == label), None)
                        if class_id is None:
                            continue  # 如果在 class_info 中未找到标签，则跳过

                        bbox = rect.get('points', [])
                        if len(bbox) != 2:
                            print(f"警告：在文件 {filename} 中，矩形 {label} 的点数不为2。")
                            continue
                        x1, y1 = bbox[0]
                        x2, y2 = bbox[1]
                        x_center, y_center, bbox_w, bbox_h = self.bbox_to_xywh(x1, y1, x2, y2, image_width, image_height)
                        line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"

                        # 获取当前类别的最大关键点数量
                        max_kp = global_max_kp  # 使用全局最大关键点数量

                        # 处理关键点
                        keypoints_coords = []
                        for kp_shape in obj['keypoints']:
                            point = kp_shape.get('points', [[]])[0]
                            if len(point) != 2:
                                print(f"警告：在文件 {filename} 中，关键点格式不正确：{kp_shape}")
                                continue
                            normalized_point = normalize_coordinates([point], image_width, image_height)[0]
                            keypoints_coords.extend([f"{normalized_point[0]:.6f}", f"{normalized_point[1]:.6f}"])

                        # 计算需要补全的关键点数量
                        current_kp = len(obj['keypoints'])
                        if current_kp < global_max_kp:
                            for _ in range(global_max_kp - current_kp):
                                keypoints_coords.extend(["0", "0"])

                        # Debug: 打印每个对象的关键点信息
                        print(f"处理文件 {filename}，类别 {label}，类ID {class_id}")
                        print(f"关键点数量: {current_kp}, 全局最大关键点数量: {global_max_kp}")
                        print(f"关键点坐标: {keypoints_coords}")

                        # 将关键点坐标添加到行中
                        if keypoints_coords:
                            keypoints_str = ' '.join(keypoints_coords)
                            line += f' {keypoints_str}'

                        # Debug: 打印最终写入的行
                        print(f"写入行: {line}")

                        txt_file.write(line + '\n')
                processed_files += 1
                progress_percent = int((processed_files / total_files) * 100)
                self.convert_progress_bar.setValue(progress_percent)

        QMessageBox.information(self, '成功', '所有 JSON 文件已成功转换为 TXT 文件。')





    def bbox_to_xywh(self, x1, y1, x2, y2, img_w, img_h):
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        return x / img_w, y / img_h, w / img_w, h / img_h

    def browse_root_path(self):
        dir = QFileDialog.getExistingDirectory(self, '选择根目录')
        if dir:
            self.root_path_input.setText(dir)

    def split_dataset(self):
        root_path = self.root_path_input.text().replace('\\', '/')
        train_test_percent = self.train_test_percent_input.value() / 100
        train_valid_percent = self.train_valid_percent_input.value() / 100

        jpegimages_path = os.path.join(root_path, 'dataset/voc/JPEGImages').replace('\\', '/')
        txtsavepath = os.path.join(root_path, 'dataset/voc/worktxt').replace('\\', '/')
        ImageSetspath = os.path.join(root_path,'dataset/voc/ImageSets').replace('\\', '/')
        
        if not os.path.exists(ImageSetspath):
            os.makedirs(ImageSetspath)
        if not os.path.exists(jpegimages_path) or not os.path.exists(txtsavepath):
            QMessageBox.warning(self, '错误', '未找到所需的目录。请确保已完成增强和转换。')
            return

        total_images = [f for f in os.listdir(jpegimages_path) if f.endswith('.jpg')]
        total_txts = [f for f in os.listdir(txtsavepath) if f.endswith('.txt')]

        # 获取图像和标签的基名
        image_basenames = set(os.path.splitext(f)[0] for f in total_images)
        label_basenames = set(os.path.splitext(f)[0] for f in total_txts)

        # 找到同时存在的基名
        common_basenames = list(image_basenames & label_basenames)
        num = len(common_basenames)
        if num == 0:
            QMessageBox.warning(self, '错误', '未找到匹配的图像和标签文件。')
            return

        # 随机打乱并划分数据集
        random.shuffle(common_basenames)
        tv = int(num * train_test_percent)
        tr = int(tv * train_valid_percent)
        trainval = common_basenames[:tv]
        train = trainval[:tr]
        valid = trainval[tr:]
        test = common_basenames[tv:]

        # 写入文件列表
        with open(os.path.join(ImageSetspath, 'train.txt'), 'w', encoding='utf-8') as train_txt, \
             open(os.path.join(ImageSetspath, 'valid.txt'), 'w', encoding='utf-8') as valid_txt, \
             open(os.path.join(ImageSetspath, 'test.txt'), 'w', encoding='utf-8') as test_txt, \
             open(os.path.join(ImageSetspath, 'img_train.txt'), 'w', encoding='utf-8') as train_img_txt, \
             open(os.path.join(ImageSetspath, 'img_valid.txt'), 'w', encoding='utf-8') as valid_img_txt, \
             open(os.path.join(ImageSetspath, 'img_test.txt'), 'w', encoding='utf-8') as test_img_txt:

            for basename in train:
                train_txt.write(basename + '.txt\n')
                train_img_txt.write(basename + '.jpg\n')

            for basename in valid:
                valid_txt.write(basename + '.txt\n')
                valid_img_txt.write(basename + '.jpg\n')

            for basename in test:
                test_txt.write(basename + '.txt\n')
                test_img_txt.write(basename + '.jpg\n')

        self.organize_dataset(root_path, txtsavepath, ImageSetspath)

    def organize_dataset(self, root_path, txtsavepath, ImageSetspath):
        img_txt_cg_train = []
        img_txt_cg_test = []
        img_txt_cg_valid = []
        label_txt_cg_train = []
        label_txt_cg_test = []
        label_txt_cg_valid = []

        for line in open(os.path.join(ImageSetspath, "img_train.txt"), 'r', encoding='utf-8'):
            img_txt_cg_train.append(line.strip('\n'))
        for line in open(os.path.join(ImageSetspath, "img_test.txt"), 'r', encoding='utf-8'):
            img_txt_cg_test.append(line.strip('\n'))
        for line in open(os.path.join(ImageSetspath, "img_valid.txt"), 'r', encoding='utf-8'):
            img_txt_cg_valid.append(line.strip('\n'))

        for line in open(os.path.join(ImageSetspath, "train.txt"), 'r', encoding='utf-8'):
            label_txt_cg_train.append(line.strip('\n'))
        for line in open(os.path.join(ImageSetspath, "test.txt"), 'r', encoding='utf-8'):
            label_txt_cg_test.append(line.strip('\n'))
        for line in open(os.path.join(ImageSetspath, "valid.txt"), 'r', encoding='utf-8'):
            label_txt_cg_valid.append(line.strip('\n'))

        new_dataset_train = os.path.join(root_path, 'dataset/voc/data/train/images').replace('\\', '/')
        new_dataset_test = os.path.join(root_path, 'dataset/voc/data/test/images').replace('\\', '/')
        new_dataset_valid = os.path.join(root_path, 'dataset/voc/data/valid/images').replace('\\', '/')

        new_dataset_trainl = os.path.join(root_path, 'dataset/voc/data/train/labels').replace('\\', '/')
        new_dataset_testl = os.path.join(root_path, 'dataset/voc/data/test/labels').replace('\\', '/')
        new_dataset_validl = os.path.join(root_path, 'dataset/voc/data/valid/labels').replace('\\', '/')

        if not os.path.exists(new_dataset_train):
            os.makedirs(new_dataset_train)
        if not os.path.exists(new_dataset_test):
            os.makedirs(new_dataset_test)
        if not os.path.exists(new_dataset_valid):
            os.makedirs(new_dataset_valid)
        if not os.path.exists(new_dataset_trainl):
            os.makedirs(new_dataset_trainl)
        if not os.path.exists(new_dataset_testl):
            os.makedirs(new_dataset_testl)
        if not os.path.exists(new_dataset_validl):
            os.makedirs(new_dataset_validl)

        fimg = os.path.join(root_path, 'dataset/voc/JPEGImages').replace('\\', '/')
        flabel = os.path.join(root_path, 'dataset/voc/worktxt').replace('\\', '/')

        # 复制训练集
        for img_name in img_txt_cg_train:
            shutil.copy(os.path.join(fimg, img_name), new_dataset_train)
        for label_name in label_txt_cg_train:
            shutil.copy(os.path.join(flabel, label_name), new_dataset_trainl)

        # 复制验证集
        for img_name in img_txt_cg_valid:
            shutil.copy(os.path.join(fimg, img_name), new_dataset_valid)
        for label_name in label_txt_cg_valid:
            shutil.copy(os.path.join(flabel, label_name), new_dataset_validl)

        # 复制测试集
        for img_name in img_txt_cg_test:
            shutil.copy(os.path.join(fimg, img_name), new_dataset_test)
        for label_name in label_txt_cg_test:
            shutil.copy(os.path.join(flabel, label_name), new_dataset_testl)

        QMessageBox.information(self, '成功', '数据集已成功组织。')

        # 重新打开并重写 img_train.txt、img_valid.txt 和 img_test.txt 文件
        with open(os.path.join(ImageSetspath, 'img_train.txt'), 'w', encoding='utf-8') as train_img_txt:
            for img_name in img_txt_cg_train:
                train_img_txt.write(os.path.join(root_path, 'dataset/voc/data/train/images/', img_name) + '\n')

        with open(os.path.join(ImageSetspath, 'img_valid.txt'), 'w', encoding='utf-8') as valid_img_txt:
            for img_name in img_txt_cg_valid:
                valid_img_txt.write(os.path.join(root_path, 'dataset/voc/data/valid/images/', img_name) + '\n')

        with open(os.path.join(ImageSetspath, 'img_test.txt'), 'w', encoding='utf-8') as test_img_txt:
            for img_name in img_txt_cg_test:
                test_img_txt.write(os.path.join(root_path, 'dataset/voc/data/test/images/', img_name) + '\n')

        self.generate_yaml(root_path, ImageSetspath)

    # def generate_yaml(self, root_path, ImageSetspath):
    #     # 从 class_info 创建 names 字典
    #     names_dict = {class_id: label for label, class_id in self.class_info}
    #     data = {
    #         'path': os.path.join(root_path, 'dataset').replace('\\', '/'),
    #         'train': os.path.join(ImageSetspath, 'img_train.txt').replace('\\', '/'),
    #         'val': os.path.join(ImageSetspath, 'img_valid.txt').replace('\\', '/'),
    #         'test': os.path.join(ImageSetspath, 'img_test.txt').replace('\\', '/'),
    #         'names': names_dict
    #     }

    #     options = QFileDialog.Options()
    #     yaml_path, _ = QFileDialog.getSaveFileName(self, '保存 YAML 文件', '', 'YAML 文件 (*.yaml)', options=options)
    #     if yaml_path:
    #         with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
    #             yaml.dump(data, yaml_file, allow_unicode=True)
    #         QMessageBox.information(self, '成功', f'YAML 文件已保存到 {yaml_path}')
    def generate_yaml(self, root_path, ImageSetspath):
        # 从 class_info 创建 names 列表，按照类 ID 排序
        names_list = [label for label, class_id in sorted(self.class_info, key=lambda x: x[1])]
        data = {
            'path': os.path.join(root_path, 'dataset').replace('\\', '/'),
            'test': os.path.join(ImageSetspath, 'img_test.txt').replace('\\', '/'),
            'train': os.path.join(ImageSetspath, 'img_train.txt').replace('\\', '/'),
            'val': os.path.join(ImageSetspath, 'img_valid.txt').replace('\\', '/'),
            'nc': len(names_list),
            'names': names_list
        }

        options = QFileDialog.Options()
        yaml_path, _ = QFileDialog.getSaveFileName(self, '保存 YAML 文件', '', 'YAML 文件 (*.yaml)', options=options)
        if yaml_path:
            with open(yaml_path, 'w', encoding='utf-8') as yaml_file:
                # 手动写入 YAML 文件，确保格式与示例一致
                yaml_file.write(f"path: {data['path']}\n")
                yaml_file.write(f"test: {data['test']}\n")
                yaml_file.write(f"train: {data['train']}\n")
                yaml_file.write(f"val: {data['val']}\n\n")
                yaml_file.write(f"# number of classes\n")
                yaml_file.write(f"nc: {data['nc']}\n\n")
                # 将 names_list 转换为所需的字符串格式
                names_str = "['" + "','".join(names_list) + "']"
                yaml_file.write(f"names: {names_str}\n")
            QMessageBox.information(self, '成功', f'YAML 文件已保存到 {yaml_path}')

class BasicApp(QWidget):
    def __init__(self, expert_app):
        super().__init__()
        self.expert_app = expert_app
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # JSON 文件组
        json_group = QGroupBox("JSON 文件")
        json_layout = QHBoxLayout()
        self.filename_input = QLineEdit(self)
        self.filename_button = QPushButton('浏览', self)
        self.filename_button.clicked.connect(self.browse_filename)
        json_layout.addWidget(self.filename_input)
        json_layout.addWidget(self.filename_button)
        json_group.setLayout(json_layout)

        # 增强次数组
        aug_times_group = QGroupBox("增强次数")
        aug_times_layout = QHBoxLayout()
        self.aug_times_input = QSpinBox(self)
        self.aug_times_input.setMinimum(1)
        self.aug_times_input.setValue(5)
        aug_times_layout.addWidget(self.aug_times_input)
        aug_times_group.setLayout(aug_times_layout)

        # 创建数据集按钮
        self.make_dataset_button = QPushButton('创建数据集', self)
        self.make_dataset_button.clicked.connect(self.make_dataset)

        layout.addWidget(json_group)
        layout.addWidget(aug_times_group)
        layout.addWidget(self.make_dataset_button)

        self.setLayout(layout)

    def browse_filename(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, '选择 JSON 文件', '', 'JSON 文件 (*.json)', options=options)
        if filename:
            self.filename_input.setText(filename)
            self.expert_app.filename_input.setText(filename)
            # 自动将输出路径设置为包含 JSON 文件的目录
            self.outputpath = os.path.dirname(filename)

    def make_dataset(self):
        QMessageBox.information(
            self, '请稍候', '请稍候，正在创建数据集...')
        self.expert_app.aug_times_input.setValue(
            self.aug_times_input.value())

        json_file = self.filename_input.text()
        if not json_file or not os.path.exists(json_file):
            QMessageBox.warning(
                self, '错误', '请选择有效的 JSON 文件。')
            return

        # 将输出路径设置为包含 JSON 文件的目录
        output_path = os.path.dirname(json_file)
        self.outputpath = output_path

        # 更新专家应用程序的路径
        self.expert_app.outputpath_input.setText(output_path)
        self.expert_app.json_folder_input.setText(os.path.join(output_path, 'dataset/voc/worktxt'))
        self.expert_app.output_folder_input.setText(
            os.path.join(output_path, 'dataset/voc/worktxt'))
        self.expert_app.root_path_input.setText(output_path)

        # 设置默认值
        self.expert_app.train_test_percent_input.setValue(80)
        self.expert_app.train_valid_percent_input.setValue(80)

        # 选择所有增强复选框
        for checkbox in [
            self.expert_app.dropout_checkbox,
            self.expert_app.elastic_transform_checkbox,
            self.expert_app.gaussian_blur_checkbox,
            self.expert_app.multiply_brightness_checkbox,
            self.expert_app.multiply_hue_saturation_checkbox,
            self.expert_app.add_hue_checkbox,
            self.expert_app.add_saturation_checkbox,
            self.expert_app.horizontal_jitter_checkbox
        ]:
            checkbox.setChecked(True)

        # 开始增强、转换和划分过程
        self.expert_app.start_augmentation()

        # 更新 json_folder_path，以包含新生成的 JSON 文件
        self.expert_app.json_folder_path = os.path.join(output_path, 'dataset/voc/worktxt')
        self.expert_app.json_folder_input.setText(self.expert_app.json_folder_path)
        self.expert_app.enter_class_info()
        if not self.expert_app.class_info:
            QMessageBox.warning(
                self, '错误', '未找到或匹配任何类别信息。')
            return

        self.expert_app.start_conversion()
        self.expert_app.split_dataset()

        QMessageBox.information(
            self, '成功', '数据集已成功创建。')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CombinedApp()
    window.show()
    sys.exit(app.exec_())
