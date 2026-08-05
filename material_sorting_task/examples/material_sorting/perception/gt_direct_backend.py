#!/usr/bin/env python3
"""无渲染真值检测后端 - 直接从MuJoCo获取物体世界坐标。

此后端用于本地无渲染开发，绕过RGB-D相机图像和像素投影，直接从MuJoCo物理引擎
读取彩色盒的真实世界坐标。这使得在 MATERIAL_ENABLE_RENDER=0 时仍能接通完整的
client → perception → detection → control 端到端流程。

与 GtProjectionBackend 的区别：
- GtProjectionBackend：从layout.json读取初始位置 → 投影到像素 → 需要相机图像
- GtDirectBackend：从ROS2话题实时读取MuJoCo物体状态 → 直接返回世界坐标

输出格式与其他backend完全兼容，可在 box_detect.py 中无缝切换。
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GtDirectBackend:
    """直接从MuJoCo真值获取彩色盒世界坐标（无需相机渲染）。

    通过订阅 /material/gt_objects 话题获取Server端发布的物体真值。
    如果话题不存在，则从 layout JSON 读取静态位置（仅适用于固定布局）。
    """

    def __init__(self, layout_path=None, ros_node=None):
        """
        Args:
            layout_path: material_competition_layout.json 路径（备用静态数据源）
            ros_node: ROS2 Node 用于订阅真值话题（推荐方式）
        """
        self.boxes = []
        self.ros_node = ros_node
        self.gt_sub = None

        # 如果提供了ROS节点，订阅真值话题
        if ros_node is not None:
            self.gt_sub = ros_node.create_subscription(
                String, "/material/gt_objects", self._gt_callback, 10
            )
            ros_node.get_logger().info("[GtDirectBackend] subscribed to /material/gt_objects")

        # 只有脱离 ROS 单独使用后端时才允许静态回退。ROS 随机布局下在第一帧
        # 真值到达前发布固定 layout 坐标，会让 client 锁定错误位置。
        if layout_path and ros_node is None:
            import json
            with open(layout_path, "r", encoding="utf-8") as f:
                layout = json.load(f)
            self.boxes = layout["movable_boxes"]
            if ros_node:
                ros_node.get_logger().info(
                    f"[GtDirectBackend] loaded {len(self.boxes)} boxes from layout (static fallback)"
                )

    def _gt_callback(self, msg):
        """从Server端接收物体真值更新"""
        import json
        try:
            data = json.loads(msg.data)
            self.boxes = data.get("movable_boxes", [])
        except Exception as e:
            if self.ros_node:
                self.ros_node.get_logger().warn(f"[GtDirectBackend] failed to parse gt_objects: {e}")

    def detect(self, rgb, depth, K, T_cam_world=None):
        """返回彩色盒的世界坐标。

        Args:
            rgb: RGB图像（此后端不使用，兼容接口）
            depth: 深度图像（此后端不使用，兼容接口）
            K: 相机内参（此后端不使用，兼容接口）
            T_cam_world: 相机->世界变换矩阵（此后端不使用，兼容接口）

        Returns:
            list[dict]: 检测结果列表，每个字典包含：
                - class: 颜色类别 'pink'/'yellow'/'brown'
                - x, y: bbox中心像素坐标（占位，设为图像中心）
                - w, h: bbox像素尺寸（占位，设为固定值）
                - conf: 置信度（真值固定为1.0）
                - world_pos: 世界坐标 [x, y, z]（关键输出）
                - body: MuJoCo body名称
        """
        out = []

        if not self.boxes:
            return out

        # 为了兼容box_detect.py的处理流程，我们需要提供像素坐标占位
        # 但下游会用world_pos直接覆盖投影结果
        H, W = rgb.shape[:2] if rgb is not None else (480, 640)
        placeholder_x = W // 2
        placeholder_y = H // 2

        for box in self.boxes:
            # 只返回可见的盒子（world_position非空）
            if "world_position" not in box:
                continue

            world_pos = np.array(box["world_position"], dtype=float)

            # 检查盒子是否在合理范围内（过滤掉掉落或移出场景的盒子）
            if world_pos[2] < 0.0 or world_pos[2] > 2.0:  # z坐标合理范围
                continue

            out.append({
                "class": box["color"],
                "x": placeholder_x,
                "y": placeholder_y,
                "w": 40,  # 占位bbox尺寸
                "h": 50,
                "conf": 1.0,  # 真值置信度为1.0
                "world_pos": world_pos,  # ⭐ 关键：直接提供世界坐标
                "body": box.get("body", "unknown"),
                "gt_direct": True,  # 标记来源为直接真值
            })

        return out
