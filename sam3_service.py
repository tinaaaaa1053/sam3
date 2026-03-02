import os
import io
import uuid
import torch
import pickle
import logging
import numpy as np
from PIL import Image

# SAM 3 引用
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor

# 定义临时文件存放目录
TEMP_DIR = "./temp_uploads"
OUTPUT_DIR = "./temp_outputs"

# 确保目录存在
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

class SAM3Service:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAM3Service, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        logger.info("正在加载 SAM 3 模型...")
        try:
            self.image_model = build_sam3_image_model()
            self.image_processor = Sam3Processor(self.image_model)
            # 显存允许的话加载视频模型
            self.video_predictor = build_sam3_video_predictor()
            logger.info("SAM 3 模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e

    def process_image_file(self, image_path: str, prompt: str) -> str:
        """
        处理本地图片文件，保存结果为 pkl，返回 pkl 路径
        """
        try:
            # 1. 读取图片
            image_pil = Image.open(image_path).convert("RGB")
            
            
            # 2. 推理
            inference_state = self.image_processor.set_image(image_pil)
            output = self.image_processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            # 3. 提取数据并转为 Numpy
            # --- 修改开始 ---
            
            # Masks 通常是布尔值，直接转即可；如果是数值型，也建议先转 float
            # 为了保险，我们先判断一下，或者直接用 .cpu().numpy() 对于布尔值没问题
            masks_tensor = output["masks"]
            if masks_tensor.dim() == 4:   # [B,1,H,W]
                masks_tensor = masks_tensor.squeeze(1)

            masks_np = masks_tensor.cpu().numpy()

            # Boxes 和 Scores 是 BFloat16，必须先转成 float32
            boxes_np = output["boxes"].float().cpu().numpy()   # 加了 .float()
            scores_np = output["scores"].float().cpu().numpy() # 加了 .float()
            
            # --- 修改结束 ---

            # 4. 封装结果
            result_data = {
                "type": "image_result",
                "prompt": prompt,
                "masks": masks_np,
                "boxes": boxes_np,
                "scores": scores_np,
                "image_size": image_pil.size
            }

            # 5. 保存为 Pickle 文件
            pkl_filename = f"sam3_img_{uuid.uuid4().hex}.pkl"
            pkl_path = os.path.join(OUTPUT_DIR, pkl_filename)
            
            with open(pkl_path, "wb") as f:
                pickle.dump(result_data, f)
            
            return pkl_path

        except Exception as e:
            logger.error(f"图片处理失败: {e}", exc_info=True)
            raise e

    def process_video_file(self, video_path: str, prompt: str) -> str:
        """
        处理本地视频文件，保存结果为 pkl
        """
        try:
            # 1. 启动 Session
            response = self.video_predictor.handle_request(
                request={
                    "type": "start_session",
                    "resource_path": video_path
                }
            )
            session_id = response["session_id"]
            
            # 2. 添加 Prompt (以第0帧为例)
            # 注意：实际视频处理可能需要遍历所有帧的 inference，这里演示简单调用
            prompt_resp = self.video_predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt
                }
            )
            
            # 3. 提取结果 (假设取第一帧结果或聚合结果)
            outputs = prompt_resp.get("outputs")
            if not outputs:
                raise ValueError("视频推理未返回有效 outputs")
            if isinstance(outputs, list):
                outputs = outputs[0]

            # 提取并转换数据
            # 视频接口返回的通常已经是 numpy 或者 list，确保安全转换
            def safe_numpy(data):
                if hasattr(data, 'cpu'):
                    if data.dtype == torch.bfloat16 or data.dtype == torch.float16:
                        return data.float().cpu().numpy()
                    return data.cpu().numpy()

                if isinstance(data, np.ndarray):
                    return data
                return np.array(data)

            result_data = {
                "type": "video_result",
                "prompt": prompt,
                "object_ids": safe_numpy(outputs.get("out_obj_ids", [])),
                "scores": safe_numpy(outputs.get("out_probs", [])),
                "boxes_xywh": safe_numpy(outputs.get("out_boxes_xywh", [])),
                # 视频 mask 数据量极大，如果显存/内存够可以放进去
                "masks": safe_numpy(outputs.get("out_binary_masks", [])) 
            }

            # 4. 保存 Pickle
            pkl_filename = f"sam3_vid_{uuid.uuid4().hex}.pkl"
            pkl_path = os.path.join(OUTPUT_DIR, pkl_filename)
            
            with open(pkl_path, "wb") as f:
                pickle.dump(result_data, f)
                
            return pkl_path

        except Exception as e:
            logger.error(f"视频处理失败: {e}", exc_info=True)
            raise e