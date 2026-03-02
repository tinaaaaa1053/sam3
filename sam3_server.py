import os
import shutil
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sam3_service import SAM3Service, TEMP_DIR, OUTPUT_DIR

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM 3 Local File Service")

# 启动时加载模型
@app.on_event("startup")
def startup_event():
    SAM3Service()

# 清理任务：发送完文件后删除临时文件
def cleanup_files(file_paths: list):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"已清理临时文件: {path}")
        except Exception as e:
            logger.warning(f"清理文件失败 {path}: {e}")

@app.post("/sam3/upload/image")
async def upload_and_process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    上传图片 -> SAM3处理 -> 下载 pkl 结果
    """
    # 1. 保存上传的文件
    file_ext = file.filename.split('.')[-1]
    temp_filename = f"upload_{uuid.uuid4().hex}.{file_ext}"
    temp_input_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"接收图片: {file.filename}, Prompt: {prompt}")

        # 2. 调用服务处理
        service = SAM3Service()
        pkl_output_path = service.process_image_file(temp_input_path, prompt)

        # 3. 设置后台清理任务 (清理 输入图片 和 输出pkl)
        background_tasks.add_task(cleanup_files, [temp_input_path, pkl_output_path])

        # 4. 返回文件下载
        # filename 参数决定了用户下载时看到的文件名
        return FileResponse(
            path=pkl_output_path, 
            filename=f"sam3_results_{file.filename}.pkl",
            media_type='application/octet-stream'
        )

    except Exception as e:
        # 如果出错，也要尝试清理输入文件
        cleanup_files([temp_input_path])
        logger.error(f"处理请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sam3/upload/video")
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    上传视频 -> SAM3处理 -> 下载 pkl 结果
    """
    file_ext = file.filename.split('.')[-1]
    temp_filename = f"upload_{uuid.uuid4().hex}.{file_ext}"
    temp_input_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"接收视频: {file.filename}, Prompt: {prompt}")

        service = SAM3Service()
        pkl_output_path = service.process_video_file(temp_input_path, prompt)

        background_tasks.add_task(cleanup_files, [temp_input_path, pkl_output_path])

        return FileResponse(
            path=pkl_output_path, 
            filename=f"sam3_results_{file.filename}.pkl",
            media_type='application/octet-stream'
        )

    except Exception as e:
        cleanup_files([temp_input_path])
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 监听 8001 端口
    uvicorn.run(app, host="0.0.0.0", port=8001)