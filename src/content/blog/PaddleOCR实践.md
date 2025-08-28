---
date: 2025-08-26
title: PaddleOCR实践
description: 将文档和图像转换为结构化、AI友好的数据，50000+ star
mermaid: true
mathjax: true
category:
  - 实用工具
tags:
  - 数据处理
  - OCR
  - 多模态大模型
  - 项目实践
ogImage: https://astro-yi.obs.cn-east-3.myhuaweicloud.com/avatar.png
---

# 项目介绍
项目：https://github.com/PaddlePaddle/PaddleOCR
使用教程：[使用教程 - PaddleOCR 文档](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PP-StructureV3.html#42)

![paddleOCR](/pic/paddleocr/paddleOCR.png)

**PP-OCRv5**：单模型支持五种文字类型（简中、繁中、英文、日文及拼音）
**PP-StructureV3**：将复杂PDF和文档图像智能转换为保留原始结构的Markdown文件和JSON文件，完美保持文档版式和层次结构

# 快速开始

## 在线体验
PP-OCRv5（通用文字识别）：https://aistudio.baidu.com/community/app/91660/webUI
**PP-StructureV3**（文档解析）：https://aistudio.baidu.com/community/app/518494/webUI
PP-ChatOCRv4（场景信息抽取）：https://aistudio.baidu.com/community/app/518493/webUI

## 本地安装
1. 创建虚拟环境
`conda create -n paddle_ocr_env python=3.9`

2. 确定cuda版本
`nvidia-smi`

3. 安装 PaddlePaddle 3.0
[开始使用_飞桨-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)
以 CUDA 12.6 为例：
` python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/`

4. 安装 paddleocr
`pip install paddleocr`
`pip install "paddlex[ocr]"`

5. 推理测试（命令行方式）
`pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False`

第一次运行时会下载PP-StructureV3模型的所有组件。PP-StructureV3 产线中包含以下7个模块，每个均可独立进行训练和推理，并包含多个模型
- [版面区域检测模块](https://www.paddleocr.ai/main/version3.x/module_usage/layout_detection.html)
- [通用OCR子产线](https://www.paddleocr.ai/main/version3.x/pipeline_usage/OCR.html)
- [文档图像预处理子产线](https://www.paddleocr.ai/main/version3.x/pipeline_usage/doc_preprocessor.html) （可选）
- [表格识别子产线](https://www.paddleocr.ai/main/version3.x/pipeline_usage/table_recognition_v2.html) （可选）
- [印章文本识别子产线](https://www.paddleocr.ai/main/version3.x/pipeline_usage/seal_recognition.html) （可选）
- [公式识别子产线](https://www.paddleocr.ai/main/version3.x/pipeline_usage/formula_recognition.html) （可选）
- [图表解析模块](https://www.paddleocr.ai/main/version3.x/module_usage/chart_parsing.html) （可选）

# 实践

## 单图识别
```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False)
output = pipeline.predict(
    input="./pp_structure_v3_demo.png",          
)
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```


## 长PDF 识别
保存为长md文档
```python
from pathlib import Path
from paddleocr import PPStructureV3
from PyPDF2 import PdfReader, PdfWriter

def process_large_pdf(input_file: str, output_path: Path):
    """
    处理大型 PDF 文件，使用流式方法逐页处理并保存结果。

    Args:
        input_file: 输入的 PDF 文件路径。
        output_path: 输出目录路径。
    """
    input_path = Path(input_file)
    output_path.mkdir(parents=True, exist_ok=True)
    # 初始化 PaddleOCR 管道
    pipeline = PPStructureV3(use_formula_recognition=False)
    # 初始化输出文件路径
    mkd_file_path = output_path / f"{input_path.stem}.md"
    # 逐页处理 PDF
    pdf_reader = PdfReader(input_path)
    num_pages = len(pdf_reader.pages)
    # 以追加模式打开 Markdown 文件
    with open(mkd_file_path, "a", encoding="utf-8") as md_file:
        for page_num in range(num_pages):
            print(f"正在处理第 {page_num + 1} 页...")
            # 创建一个临时的单页 PDF
            pdf_writer = PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_num])
            temp_pdf_path = output_path / f"temp_page_{page_num + 1}.pdf"
            with open(temp_pdf_path, "wb") as temp_file:
                pdf_writer.write(temp_file)

            # 使用 PPStructureV3 处理单页 PDF
            output = pipeline.predict(input=str(temp_pdf_path))

            # 清理临时文件
            temp_pdf_path.unlink()

            # 提取 Markdown 和图片信息
            if output:
                res = output[0]
                md_info = res.markdown
                md_text = pipeline.concatenate_markdown_pages([md_info])

                # 将 Markdown 文本追加到文件
                md_file.write(md_text)

                # 保存图片
                if "markdown_images" in md_info and md_info["markdown_images"]:
                    for path, image in md_info["markdown_images"].items():
                        image_file_path = output_path / path
                        image_file_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(image_file_path)

# --- 使用示例 ---
input_file = "./test_doc/china_tea.pdf"
output_path = Path("./output_stream")

process_large_pdf(input_file, output_path)
```

