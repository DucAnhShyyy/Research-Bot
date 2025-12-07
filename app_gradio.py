# # src/app_gradio.py
# """
# Gradio demo: upload PDFs, index them, and chat using RAG pipeline.
# """
# import gradio as gr
# from pathlib import Path
# from .ingest import index_folder
# from .retriever_hybrid import HybridRetriever
# from .generation_strict import StrictGenerator

# COLLECTION = 'papers'

# # Initialize once (loads models)
# retriever = HybridRetriever(qdrant_collection=COLLECTION)
# generator = StrictGenerator()

# def index_uploaded(files):
#     """
#     files: list of tempfile objects from gradio
#     Saves them into sample_data and runs index_folder.
#     """
#     sample_dir = Path('sample_data')
#     sample_dir.mkdir(parents=True, exist_ok=True)
#     saved = 0
#     for f in files:
#         dest = sample_dir / Path(f.name)
#         with open(dest, 'wb') as out:
#             out.write(f.read())
#         saved += 1
#     index_folder(str(sample_dir), collection_name=COLLECTION)
#     return f"Indexed {saved} files"

# def answer_question(query: str):
#     """
#     Retrieve and generate answer for a given query.
#     """
#     retrieved = retriever.merge_and_rerank(query, top_k=5)
#     answer = generator.generate(query, retrieved)
#     sources = '\n'.join([f"- {r['meta'].get('source')} (chunk {r['meta'].get('chunk_id')})" for r in retrieved])
#     return answer + '\n\nRetrieved sources:\n' + sources

# def build_ui():
#     with gr.Blocks() as demo:
#         gr.Markdown("# Research Assistant Chatbot — Demo")
#         with gr.Row():
#             upload = gr.File(file_count='multiple', label='Upload PDFs to index')
#             btn_index = gr.Button('Index uploaded PDFs')
#         status = gr.Textbox(label='Index status')
#         btn_index.click(fn=index_uploaded, inputs=[upload], outputs=[status])

#         query = gr.Textbox(label='Ask a question', placeholder='E.g. What method does paper X use?')
#         btn_ask = gr.Button('Ask')
#         output = gr.Textbox(label='Answer', lines=12)
#         btn_ask.click(fn=answer_question, inputs=[query], outputs=[output])
#     return demo

# if __name__ == '__main__':
#     demo = build_ui()
#     demo.launch(server_name='0.0.0.0', server_port=7860)

# src/app_gradio.py
# """
# Gradio demo: upload PDFs, index them, and chat using RAG pipeline.
# """

# import gradio as gr
# from pathlib import Path
# import shutil

# from .ingest import index_folder
# from .retriever_hybrid import HybridRetriever
# from .generation_strict import StrictGenerator

# COLLECTION = 'papers'

# # Initialize once (loads models)
# retriever = HybridRetriever(qdrant_collection=COLLECTION)
# generator = StrictGenerator()


# def index_uploaded(files):
#     """
#     files: list of NamedString objects from gradio
#     Saves them into sample_data and runs index_folder.
#     """

#     sample_dir = Path("sample_data")
#     sample_dir.mkdir(parents=True, exist_ok=True)

#     saved = 0
#     for f in files:

#         # f.name = đường dẫn file tạm Gradio lưu
#         src_path = Path(f.name)

#         # nơi bạn muốn copy tới
#         dest_path = sample_dir / src_path.name

#         # copy đúng chuẩn (thay cho f.read())
#         shutil.copy(src_path, dest_path)

#         saved += 1

#     # index thư mục sample_data
#     index_folder(str(sample_dir), collection_name=COLLECTION)

#     return f"Indexed {saved} files"


# def answer_question(query: str):
#     """
#     Retrieve and generate answer for a given query.
#     """
#     raw = retriever.merge_and_rerank(query, top_k=5)
#     retrieved = retriever.convert_for_generator(raw)   # 🔥 MUST FIX
#     answer = generator.generate(query, retrieved)

#     sources = '\n'.join([
#         f"- {r['meta'].get('source')} (chunk {r['meta'].get('chunk_id')})"
#         for r in retrieved
#     ])
#     return answer + '\n\nRetrieved sources:\n' + sources


# def build_ui():
#     with gr.Blocks() as demo:
#         gr.Markdown("# Research Assistant Chatbot — Demo")

#         with gr.Row():
#             upload = gr.File(file_count="multiple", label="Upload PDFs to index")
#             btn_index = gr.Button("Index uploaded PDFs")

#         status = gr.Textbox(label="Index status")
#         btn_index.click(fn=index_uploaded, inputs=[upload], outputs=[status])

#         query = gr.Textbox(label="Ask a question",
#                            placeholder="E.g. What method does paper X use?") #How does Self attention mechanism works?
#         btn_ask = gr.Button("Ask")
#         output = gr.Textbox(label="Answer", lines=12)

#         btn_ask.click(fn=answer_question, inputs=[query], outputs=[output])

#     return demo


# if __name__ == "__main__":
#     demo = build_ui()
#     demo.launch(server_name="0.0.0.0", server_port=7860)

# src/app_gradio.py
import gradio as gr
from pathlib import Path
import shutil
import logging

# Import các module của bạn
from .ingest import index_folder
from .retriever_hybrid import HybridRetriever
from .generation_strict import StrictGenerator

# Setup logging để debug dễ hơn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION = 'papers'

# Initialize models
try:
    retriever = HybridRetriever(qdrant_collection=COLLECTION)
    generator = StrictGenerator()
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")

def index_uploaded(files):
    if not files:
        return "No files uploaded."
        
    sample_dir = Path("sample_data")
    sample_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        src_path = Path(f.name)
        dest_path = sample_dir / src_path.name
        try:
            shutil.copy(src_path, dest_path)
            saved += 1
        except Exception as e:
            logger.error(f"Error copying file {src_path}: {e}")

    # Re-index
    index_folder(str(sample_dir), collection_name=COLLECTION)
    return f"Indexed {saved} files successfully."

def answer_question(query: str):
    if not query.strip():
        return "Vui lòng nhập câu hỏi."

    # 1. Retrieve
    raw = retriever.merge_and_rerank(query, top_k=5)
    
    if not raw:
        return "Không tìm thấy tài liệu liên quan. Hãy thử index lại PDF hoặc hỏi câu khác."

    # 2. Convert format
    retrieved = retriever.convert_for_generator(raw)

    # 3. Generate
    try:
        answer = generator.generate(query, retrieved)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return f"Lỗi khi sinh câu trả lời: {e}"

    # 4. Format Sources
    sources_text = '\n'.join([
        f"- {r['meta'].get('source')} (chunk {r['meta'].get('chunk_id')})"
        for r in retrieved
    ])
    
    return f"{answer}\n\n----------------\nRetrieved sources:\n{sources_text}"

def build_ui():
    with gr.Blocks(title="Research Assistant") as demo:
        gr.Markdown("# 🤖 Research Assistant Chatbot")

        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(file_count="multiple", label="Upload PDFs")
                btn_index = gr.Button("Index Data", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                chatbot_output = gr.Textbox(label="Answer", lines=10, interactive=False)
                query = gr.Textbox(label="Your Question", placeholder="Enter your question here...")
                btn_ask = gr.Button("Ask", variant="primary")

        # Events
        btn_index.click(fn=index_uploaded, inputs=[upload], outputs=[status])
        btn_ask.click(fn=answer_question, inputs=[query], outputs=[chatbot_output])
        query.submit(fn=answer_question, inputs=[query], outputs=[chatbot_output])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)