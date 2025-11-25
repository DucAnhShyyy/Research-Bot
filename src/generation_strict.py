# src/generation_strict.py
"""
Strict generator that forces citation usage and performs basic grounding checks.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import re

MODEL = os.getenv('GENERATOR_MODEL', 'google/flan-t5-small')
DEVICE = int(os.getenv('DEVICE', -1))

class StrictGenerator:
    def __init__(self, model_name: str = MODEL, device: int = DEVICE):
        """
        Initialize seq2seq model for text2text generation.
        For CPU usage: device=-1 (default). For GPU, set device=0 (and ensure CUDA).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=device)

    # def convert_for_generator(self, merged_results):
    #     """
    #     Convert results into the format required by StrictGenerator.
    #     """
    #     converted = []
    #     for item in merged_results:
    #         payload = item.get("payload", {})
    #         converted.append({
    #             "meta": {
    #                 "source": payload.get("source", "unknown"),
    #                 "chunk_id": payload.get("chunk_id", 0),
    #                 "text": payload.get("text", "")
    #             }
    #         })
    #     return converted

    def build_context_block(self, retrieved):
        """
        Build context blocks with tags: [DOC:filename|chunk:N]
        retrieved: list of dicts with 'meta' containing source, chunk_id, text
        """
        parts = []
        for r in retrieved:
            meta = r.get('meta') or r
            src = meta.get('source') or meta.get('doc_id') or 'unknown'
            chunk_id = meta.get('chunk_id', meta.get('chunk', 0))
            tag = f"[DOC:{src}|chunk:{chunk_id}]"
            parts.append(f"{tag}\n{meta.get('text')}\n")
        return "\n---\n".join(parts)

    def build_prompt(self, question, retrieved):
        """
        Build instruction prompt that constrains model to use only context.
        """
        ctx = self.build_context_block(retrieved)
        prompt = (
            "You are an academic assistant. ANSWER using ONLY the CONTEXT blocks below. "
            "For every factual claim include an inline citation exactly like [DOC:filename.pdf|chunk:3]. "
            "If the information is not present, reply: 'Không đủ thông tin trong tài liệu đã cung cấp.'\n\n"
            f"CONTEXT:\n{ctx}\n\nQUESTION:\n{question}\n\nANSWER:\n"
        )
        return prompt

    def generate(self, question, retrieved, max_length=512):
        """
        Generate answer and perform a simple citation check.
        """
        prompt = self.build_prompt(question, retrieved)
        out = self.pipe(prompt, max_length=max_length, do_sample=False)[0]['generated_text']
        # verify citations existence
        citations = re.findall(r"\[DOC:([^\]]+)\]", out)
        tags = {f"{r['meta'].get('source')}|chunk:{r['meta'].get('chunk_id')}" for r in retrieved}
        invalid = [c for c in citations if c not in tags]
        if citations and invalid:
            out += "\n\n[WARNING] Một số citation không xuất hiện trong ngữ cảnh đã truy xuất; kiểm tra kết quả cẩn thận."
        return out