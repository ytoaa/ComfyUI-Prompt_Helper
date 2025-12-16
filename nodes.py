import folder_paths
import os
import gc
import torch

# llama_cpp 라이브러리 로드 확인
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("\033[31m[Qwen Node Error] llama-cpp-python 未安装！无法加载 GGUF 模型。请运行: pip install llama-cpp-python\033[0m")

class Qwen3Engineer:
    """
    Qwen GGUF 모델을 mmap(Memory-mapped) 방식으로 로드하여
    I/O 속도를 극한으로 끌어올리고 VRAM을 효율적으로 관리하는 노드입니다.
    """
    _model_cache = {}
    
    @classmethod
    def INPUT_TYPES(s):
        file_list = []
        text_encoder_paths = folder_paths.get_folder_paths("text_encoders")
        for path in text_encoder_paths:
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.gguf'):
                            rel_path = os.path.relpath(os.path.join(root, file), path)
                            file_list.append(rel_path.replace(os.sep, '/'))
        
        if not file_list:
            file_list = ["Put_GGUF_in_models/text_encoders_folder.gguf"]

        return {
            "required": {
                "gguf_name": (file_list, ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # ▼▼▼ VRAM 관리 정책 ▼▼▼
                "vram_policy": (
                    ["Always Unload (Safe)", "Unload VRAM (Keep RAM)", "Keep Loaded (Fast)"], 
                    {"default": "Always Unload (Safe)"}
                ),
                "system_prompt": ("STRING", {
                    "default": """You are Z-Engineer... (생략)...""", 
                    "multiline": True
                }),
                "prompt": ("STRING", {
                    "default": "Generate a detailed prompt.", 
                    "multiline": True
                }),
                "n_ctx": ("INT", {"default": 2048, "min": 64, "max": 32768, "tooltip": "프롬프트 작업용이면 1024~2048이면 충분합니다."}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100}),
                "max_new_tokens": ("INT", {"default": 512, "min": 16, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "QwenTextEngineer"

    def load_gguf(self, gguf_name, n_ctx, n_gpu_layers):
        cache_key = f"{gguf_name}_{n_ctx}_{n_gpu_layers}"
        # 캐시된 모델이 있고 유효하면 재사용
        if cache_key in self._model_cache and self._model_cache[cache_key] is not None:
            return self._model_cache[cache_key]

        if not LLAMA_CPP_AVAILABLE:
            raise Exception("缺少 llama-cpp-python 库。")

        gguf_path = folder_paths.get_full_path("text_encoders", gguf_name)
        if not gguf_path:
            text_encoder_paths = folder_paths.get_folder_paths("text_encoders")
            for path in text_encoder_paths:
                if os.path.exists(path):
                    full_path = os.path.join(path, gguf_name)
                    if os.path.isfile(full_path):
                        gguf_path = full_path
                        break
        
        if not gguf_path or not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"File not found: {gguf_name}")

        print(f"[Qwen Node] Loading GGUF Model via mmap: {gguf_path}...")
        
        try:
            llm = Llama(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=True,
                # ▼▼▼ [최적화 핵심] mmap 명시적 활성화 ▼▼▼
                use_mmap=True,   # 디스크->RAM 복사 없이 직접 매핑 (Zero-Copy)
                use_mlock=False, # RAM 고정 방지 (시스템 유연성 확보)
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                logits_all=True
            )
            self._model_cache[cache_key] = llm
            return llm
        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            raise e

    def generate(self, gguf_name, seed, vram_policy, system_prompt, prompt, n_ctx, n_gpu_layers, max_new_tokens, temperature):
        # 1. 모델 로드 (mmap 덕분에 두 번째부터는 즉시 로드됨)
        llm = self.load_gguf(gguf_name, n_ctx, n_gpu_layers)
        
        full_system_prompt = system_prompt + "\n\nIMPORTANT: Do NOT repeat the input. START DIRECTLY with the enhanced prompt."
        
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed
            )
            output_text = response["choices"][0]["message"]["content"]

            # 태그 제거
            if "### Response:" in output_text:
                output_text = output_text.split("### Response:")[-1].strip()
            elif "Response:" in output_text:
                output_text = output_text.split("Response:")[-1].strip()
            if "### Input:" in output_text:
                 parts = output_text.split("### Input:")
                 if len(parts) > 1: pass
            output_text = output_text.strip()

            # ▼▼▼ VRAM 관리 로직 ▼▼▼
            cache_key = f"{gguf_name}_{n_ctx}_{n_gpu_layers}"

            if vram_policy == "Keep Loaded (Fast)":
                pass
            else:
                # mmap을 썼기 때문에 del을 해도 데이터는 OS 페이지 캐시에 매핑된 상태로 남음
                # 재로딩 시 디스크 I/O가 거의 발생하지 않음
                if cache_key in self._model_cache:
                    del self._model_cache[cache_key]
                del llm
                
                if vram_policy == "Always Unload (Safe)":
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    print("[Qwen Node] VRAM Released. (Safe Mode)")
                else:
                    print("[Qwen Node] VRAM Released. (Keep RAM Mode)")

            return (output_text,)
            
        except Exception as e:
            return (f"Error: {e}",)

NODE_CLASS_MAPPINGS = {
    "QwenImageEngineer": Qwen3Engineer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEngineer": "Qwen3 Engineer (mmap optimized)"
}
