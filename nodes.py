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
    Qwen GGUF 모델을 사용하여 프롬프트를 확장하고, 
    VRAM 관리 정책을 3단계로 선택할 수 있는 노드입니다.
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
                # ▼▼▼ [핵심] VRAM 관리 정책 (3가지 옵션) ▼▼▼
                "vram_policy": (
                    ["Always Unload (Safe)", "Unload VRAM (Keep RAM)", "Keep Loaded (Fast)"], 
                    {"default": "Always Unload (Safe)"}
                ),
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                "system_prompt": ("STRING", {
                    "default": """You are Z-Engineer, an expert prompt engineering AI specializing in the Z-Image Turbo architecture (S3-DiT). Your goal is to rewrite simple user inputs into high-fidelity, "Positive Constraint" prompts optimized for the Qwen-3 text encoder.

**CORE OPERATIONAL RULES:**
1. NO Negative Prompts: Use "Positive Constraints."
2. Natural Language Syntax: Use coherent, grammatical sentences.
3. Texture Density: Aggressively describe textures.
4. Spatial Precision: Use specific spatial prepositions.
5. Text Handling: Enclose text in double quotes.
6. Proper Anatomy: Explicitly state proper anatomy.
7. Camera & Lens: ALWAYS explicitly use "shot on" or "shot with".

**OUTPUT FORMAT:**
Return ONLY the enhanced prompt string.""", 
                    "multiline": True
                }),
                "prompt": ("STRING", {
                    "default": "Generate a detailed prompt.", 
                    "multiline": True
                }),
                "n_ctx": ("INT", {"default": 4096, "min": 2048, "max": 32768}),
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
        # 캐시에 있고, 모델 객체가 살아있으면 재사용
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

        print(f"[Qwen Node] Loading GGUF Model: {gguf_path}...")
        
        try:
            llm = Llama(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=True,
                logits_all=True
            )
            self._model_cache[cache_key] = llm
            return llm
        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            raise e

    def generate(self, gguf_name, seed, vram_policy, system_prompt, prompt, n_ctx, n_gpu_layers, max_new_tokens, temperature):
        # 1. 모델 로드
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

            # ▼▼▼ [핵심] 3단계 VRAM 관리 로직 ▼▼▼
            cache_key = f"{gguf_name}_{n_ctx}_{n_gpu_layers}"

            if vram_policy == "Keep Loaded (Fast)":
                # 아무것도 안 함 (메모리 유지)
                pass

            else:
                # "Always Unload" 또는 "Unload VRAM" 선택 시
                # llama-cpp 특성상 VRAM을 비우려면 객체를 삭제해야 함
                print(f"[Qwen Node] Unloading model based on policy: {vram_policy}")
                
                # 1. 캐시 목록에서 제거
                if cache_key in self._model_cache:
                    del self._model_cache[cache_key]
                
                # 2. 객체 삭제 (이 시점에서 VRAM 해제 시작)
                del llm
                
                # 3. 추가 정리 (정책에 따라 강도 조절)
                if vram_policy == "Always Unload (Safe)":
                    # 가장 강력한 청소: GC + CUDA 캐시 비우기
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    print("[Qwen Node] Full Cleanup Complete (RAM + VRAM).")
                else:
                    # Unload VRAM (Keep RAM)
                    # CUDA 캐시를 강제로 비우지 않음으로써 시스템 RAM에 잔여 데이터가 남을 가능성을 열어둠
                    # (OS 파일 캐싱 효과를 기대)
                    print("[Qwen Node] Object deleted (VRAM freed). OS Cache preserved.")

            return (output_text,)
            
        except Exception as e:
            return (f"Error: {e}",)

NODE_CLASS_MAPPINGS = {
    "QwenImageEngineer": Qwen3Engineer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEngineer": "Qwen3 Engineer"
}
