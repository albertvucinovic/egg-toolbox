import argparse


def main():
    parser = argparse.ArgumentParser(
        description="egg-toolbox: Universal tool calling middleware for local LLMs"
    )
    parser.add_argument("model", help="Path to model (GGUF file or HF model ID)")
    parser.add_argument("--backend", choices=["tinygrad", "vllm", "sglang", "llamacpp"],
                        default="tinygrad", help="Backend to use (default: tinygrad)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--chat-template", help="Override chat template (path to .jinja file)")
    parser.add_argument("--tool-format",
                        choices=["hermes", "llama3", "mistral", "deepseek",
                                 "functionary", "command_r", "generic"],
                        help="Override auto-detected tool format")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Default sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Default max tokens")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Cap the model's context window at load time. "
                             "tinygrad preallocates the full KV cache at first "
                             "forward pass, so native 128K-context models will "
                             "OOM even a 24 GB GPU.  4096-16384 is usually "
                             "plenty for tool-call testing.")
    parser.add_argument("--keep-packed", action="store_true",
                        help="(tinygrad backend) Skip the .contiguous()+realize() "
                             "on weights at load.  tinygrad's scheduler fuses "
                             "dequantize into each matmul so packed GGUF weights "
                             "stay on device -- ~4x lower weight memory on Q4_0 "
                             "at the cost of slower generation.")
    parser.add_argument("--gpu-layers", type=int, default=-1,
                        help="GPU layers for llamacpp backend (default: -1 = all)")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel size for vllm/sglang (default: 1)")

    args = parser.parse_args()

    # Load backend
    backend = _create_backend(args)
    load_kwargs: dict = {}
    if args.context_length is not None:
        load_kwargs["max_context"] = args.context_length
    if args.keep_packed:
        load_kwargs["keep_packed"] = True
    backend.load_model(args.model, **load_kwargs)

    # Create orchestrator
    from .orchestrator import Orchestrator
    from .api.middleware import create_app
    orchestrator = Orchestrator(backend)

    # Create and run ASGI app
    import uvicorn
    app = create_app(orchestrator)
    uvicorn.run(app, host=args.host, port=args.port)


def _create_backend(args):
    if args.backend == "tinygrad":
        from .backends.tinygrad import TinygradBackend
        return TinygradBackend()
    elif args.backend == "vllm":
        raise NotImplementedError("vLLM backend not yet implemented (Phase 3)")
    elif args.backend == "sglang":
        raise NotImplementedError("SGLang backend not yet implemented (Phase 3)")
    elif args.backend == "llamacpp":
        raise NotImplementedError("llama-cpp-python backend not yet implemented (Phase 3)")
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
