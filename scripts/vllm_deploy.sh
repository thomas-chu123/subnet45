python3 -m venv vllm
source vllm/bin/activate
pm2 start --name "mistral" "vllm serve thesven/Mistral-7B-Instruct-v0.3-GPTQ  --max-model-len 16384 --quantization gptq --dtype half --gpu-memory-utilization 0.50 --port 8002"
pm2 start --name "qwen" "vllm serve Qwen/CodeQwen1.5-7B-AWQ  --max-model-len 8192 --quantization awq --dtype half --gpu-memory-utilization 0.50 --port 8000"
