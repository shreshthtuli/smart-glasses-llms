# Run mt_bench
python3 gen_model_answer.py --model-path Locutusque/TinyMistral-248M-v2.5 --model-id TinyMistral-248M-v2.5 --num-gpus-total 12
python3 gen_model_answer.py --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-id TinyLlama-1.1B-Chat-v1.0 --num-gpus-total 10
python3 gen_model_answer.py --model-path Writer/palmyra-small --model-id palmyra-small --num-gpus-total 10
python3 gen_model_answer.py --model-path Writer/palmyra-3B --model-id palmyra-3B --num-gpus-total 3
python3 gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --num-gpus-total 2
python3 gen_model_answer.py --model-path google/gemma-7b --model-id gemma-7b --num-gpus-total 2
python3 gen_model_answer.py --model-path google/gemma-2b --model-id gemma-2b --num-gpus-total 10
python3 gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.2 --model-id Mistral-7B-Instruct-v0.2 --num-gpus-total 2
python3 gen_model_answer.py --model-path openchat/openchat-3.5-0106 --model-id openchat-3.5-0106 --num-gpus-total 2
python3 gen_model_answer.py --model-path microsoft/phi-2 --model-id phi-2 --num-gpus-total 10
python3 gen_model_answer.py --model-path microsoft/phi-1_5 --model-id phi-1_5 --num-gpus-total 10

# Run ihap_bench
python3 gen_model_answer.py --model-path Locutusque/TinyMistral-248M-v2.5 --model-id TinyMistral-248M-v2.5 --num-gpus-total 12 --bench-name ihap_bench
python3 gen_model_answer.py --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-id TinyLlama-1.1B-Chat-v1.0 --num-gpus-total 10 --bench-name ihap_bench
python3 gen_model_answer.py --model-path Writer/palmyra-small --model-id palmyra-small --num-gpus-total 10 --bench-name ihap_bench
python3 gen_model_answer.py --model-path Writer/palmyra-3B --model-id palmyra-3B --num-gpus-total 3 --bench-name ihap_bench
python3 gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --num-gpus-total 2 --bench-name ihap_bench
python3 gen_model_answer.py --model-path google/gemma-7b --model-id gemma-7b --num-gpus-total 2 --bench-name ihap_bench
python3 gen_model_answer.py --model-path google/gemma-2b --model-id gemma-2b --num-gpus-total 10 --bench-name ihap_bench
python3 gen_model_answer.py --model-path mistralai/Mistral-7B-Instruct-v0.2 --model-id Mistral-7B-Instruct-v0.2 --num-gpus-total 2 --bench-name ihap_bench
python3 gen_model_answer.py --model-path openchat/openchat-3.5-0106 --model-id openchat-3.5-0106 --num-gpus-total 2 --bench-name ihap_bench
python3 gen_model_answer.py --model-path microsoft/phi-2 --model-id phi-2 --num-gpus-total 10 --bench-name ihap_bench
python3 gen_model_answer.py --model-path microsoft/phi-1_5 --model-id phi-1_5 --num-gpus-total 10 --bench-name ihap_bench