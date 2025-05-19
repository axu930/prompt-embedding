An attempt to use https://arxiv.org/pdf/2501.06730 to get better embedding for RAG retrieval https://arxiv.org/pdf/2308.03281

Maybe apply the 3 part training from https://sander.ai/2025/04/15/latents.html with the decoder model running on jacobi decoding https://arxiv.org/pdf/2403.00835

Models to try:
Qwen 3 0.6B/1.7B
Llama 3.2 1B

Approx. pipeline:
1) Start with pretrained, instruction tuned LM
2) Train ICAE style (https://arxiv.org/pdf/2307.06945) encoder, using https://arxiv.org/pdf/2501.06730
3) To train a retieval model .. 
4) To train 