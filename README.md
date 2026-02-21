# Adaptive Diffusion Agent: Continual Learning & Concept Drift

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace Diffusers](https://img.shields.io/badge/Diffusers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-blue?style=for-the-badge)

A lightweight reinforcement/continual learning proof-of-concept that demonstrates how a Text-to-Image (T2I) model can actively adapt to shifting human preferences (concept drift) without suffering from catastrophic forgetting.

## The Problem: Concept Drift in Generative AI
When generative models are deployed, user preferences change over time. A model trained to generate "anime" might suddenly face a user base that prefers "photorealism." Traditional models fail here they either keep generating outdated styles, or they require a massive, expensive retraining phase that overwrites their previous knowledge (catastrophic forgetting).

This project solves that by implementing an **Active Adaptation Loop** using `segmind/tiny-sd`, OpenAI's `CLIP`, and dynamic LoRA (Low-Rank Adaptation) experts.

## Architecture & How It Works

The system is broken down into three main components:

1. **The Hidden Grader (Reward Model):** Uses `CLIP` to score generated images against a hidden text prompt (e.g., "dark cyberpunk"). It randomly switches this hidden preference to simulate unannounced **Concept Drift**.
2. **The Adaptive Router (Drift Detector):** Tracks the moving average of the Grader's scores. If the short-term reward drops significantly below the long-term baseline, it mathematically flags that a concept drift has occurred.
3. **The Continual Learning Agent (Generator):** When drift is detected, the agent freezes its outdated LoRA weights and spins up a fresh adapter (creating a Mixture-of-Experts style history). It enters an **Exploration Phase**, appending random style keywords to its prompt until the Grader gives a high score. It then saves those successful images to a Replay Buffer and fine-tunes the new LoRA adapter via MSE Diffusion Loss. 

## Installation

To run this locally or in a Colab environment, install the required dependencies:

```bash
pip install torch torchvision diffusers transformers accelerate peft
