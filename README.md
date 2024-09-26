# ET_Robot Project

This repository contains the necessary scripts, datasets, and configurations for training, fine-tuning, and evaluating robot behavior models using various tools and datasets.

## Directory Structure

- **box_evaluation/**:  
  Folder for evaluating performance and accuracy of box-related tasks or object manipulation.

- **phi-3-mini-LoRA/**:  
  Directory for the low-rank adaptation (LoRA) models used in fine-tuning the robot models with a focus on performance and efficiency.

- **prompts_folder/**:  
  Includes various prompts used during training and fine-tuning to guide the model's behavior.

- **robosuite/**:  
  This folder includes the Robosuite framework for creating simulation environments and tasks for the robot. Robosuite is likely used here for simulating the tasks.

## Scripts and Files

- **data_generation.py**:  
  Script for generating finetuning data to train phi-3-mini models.

- **data_prep_stack_three.txt** & **data_prep_stack_two.txt**:  
  Configuration or instruction files for preparing data specific to the stacking tasks. Likely contains preprocessing steps for different types of data stacks.

- **finetuning.py**:  
  Script for fine-tuning pre-trained models on specific tasks.

- **main_retrieval_GPT-4o.py**:  
  Script for retrieving information and generating tasks using GPT-4-based models in conjunction with the robot’s system.

- **stack_multiple_record.py**:  
  Script for managing and recording results from multiple stack trials or tasks.

- **updated_val_data.json**:  
  JSON file containing validation data used during model training and evaluation phases.

## How to Run

1. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt

2. Install necessary dependencies from robosuite: https://robosuite.ai/
