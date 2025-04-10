# Active Learning-Based RLHF Implementation

## Overview
This repository implements Reinforcement Learning with Human Feedback (RLHF) using active learning. The implementation will focus on improving the efficiency of learning from multiple feedback types by selecting the most informative samples for human feedback.

## Features
- Active learning strategies for optimal feedback sample selection
- Support for multiple types of human feedback
- Improved learning efficiency through strategic sampling
- Reinforcement learning algorithms adapted for human feedback

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/active-rlhf.git
cd active-rlhf

# Create and activate conda environment
conda create -n active-rlhf python=3.8
conda activate active-rlhf

# Install dependencies
pip install -r requirements.txt

# Run the MC Dropout test file
python online_rlhf/mcdropout_test.py
```

If you want to run with the reward model frozen (as a sanity check):

```bash
# Edit the mcdropout_test.py file to initialize with frozen reward model
# Then run the test
python online_rlhf/mcdropout_test.py
```
