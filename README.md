# APSEIPIK: Adaptive Prompt-Based Semantic Embedding with Inspire Potential of Implicit Knowledge for Cross-Modal Retrieval

This work, APSEIPIK, has been accepted by AAAI 2025🎉🎉🎉

This document provides the official code implementation of the APSEIPIK method.

## Implementation Details

To improve training efficiency and flexibility, we have optimized the implementation process of APSEIPIK. Specifically, we have decoupled the image description generation process of BLIP2. This design makes it unnecessary to repeatedly generate descriptions during the model training phase, thereby significantly improving overall training efficiency.

### Generating Multi-perspective Image Descriptions with BLIP2

We provide specific code to help you batch-generate multi-view image descriptions for the COCO and Flickr30k datasets using BLIP2. These pre-generated descriptions can be directly used for training the APSEIPIK model.

* **Dataset Description Generation:**:
  
        
        python Use_BLIP2_gen_text/generate_text_from_BLIP2.py
     

### Flickr30k Trained Weights:

We provide here the APSEIPIK model weights trained on the Flickr30k dataset, for convenient subsequent evaluation and research.

  * Flickr30k pre-trained weights: `[链接到您的权重文件]`
    * You can load the weights as follows:
        ```python
        model.load_state_dict(torch.load('path/to/your/flickr30k_weights.pth'))
        ```

## Core Dependencies

```bash
 pytorch-lightning         1.3.2 
 pytorch                   2.0.1
 python                    3.9.19
