# Multimodal Mixture-of-Experts Model from Fine-Tuned VLM
Developed multimodal MoE model from multiple open-source fine-tuned VLMs for generating detailed image captions and tagging by dynamically selecting specialized experts for diverse image-text understanding tasks. 

## Inference
The inputs require question and image URL then the final prompt combines the question with the Base64 string of the image then sends request to the fine-tuned MoE model hosted using AWS SageMaker AI via endpoint then returns the generated response.
```bash
Question -> "What is shown in this image, and what is the relevance for materials design? Include a discussion of multi-agent AI."
Image URL -> "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg"

Response:
The image shows a group of red ants (Solenopsis invicta) climbing over a vertical wooden post. The ants are using their long legs and antennae to navigate the rough surface of the wood, demonstrating their ability to adapt to different materials and environments. This behavior is relevant for materials design because it highlights the importance of considering the interactions between materials and living organisms, such as ants, when designing new materials.

Multi-agent AI (Artificial Intelligence) is a field of study that focuses on the development of AI systems that can work together with other AI systems to achieve a common goal. In the context of this image, multi-agent AI could be used to design materials that are more compatible with the natural behaviors of living organisms, such as ants, and that can adapt to different environments and conditions.

By studying the behavior of ants and other living organisms, researchers can gain insights into how materials can be designed to better interact with these organisms and to better mimic their natural behaviors. This can lead to the development of new materials that are more sustainable, efficient, and effective in a variety of applications.

In summary, the image of red ants climbing over a wooden post highlights the importance of considering the interactions between materials and living organisms when designing new materials, and the potential of multi-agent AI to help achieve this goal.
```
