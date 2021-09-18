# KoGPT2novel
novel finetuned from skt/kogpt2-base-v2   
web demo:  
colab: https://colab.research.google.com/drive/1QYRKu3RI5mmIJcMDOa9NRbq_ETzcYJ7z?usp=sharing

# Result   
![result](doc/screenshot_1.png)    


# Required environment to run    
pip install torch==1.7.1+cu110   
pip install fastai==2.4   
pip install transformers==4.10.2    
pip install BentoML==0.13.1    

# Use in transformers
```python
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("ttop324/kogpt2novel")
model = AutoModelWithLMHead.from_pretrained("ttop324/kogpt2novel")

inputs = tokenizer.encode("안녕", return_tensors="pt")
output = model.generate(inputs, repetition_penalty=2.0,)
output = tokenizer.decode(output[0])
print(output)
```

# Run train  
train.ipynb  

# Run deploy    
deploy.ipynb  

# Acknowledgement and References      
- [KoGPT2](https://github.com/SKT-AI/KoGPT2)       
- [huggingface_sharing](https://huggingface.co/transformers/model_sharing.html)        
- [BentoML](https://github.com/bentoml/BentoML)       
- [BentoML_transformers](https://docs.bentoml.org/en/latest/frameworks.html#transformers)       
- [BentoML_versailles](https://github.com/getlegist/versailles)   
- [BentoML_iris-classifier](https://github.com/bentoml/gallery/tree/master/scikit-learn/iris-classifier)                 
- [kubernetes](https://kubernetes.io/)       
- [kubernautic](https://login.kubernautic.com/login)     
- [deployment_docker_kubernetes](https://course19.fast.ai/deployment_docker_kubernetes.html)   
- [kubernetes_2](https://bcho.tistory.com/1256)   
- [Kubernetes_Pod](https://honggg0801.tistory.com/44)    
- [typewriter-effect](https://css-tricks.com/snippets/css/typewriter-effect/)  
    

