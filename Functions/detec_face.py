#%%
from deepface import DeepFace
objs = DeepFace.analyze(img_path = "D:\Borrar\ESPE_deep_face\imgs/fotonoticia_20150331134913-15031252319_420.jpg", 
        actions = ['age', 'gender', 'race', 'emotion']
)

#%%
import json
datos = objs
print(datos)
"""
# %%
embedding_objs = DeepFace.represent(img_path = "D:\Borrar\ESPE_deep_face\imgs/fotonoticia_20150331134913-15031252319_420.jpg")
# %%
print(embedding_objs)
embedding = embedding_objs[0]["embedding"]
print(embedding)
# %%
"""