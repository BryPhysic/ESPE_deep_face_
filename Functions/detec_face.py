#%%
from deepface import DeepFace
objs = DeepFace.analyze(img_path = 'D:\Proyectos\ESPE_deep_face\imgs\mau.jpg', 
        actions = ['age', 'gender', 'race', 'emotion']
)

# %%
