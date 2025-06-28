"""
    Problem tanimi: Oneri sistemi
    user (kullanici) - item (urunler) - rating (puanlar)

"""

# import libraries
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot,Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# veri serini olustur
user_id=np.array([0,1,2,3,4,0,1,2,3,4])
item_id=np.array([0,1,2,3,4,1,2,3,4,5])
ratings=np.array([5,4,3,2,1,4,5,3,2,1])

# train test split

user_id_train,user_id_test,item_id_train,item_id_test,ratings_train,ratings_test=train_test_split(user_id, item_id,ratings,test_size=0.2,random_state=21)

# create neural Network

def create_model(num_users,num_items,embedding_dim):
    user_input=Input(shape=(1,),name="user")
    item_input=Input(shape=(1,),name="item")
    
    user_embedding=Embedding(input_dim=num_users,output_dim=embedding_dim,name="user_embedding")(user_input)
    item_embedding=Embedding(input_dim=num_items,output_dim=embedding_dim,name="item_embedding")(item_input)
    
    # vektörleri düzleşitr
    user_vecs=Flatten()(user_embedding)
    item_vecs=Flatten()(item_embedding)
    
    dot_product= Dot(axes=1)([user_vecs,item_vecs])
    output=Dense(1)(dot_product)
    
    model=Model(inputs=[user_input,item_input],outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    
    return model

# train and test

num_users=5
num_items=6
embedding_dim=8
model=create_model(num_users,num_items,embedding_dim)
model.fit([user_id_train,item_id_train], ratings_train,epochs=10,verbose=1,validation_split=0.1)

loss=model.evaluate([user_id_test,item_id_test],ratings_test)
print(f"test: {loss}")

user_id