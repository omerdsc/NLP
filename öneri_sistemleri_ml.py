# import lbraries

from surprise import Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split



# import veri seti
data=Dataset.load_builtin("ml-100k") # (kullanici id, film id, puan)



# train test split
trainset,testset=train_test_split(data, test_size=0.2)



# ML model create: KNN
model_options={
    "name":"cosine",
    "user_based":True # kullanıcılar arası benzerlik
    }


# train
model=KNNBasic(sim_options=model_options)
model.fit(trainset)

# test
prediciton=model.test(testset)
accuracy.rmse(prediciton)


# recommendation system  

def get_top_n(predictions, n=10):
    top_n={}
    
    for uid,iid,true_r,esy, _ in predictions:
        if not top_n.gey(uid):
            top_n[uid]=[]
        top_n[uid].append((iid,est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1],reverse=True)
        top_n[uid]=user_ratings[:n]
    
    return top_n

top_n=get_top_n(prediciton,5)

user_id="2"

print(f"top {n}, recommendation for user {user_id}")

for item_id, rating in top_n[user_id]:
    print(f"item id: {item_id}, score: {rating}")