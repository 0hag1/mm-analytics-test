import MeCab
from pymongo import MongoClient

m = MeCab.Tagger('-Owakati')

client = MongoClient('localhost', 27017)
db = client.scraped
col = db.amazon_reviews

reviews = col.find()

for review in reviews:
    train = "__label__" + str(int(review['evaluation'])) + ", " + m.parse(review['review']).strip()
    print(train)

    
