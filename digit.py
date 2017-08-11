import timing
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('./input/train.csv') #now a pd.Dataframe that allows for methods to be applied.
images = labeled_images.iloc[0:5000,1:] #split img and lab for supervised learning.
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

test_images[test_images>0]=1
train_images[train_images>0]=1

i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='binary') #change out to 'grey' if that makes more sense.
plt.title(train_labels.iloc[i]) #show image #i

plt.hist(train_images.iloc[i])
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print clf.score(test_images,test_labels)

test_data=pd.read_csv('./input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)
