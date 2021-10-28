import matplotlib.pyplot as plt

import pickle

f = open('historycustomsgd0.01noaugnodrop.pck', 'rb')
historycust = pickle.load(f)
f.close()

f = open('historyvggAdam0.0001noaugnodrop.pck', 'rb')
historyvgg = pickle.load(f)
f.close()

'''
f = open('historyxcep3.pckl', 'rb')
historyresnet = pickle.load(f)
f.close()
'''

f = open('historyxcepSGD0.001noaugnodrop.pckl', 'rb')
historyxcep = pickle.load(f)
f.close()

l1 = []
for i in range(0,len(historycust["accuracy"])):
	l1.append(i)
	
l2 = []
for i in range(0,len(historyvgg["accuracy"])):
	l2.append(i)
	
'''	
l3 = []
for i in range(0,len(historyresnet[0])):
	l3.append(i)
'''
	
l4 = []
for i in range(0,len(historyxcep["accuracy"])):
	l4.append(i)			
	
	
fig, ax = plt.subplots()
ax.plot(l1,historycust["accuracy"], label='Custom', color='red',linewidth=1, marker='o', markersize=1, markeredgecolor='k')
ax.plot(l2,historyvgg["accuracy"], label='Vgg16', color='green',linewidth=1, marker='o', markersize=1, markeredgecolor='k')
ax.plot(l4,historyxcep["accuracy"], label='Xception', color='blue',linewidth=1, marker='o', markersize=1, markeredgecolor='k')

legend = ax.legend(loc='lower right', fontsize='medium')

plt.xlim(0)
plt.xlabel("Epochs")
plt.ylabel("Precisão")
plt.title("Sem $\it{data}$ $\it{augmentation}$, sem $\it{Dropout}$")
plt.show()

'''

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("Xception.h5")

score = reconstructed_model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
