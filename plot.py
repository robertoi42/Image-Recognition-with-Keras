import matplotlib.pyplot as plt

import pickle

#historycustomsgd0.01noaugnodrop.pck #anomalia
#historycustomsgd0.01augnodropthisone.pck
#historycustomsgd0.01noaugdrop.pck #anomalia
#historycustomsgd0.01augdrop4.pck
f = open('historyCustomadam0.001nothingtest8.pck', 'rb')
historycust = pickle.load(f)
f.close()

#historyvggAdam0.0001noaugnodrop.pck
#historyvggAdam0.0001augmodropout2.pck
#historyvggAdam0.0001noaugdropout2.pck  #anomalia
#historyvggAdam0.001augdropout5.pck
f = open('historyCustomadam0.001nothingtest7.pck', 'rb')
historyvgg = pickle.load(f)
f.close()

#historyresnetadam0.00001noaugdrop.pckl
#historyrespadam0aug.000013.pckl
f = open('historyCustomadam0.001nothingtest6.pck', 'rb')
historyresnet = pickle.load(f)
f.close()

#historyxcepSGD0.001noaugnodrop.pckl
#historyxcepSGD0.01augnodropthisone.pckl
#historyxcepSGD0.01noaugdrop.pckl
#historyxcepSGD0.01augdropthisone.pckl
f = open('historyCustomadam0.001nothingtest5.pck', 'rb')
historyxcep = pickle.load(f)
f.close()

l1 = []
for i in range(0,len(historycust["accuracy"])):
	l1.append(i)
print(len(l1))
	
l2 = []
for i in range(0,len(historyvgg["accuracy"])):
	l2.append(i)
print(len(l2))
		
l3 = []
for i in range(0,len(historyresnet["accuracy"])):
	l3.append(i)
print(len(l3))
	
l4 = []
for i in range(0,len(historyxcep["accuracy"])):
	l4.append(i)			
print(len(l4))
	
fig, ax = plt.subplots()
'''
#ax.plot(l1,historycust["val_accuracy"], label='$\it{Data}$ $\it{augmentation}$/ $\it{Dropout}$', color='red',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')
#ax.plot(l2,historyvgg["val_accuracy"], label='$\it{Data}$ $\it{augmentation}$', color='green',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')
ax.plot(l3,historyresnet["val_accuracy"], label='$\it{Dropout}$', color='orange',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')
ax.plot(l4,historyxcep["val_accuracy"], label='Nada', color='blue',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')

'''
#ax.plot(l1,historycust["val_loss"], label='$\it{Data}$ $\it{augmentation}$/ $\it{Dropout}$', color='red',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')
#ax.plot(l2,historyvgg["val_loss"], label='$\it{Data}$ $\it{augmentation}$', color='green',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')
ax.plot(l3,historyresnet["val_loss"], label='$\it{Dropout}$', color='orange',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')
ax.plot(l4,historyxcep["val_loss"], label='Nada', color='blue',linewidth=1, marker='o', markersize=0.5, markeredgecolor='k')


legend = ax.legend(loc='lower right', fontsize='medium')
#legend = ax.legend(loc='upper right', fontsize='medium')

plt.xlim(0)
#plt.ylim(0,3)
plt.xlabel("Epochs")
plt.ylabel("Precisão")
plt.ylabel("Loss")
plt.title("$\it{Overfitting}$ (Rede customizada)")
plt.show()

'''

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("Xception.h5")

score = reconstructed_model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
