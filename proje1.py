import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk 

data = pd.read_excel("SATILIK_EV1.xlsx")
print(data.head(5))
print(data.describe())
print(data.shape)
print(data.isnull().any())
data = data.drop(columns="Unnamed: 0")
print(data.value_counts().mean)
plt.figure(figsize = (12,8))
cor = data.corr()
sns.heatmap(cor,annot=True , cmap=plt.cm.Reds)
plt.show()

# TÜM DEĞİŞKENLERİN BİRBİRLERİ İLE OLAN SERPİLME DİAGRAMLARINA BAKALIM.
sns.pairplot(data)
plt.show()

# HEDEF DEĞİŞKENİ OLAN FİYAT DEĞİŞKENİNİN DAĞILIMINA BAKALIM
sns.displot(data["Fiyat"])
sns.displot(data["Oda_Sayısı"])
sns.displot(data["Net_m2"])
plt.show()

X = data.drop(columns="Fiyat",axis=1)
y = data["Fiyat"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.35,random_state=101)

model = LinearRegression()
model.fit(X_train,y_train)

# y = ax1 + a2x2 + a3x3 + ...... an+xn + b
# b = SABİT DEĞER 
# anxn = ÖZNİTELİK KATSAYILARI

print(f"\nSabit Değer:{model.intercept_}\nÖzniteliklerin Kat sayıları{model.coef_}:")
coeff_df = pd.DataFrame(model.coef_,X.columns,columns=["Öznitelik_Katsayıları"])
print(coeff_df)

y_pred = model.predict(X_test)
for i,prediction in enumerate(y_pred):
    print(f"Tahmin Edilen Ev Fiyati:{prediction},\nGerçek Ev Fiyati:{y[i]}\n")

from sklearn.metrics import mean_absolute_error , r2_score,accuracy_score
print("Mean_absolute Score:\n",mean_absolute_error(y_test,y_pred))
print("R2 Score:\n",r2_score(y_test,y_pred))


plt.scatter(y_test,y_pred)
plt.show()

oda_s = 3
m2 = 200
katı = 10 
yas = 1

print("Tahmin Edilen Değeri : ",model.predict([[oda_s,m2,katı,yas]]))

# İNTERFACE ARAYÜZ OLUŞTURALIM 

root = tk.Tk()
canvas1 = tk.Canvas(root,width=2000 , height=2000)
canvas1.pack()

label1 = tk.Label(root,text ="Oda Sayısı: ")
canvas1.create_window(800,900,window=label1)
entry1 = tk.Entry(root)
canvas1.create_window(1000,900,window=entry1)

label2 = tk.Label(root,text ="Net_m2: ")
canvas1.create_window(800,1000,window=label2)
entry2 = tk.Entry(root)
canvas1.create_window(1000,1000,window=entry2)

label3 = tk.Label(root,text ="Katı: ")
canvas1.create_window(800,1100,window=label3)
entry3 = tk.Entry(root)
canvas1.create_window(1000,1100,window=entry3)

label4 = tk.Label(root,text ="Yaşı: ")
canvas1.create_window(800,1200,window=label4)
entry4 = tk.Entry(root)
canvas1.create_window(1000,1200,window=entry4)


def values():
    global Oda_Sayısı
    Oda_Sayısı = float(entry1.get())
    global Net_m2
    Net_m2 = float(entry2.get())
    global katı
    katı = float(entry3.get())
    global yasi
    yasi = float(entry4.get())

    Prediction_result = ("Evin Tahmin edilen fiyati:",1000*int(model.predict([[Oda_Sayısı,Net_m2,katı,yasi]])))
    label_Prediction = tk.Label(root,text = Prediction_result,bg = "lawngreen")
    canvas1.create_window(1000,1400,window=label_Prediction)


button1 = tk.Button(root,text = "Evin Tahmin Fiyatını Hesapla",command=values,bg = "orange")
canvas1.create_window(1000,1300,window=button1)
root.mainloop()

