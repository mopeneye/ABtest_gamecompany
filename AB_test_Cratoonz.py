import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import ttest_ind

pd.pandas.set_option('display.max_columns', None)

"""
A/B Test Sonuçlarının Analiz Edilmesi ve Sunulması


... Oyun firması, Cat Heroes: Puzzle Adventure oyununda bulunan Max’s Tree Tower isimli etkinlik 38. seviyeyi geçince açılmaktadır. 
Oyuncuları dönüştürme (conversion) konusunda son derece güçlü olan bu kart seçme oyununda, AB testler yapılarak daha çok gelir elde edilebileceğini düşünülmektedir.

Bu kapsamda satış tiplerine ilişkin olarak, hali hazırda mevcut olan tek ürün satışı yerine (Sadece Mavi Kristal Satan Yapı), sepet ürün satışı yöntemine
geçilirse gelirlerin ne olacağı merak edilmektedir.

Gelişmeler ışığında, yeni tip ürün satışı ile eski tip ürün satışının gelire etkisini anlamak adına A/B testi gerçekleştirilecek ve sonuçlar yorumlanacaktır.

NOT: Çalışmamızda, kullanıcılar rastsal olarak iki eşit parçaya ayrılarak test ve kontrol grubu olarak konumlandırılmışlardır. 
Çalışma 1 ay sürmüştür ve 1 ayın sonunda elde edilen veriler üzerinden analiz ve çıkarımlar yapılacaktır.


========================================================================================================================
Çalışma Konusu:
Cat Heroes: Puzzle Adventure oyununda bulunan Max’s Tree Tower isimli etkinlik 38. seviyeyi geçince açılmaktadır. 
Oyuncuları dönüştürme (conversion) konusunda son derece güçlü olan bu kart seçme oyununda, AB testler yapılarak daha çok gelir elde edilebileceğini düşünmekteyiz.
Bu etkinlik üzerinde bir AB test çalışması yapılacaktır.
• Çalışmanın amacı: Yeni tip ürün satışı (paket urun grupları) ile eski tip ürünsatışının gelire etkisini gözlemlemek.

Önemli Not: Bu çalışmada kullanılan rakamlar uydurmadır, amaç çalışmanın iskeletini oluşturmaktır.

"""

#import and concat
A=pd.read_excel(r"E:\PROJECTS\ABtest_Cratoonz\data\data.xlsx", sheet_name="A") #Hali Hazırdaki Satış Tipi, yani sadece mavi kristal satışı olan
B=pd.read_excel(r"E:\PROJECTS\ABtest_Cratoonz\data\data.xlsx", sheet_name="B")# Yeni satış tipi, sepet şeklinde, hem mavi kristal var hem de bonus ürünler.
print(A.columns) # Tek tip
print(B.columns) # Sepet Tip


## Group id
A["Group_id"]="A"
B["Group_id"]="B"

df=pd.concat([A,B], axis=0, ignore_index= True)

# Yeni Metrikler****************
# Random metrikler oluşturma.

#1- Satış_Miktarı: adet olarak ilgili ürünün 1 aylık toplam satış miktarını gösterir.
#2- Gelir: ilgili ürünün 1 aylık toplam Satis Geliri (Satış miktarı * Satış fiyatı çarpımından oluşur).
np.random.seed(1234)
df["Satis_Miktari"]= np.random.randint(0,10000, size=12, dtype=int)

df['Gelir']= df["Satis_Miktari"] * df["FİYAT"]

# A- EDA:

print(df.isnull().sum())

print('A Grubu eksik degerleri: \n', df[df["Group_id"]=="A"].isnull().sum())
print('B Grubu eksik degerleri: \n', df[df["Group_id"]=="B"].isnull().sum())

# NOT:Eksik değerler var, ancak bunlar gerçek eksik değer değil, ürün paketinde olmayan ürünler. dolayısıyla 0 ile değiştirmek daha mantıklı.

df = df.fillna(0)

print(df.describe().T)

#Yorum:
    # İki grupta ortalama 6106 adet satis yapılmış bu 1 aylık dönemde. En düşük satış miktarı 664, en yüksek ise 9449 adet  olarak görülmekte.
    # İki grupta ortalama 3445 mavi kristal satıldığında, ortalama gelir 158.908 olmuş.

# Visualization
# 1-Histogram without cross
num_cols= [col for col in df.columns if df[col].dtypes != "O"]
print(num_cols)
for col in num_cols:

    sns.distplot(df[col], kde= True)
    plt.show()

# 2-Contamination, without cross

for col in num_cols:
    sns.kdeplot(df[col], shade=True)
    plt.show()

# 3.1 -  A ve B kırılımında dağılımlarına bakalım
# Bazı ürünler sadece belli grupta olduğundan, A grubu dağılımı gözükmemektedir, bu nedenle Mavi Kristal, satış ve gelir açısından bakmakta fayda var.

num_cols= [col for col in df.columns if df[col].dtypes != "O"]
for col in num_cols:
    (sns.
     FacetGrid(df,
               hue="Group_id",
               height=5,
              )
     .map(sns.kdeplot,col, shade=True)
     .add_legend())
    plt.show()

## Yorum:
        # Gelir açısından, birbirinden farklı dağılım göstermiş test ve kontrol grubu.
        # Satış Miktarı açısından da farklı dağılım göstermiş.
        # Ancak unutmamak gerekir ki, burada dağılımlara görsel anlamda bakıyoruz, zira bu farklılık ya da benzerliklerin
        # istatistiki olarak anlamlı olup olmadıklarına bakılması gerekir.

# 4-boxplot

for col in num_cols:

    sns.boxplot(x="Group_id", y=col, data=df, orient="v")
    plt.show()

"""
- Yorum - 1 : Bazı ürünlerde aykırı değer var, ancak bu ürünler (örneğin TOP, çekiç) gerçektem kullanılan fiyat değerleri olduğu için,  aykırı olarak kabul edilmemişlerdir.
Bunlar analize dahil edilmediğinden gerek yok, ama fiyat, gelir satış miktarı vb de varsa, bunlara müdahale gerekebilir! Yapısal bir değişim olabilir. 

- Yorum - 2 : B grubuna ait aykırı değerler için önişleme kapsamında LOF veya belirlenecek eşik değerlerine baskılama yöntemleri uygulanabilir.

- Yorum - 3 : A ve B grupları arasında Satış Miktarı ve fiyat noktasında farklılık var gibi gözüküyor, ancak bu farklılık istatistiki
olarak anlamlı mıdır henüz bilinmemektedir.
 
- Yorum - 4 : A ve B grupları arasında Gelir açısından bir fark var gibi, ancak bu farklılık istatistiki olarak anlamlı mıdır henüz bilinmemektedir.

 """
# İki grubun Gelir ve Satış miktarlarının karşılaştırışması
print(df.groupby("Group_id").mean()) # gruplar açısından tüm değişkenlerin ortalamasına bakalım.

# yorum:
        # 1- yukarıda box-plot üzerinden yapılan yorumda satış gelirleri farklı değil gibi denilmişti, ama burada görüldüğü üzere farklı. bu durum istatistiki açıdan önemli mi tabii
        #burada da henüz bilmiyoruz.
        # 2- satış miktarları da ortalamada birbirinden farklı duruyor.

# c - a/b test
# hipotez olusturma ve yorumlama

# 1-  "satış miktarı"

# Ho: ortalama satış miktarı açısından , tek tip ürün ile sepet tipi ürün arasında istatistiki olarak fark yoktur.
# H1: ortalama satış miktarı açısından , tek tip ürün ile sepet tipi ürün arasında istatistiki olarak fark vardır.

df_A= df[df["Group_id"]=="A"]
df_B= df[df["Group_id"]=="B"]

# NORMALLİK VARSAYIMI

## HO: Normallik varsayımı sağlanıyor.
## H1: Normallik varsayımı sağlanmıyor.

statistic, pvalue = shapiro(df_A["Satis_Miktari"])
statistic, p1value = shapiro(df_B["Satis_Miktari"])

alpha = 0.05

if ((pvalue < alpha) == False) and ((p1value < alpha) == False):
    print("""Normal dağılım varsayımı sağlandı. yani, " H0: "Normal dağılım koşulu sağlanıyor" hipotezi reddedilemedi.""")

    # Varyansların homojenliği varsayımı

    # H0: Varyanslar homojendir.
    # H1: Varyanslar homojen değildir.

    test, l_pvalue = levene(df_A["Satis_Miktari"], df_B["Satis_Miktari"])

    if (l_pvalue < alpha) == False:
    # Normal Dağılım + Varyans Homojenliği Varsayımı Sağlandı) Parametrik bağımsız iki örneklem t testi uygulanır.

        test_ist, ind_pvalue = ttest_ind(df_A["Satis_Miktari"], df_B["Satis_Miktari"], equal_var=True)

         # Parametrik Testin P-value Değerlerinin Yorumlanması
        if (ind_pvalue < alpha):

                print(""" Varyans homojenliği sağlandı. yani "H0: Varyanslar homojen" hipotezi reddedilemedi.""")
                print( "Parametrik bağımsız iki örneklem t testi uygulanır")
                print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, ind_pvalue))
                print(" Genel Hipotez: H0, reddedildi.  A ve B grupları arasında Satis_Miktari ortalaması açısından istatistiki olarak anlamlı bir farklılık vardır.")

         else:
                print(""" Varyans homojenliği sağlandı. yani "H0: Varyanslar homojen" hipotezi reddedilemedi.""")
                print( "Parametrik bağımsız iki örneklem t testi uygulanır")
                print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, ind_pvalue))
                print("Genel Hipotez: Ho, reddedilemedi. A ve B grupları arasında Satis Miktari ortalaması açısından istatistiki olarak anlamlı bir farklılık yoktur.")
    else:
        # Welch T Testi uygulanır: Bağımsız iki örneklem T testi içinde sadece equal_var= False yapılarak uygulanabilir.
        # Non parametrik bağımsız iki örneklem t testi uygulanır.( Normal Dağılım varsayımı sağlandı, Varyans Homojenlği Sağlanamadı)

        test_ist, indv_pvalue = ttest_ind(df_A["Satis_Miktari"], df_B["Satis_Miktari"], equal_var=False)

        if (indv_pvalue < alpha) :
                print("""Varyans homojenliği varsayımı sağlanamadı, yani "H0: Varyanslar Homojendir" reddedildi""")
                print( "Non Parametrik bağımsız iki örneklem t testi uygulanır")
                print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, indv_pvalue))
                print(" Genel Hipotez: H0, reddedildi.  A ve B grupları arasında Satis_Miktari ortalaması açısından istatistiki olarak anlamlı bir farklılık vardır.")

        else:
                print("""Varyans homojenliği varsayımı sağlanamadı, yani "H0: Varyanslar Homojendir" reddedildi""")
                print( "Non Parametrik bağımsız iki örneklem t testi uygulanır")
                print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, indv_pvalue))
                print("Genel Hipotez: H0, reddedilemedi. A ve B grupları arasında Satis_Miktari ortalaması açısından istatistiki olarak anlamlı bir farklılık yoktur.")

else:
# Non Parametrik bağımsız iki örneklem t testi uygulanır (Normallik varsayımının sağlanamadığı durum)
    statist, pm_value = mannwhitneyu(df_A["Satis_Miktari"], df_B["Satis_Miktari"])

    if (pm_value < alpha ):
        print("""Normal dağılım varsayımı sağlanamadı. yani, " H0: "Normal dağılım koşulu sağlanıyor" hipotezi reddedildi.""")
        print("Normallik Varsayımı Shapirov p-value = %.4f" % (pvalue))
        print("Non parametrik bağımsız iki örneklem t testi uygulanır.")
        print("Mannwhitneyu p-value = %.4f" % (pm_value))
        print(" Genel Hipotez: H0, reddedildi.  A ve B grupları arasında Satis_Miktari ortalaması açısından istatistiki olarak anlamlı bir farklılık vardır.")

    else:
        print("""Normal dağılım varsayımı sağlanamadı. yani, " H0: "Normal dağılım koşulu sağlanıyor" hipotezi reddedildi.""")
        print("Normallik Varsayımı Shapirov p-value = %.4f" % (pvalue))
        print("Non parametrik bağımsız iki örneklem t testi uygulanır.")
        print("Mannwhitneyu p-value = %.4f" % (pm_value))
        print("Genel Hipotez: H0, reddedilemedi. A ve B grupları arasında Satis_Miktari ortalaması açısından istatistiki olarak anlamlı bir farklılık yoktur.")

print(df.groupby("Group_id").agg({"Satis_Miktari":np.mean}))


# 2- Gelir Açısından


# HO: Ortalama Gelir açısından , tek tip ürün ile sepet tipi ürün arasında istatistiki olarak anlamlı bir fark yoktur.
# H1: Ortalama Gelir açısından , tek tip ürün ile sepet tipi ürün arasında istatistiki olarak anlamlı bir fark vardır.

statistic, pvalue = shapiro(df_A['Gelir'])
statistic, p1value = shapiro(df_B['Gelir'])

alpha = 0.05

# 1- Normal Dağılım Varsayımı
## HO: Normallik varsayımı sağlanıyor
## H1: Normallik varsayımı sağlanmıyor.

if ((pvalue < alpha) == False) and ((p1value < alpha) == False):
    print("""Normal dağılım varsayımı sağlandı. yani, " H0: "Normal dağılım koşulu sağlanıyor" hipotezi reddedilemedi.""")

    # Varyansların homojenliği varsayımı

    # H0: Varyanslar homojendir.
    # H1: Varyanslar homojen değildir.

    test, l_pvalue = levene(df_A["Gelir"], df_B["Gelir"])

    if (l_pvalue < alpha) == False:
    # Normal Dağılım + Varyans Homojenliği Varsayımı Sağlandı) Parametrik bağımsız iki örneklem t testi uygulanır.
        test_ist, ind_pvalue = ttest_ind(df_A["Gelir"], df_B["Gelir"], equal_var=True)

         # Parametrik Testin P-value Değerlerinin Yorumlanması
        if (ind_pvalue< alpha):

            print(""" Varyans homojenliği sağlandı. yani "H0: Varyanslar homojen" hipotezi reddedilemedi.""")
            print("Parametrik bağımsız iki örneklem t testi uygulanır")
            print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, ind_pvalue))
            print(" Genel Hipotez: H0, reddedildi.  A ve B grupları arasında Gelir ortalaması açısından istatistiki olarak anlamlı bir farklılık vardır.")

         else:
             print(""" Varyans homojenliği sağlandı. yani "H0: Varyanslar homojen" hipotezi reddedilemedi.""")
             print("Parametrik bağımsız iki örneklem t testi uygulanır")
             print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, ind_pvalue))
             print("Genel Hipotez: Ho, reddedilemedi. A ve B grupları arasında Gelir ortalaması açısından istatistiki olarak anlamlı bir farklılık yoktur.")
    else:
        # Welch T Testi uygulanır: Bağımsız iki örneklem T testi içinde sadece equal_var= False yapılarak uygulanabilir.
        # Non parametrik bağımsız iki örneklem t testi uygulanır.( Normal Dağılım varsayımı sağlandı, Varyans Homojenlği Sağlanamadı)

        test_ist, indv_pvalue = ttest_ind(df_A["Gelir"], df_B["Gelir"], equal_var=False)

        if (indv_pvalue < alpha) :
            print("""Varyans homojenliği varsayımı sağlanamadı, yani "H0: Varyanslar Homojendir" reddedildi""")
            print("Non Parametrik bağımsız iki örneklem t testi uygulanır")
            print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, indv_pvalue))
            print(" Genel Hipotez: H0, reddedildi.  A ve B grupları arasında Gelir ortalaması ortalaması açısından istatistiki olarak anlamlı bir farklılık vardır.")

        else:
            print("""Varyans homojenliği varsayımı sağlanamadı, yani "H0: Varyanslar Homojendir" reddedildi""")
            print("Non Parametrik bağımsız iki örneklem t testi uygulanır")
            print("Test istatistiği =%.4f, p-value= %.4f" % (test_ist, indv_pvalue))
            print("Genel Hipotez: H0, reddedilemedi. A ve B grupları arasında Gelir ortalaması açısından istatistiki olarak anlamlı bir farklılık yoktur.")
else:
# Non Parametrik bağımsız iki örneklem t testi uygulanır (Normallik varsayımının sağlanamadığı durum)
    statist, pm_value = mannwhitneyu(df_A["Gelir"], df_B["Gelir"])

    if (pm_value < alpha):
        print("""Normal dağılım varsayımı sağlanamadı. yani, " H0: "Normal dağılım koşulu sağlanıyor" hipotezi reddedildi.""")
        print("Normallik Varsayımı Shapirov p-value = %.4f" % (pvalue))
        print("Non parametrik bağımsız iki örneklem t testi uygulanır.")
        print("Mannwhitneyu p-value = %.4f" % (pm_value))
        print(" Genel Hipotez: H0, reddedildi.  A ve B grupları arasında Gelir ortalaması ortalaması açısından istatistiki olarak anlamlı bir farklılık vardır.")

    else:
        print("""Normal dağılım varsayımı sağlanamadı. yani, " H0: "Normal dağılım koşulu sağlanıyor" hipotezi reddedildi.""")
        print("Normallik Varsayımı Shapirov p-value = %.4f" % (pvalue))
        print("Non parametrik bağımsız iki örneklem t testi uygulanır.")
        print("Mannwhitneyu p-value = %.4f" % (pm_value))
        print("Genel Hipotez: H0, reddedilemedi. A ve B grupları arasında Gelir ortalaması açısından istatistiki olarak anlamlı bir farklılık yoktur.")


print(df.groupby("Group_id").agg({'Gelir': np.mean}))
#
print("\n Final Yorumu 1 : Satis Miktari değişkni için eski tip ve sepet tipi uygulamıları sonucu oluşan fark (Eski tip için 5299, sepet tipi için 6913) 1614\n",
"olmasına rağmen, %95 güvenililirlik seviyesinde istatistiki olarak anlamlı değildir.\n"
, "Final Yorumu 2 : Gelir değişkni için eski tip ve sepet tipi uygulamıları sonucu oluşan fark (Eski tip için 5299, sepet tipi için 6913) 1614 \n"
, "olmasına rağmen, %95 güvenililirlik seviyesinde istatistiki olarak anlamlı değildir. \n\n"

, "Bu sonuçlara göre, aradaki fatk istatistiki olarak anlamlı olmadığından dolayı\n"
, "1. Testin periyodik olarak tekrar edielerek sonuçları gözlemlenmelidir.\n"
, "2. Test her iki tip için de verilen mavi kristal sayısının aynı olduğu alt ggruplar kırılımında özelleşltirilerek anlamlı sonuçlar alde edilerek \n"
, "yeni fiyat güven aralıkları belirlenebilir.")

#############test