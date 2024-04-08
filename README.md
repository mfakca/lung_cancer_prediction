# Akciğer Kanseri Tespiti Projesi

Bu proje, derin öğrenme ve görüntü işleme tekniklerini kullanarak akciğer kanserinin erken teşhisine yardımcı olmayı amaçlamaktadır. Proje, akciğer röntgen görüntülerinden kanseri otomatik olarak tespit edebilen bir yapay zeka modelinin geliştirilmesine odaklanmaktadır.

## Projenin Amacı

Akciğer kanseri, dünya çapında en yaygın kanser türlerinden biridir ve erken teşhis hayat kurtarıcı olabilir. Bu proje, radyologların akciğer kanserini daha hızlı ve daha doğru bir şekilde teşhis etmelerine yardımcı olacak bir araç geliştirmeyi hedeflemektedir.

## Kullanılan Teknolojiler

- Python  
  - TensorFlow 
  - OpenCV
  - Scikit-learn
  - Tkinter
  - matplotlib

## Kurulum

Model ve veri klasörleri boyutlarında dolayı Google Drive'da tutulmuştur. Uygulamayı sorunsuz kullanabilmek için bu dosyaları [buraya](https://drive.google.com/drive/folders/1_QuYW4-kE19j9LG-KaTxiJ2RcxJUx6Xl?usp=sharing) tıklayarak çalıştıracağınız ortama indirmeniz gerekmektedir.

#### Uygulama Görünümü:
![image](https://github.com/mfakca/lung_cancer_prediction/assets/56470222/2c1993b1-15b1-48ca-85d2-a737ded3616f)


## Kullanılan Modeller ve Performans
| Model       | Train Accuracy | Test Accuracy | Valid Accuracy | Train Precision | Test Precision | Valid Precision | Train Recall | Test Recall | Valid Recall | Train F1 | Test F1 | Valid F1 |
|-------------|----------------|---------------|----------------|-----------------|----------------|-----------------|--------------|-------------|--------------|----------|---------|----------|
| VGG16       | 80%            | 75%           | 81%            | 60%             | 77%            | 56%             | 89%          | 70%         | 93%          | 71%      | 73%     | 70%      |
| Inception   | 72%            | 71%           | 67%            | 67%             | 74%            | 79%             | 42%          | 66%         | 16%          | 51%      | 70%     | 26%      |
| Efficient   | 82%            | 71%           | 80%            | 87%             | 74%            | 80%             | 78%          | 74%         | 82%          | 74%      | 77%     | 77%      |


## Gelişim Yönleri: 
1. Veriseti Boyutu: Bu verisetinde train/test/validation için olmak üzere yaklaşık 1000 görsel bulunmakta. Örnek sayısı artarsa modelin başarım oranı artabilir.
2. Veriseti Kalitesi: Veriseti içerisindeki görsellerin hepsi spesifik olarak bu problem için düzenlense veya etiketlense modelin başarım oranı artabilir.
3. Sınıf Sayısı: Bu problemde 4 (3 Kanserli- 1 Kansersiz) farklı sınıfı tahmin etmek üzerine eğitilmiş bir model kullanıldı. Eğer problemi kanserli kansersiz şeklinde ele alsaydık VGG modelinin test doğruluk oranı %98 (Karmaşıklık Matrisine göre = 309/315) olacaktı.
