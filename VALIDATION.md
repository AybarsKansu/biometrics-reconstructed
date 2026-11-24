# Gerçek veri ve çalışma durumu

İndekste 13.233 okunabilir görüntü ve 5.749 kişi bulunur. Kişi grupları G/P/H için 1.000/2.000/2.500, kullanılmayanlar için 249'dur. Yerel çift dosyası standart sırada 6.000 çift içerir.

Embedding üretimi beklemektedir. G rollerinin son hali ve gerekebilecek P değişimleri çıkarım sonrasında belirlenir. Yerel model indirmeleri tamamlanmamıştır. Model dosyaları ve requirements.txt içindeki bağımlılıklar hazırlandıktan sonra prepare.py çalıştırılmalıdır.

Sadeleştirilen kaynakların sözdizimi ve modül importları kontrol edilmiştir. Gerçek embedding çıkarımı, 200 galerili kalibrasyon, LFW doğruluk ölçümü veya saldırı koşusu bu düzenleme sırasında çalıştırılmamıştır.

Proje komut satırı argümanı almaz; config.json kullanır. Kod veya embedding değişimlerini izleyen bir hash sistemi yoktur. Kalibrasyonun ne zaman tekrar çalıştırılacağı kullanıcı tarafından belirlenir.
