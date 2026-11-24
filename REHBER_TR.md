# Biometrics projesinin Türkçe kod rehberi

Proje, yüz fotoğraflarını 512 sayılık vektörlere çevirip bu vektörlerle çalışan yerel bir sistem kuruyor. Sonra saldırı algoritmaları sisteme aday vektörler gönderiyor; biz de ilk kabulün kaçıncı sorguda geldiğini kaydediyoruz. Algoritmaların gerçek yüz fotoğrafı üretmesi veya modeli eğitmesi söz konusu değil.

Kişiler üç gruba ayrılıyor: **G**, sisteme kayıtlı 1.000 hedef kişi; **P**, saldırıların başlangıç vektörlerini aldığı 2.000 kişi; **H**, kabul eşiklerini belirlediğimiz 2.500 kişi. Aynı kişi iki grupta bulunmuyor. G'de her kişinin bir fotoğrafının vektörü sisteme kaydediliyor; buna **enrollment** deniyor. Diğer fotoğrafları **held-out** olarak ayrılıyor ve yalnızca sonuç değerlendirmesinde kullanılıyor.

Akış `prepare.py → calibrate.py → experiment.py → report.py` şeklinde. Aşağıda sekiz Python dosyasındaki 40 fonksiyon, bu akıştaki yerleriyle anlatılıyor.

## common.py

Diğer dosyaların ortak kullandığı küçük işlemler burada. Dosya okuma, normalizasyon veya gruba göre vektör seçme kodunu her scriptte tekrar yazmamak için ayrılmış.

### read_json

Verilen yoldaki JSON dosyasını okuyup Python sözlüğüne veya listesine çeviriyor. Örneğin config'i okuduğumuzda dosyadaki ayarlara `config["experiment"]["query_budget"]` şeklinde ulaşabiliyoruz. UTF-8 kullanıldığı için Türkçe metinler de korunuyor. Dosya yoksa veya JSON bozuksa Python'un ilgili hatası geliyor.

### write_json

Bir sözlüğü veya listeyi JSON dosyasına yazıyor. Kullanılan ayarları, hesaplanan eşikleri ve kısa veri bilgilerini bu şekilde saklıyoruz. Üst klasör yoksa oluşturuyor, metni girintili yazıyor ve Türkçe karakterleri koruyor. `allow_nan=False` sayesinde NaN gibi geçersiz sayısal değerler sonuç dosyasına normal bir sayıymış gibi yazılamıyor.

### load_config

`config.json` dosyasını okuyup içindeki yolları tam dosya yollarına çeviriyor. Böylece `work/embeddings.npz` gibi bir yol, config'in bulunduğu proje klasörüne göre yorumlanıyor. Ardından modelin buffalo_l ve vektör boyutunun 512 olduğunu kontrol edip config sözlüğünü döndürüyor. Deney seçimi ve parametreler bu sözlükten okunuyor; komuta argüman vermiyoruz.

### read_index

Görüntü indeksinin CSV dosyasını okuyup her satırı bir sözlük haline getiriyor. Bir satır tek fotoğrafı temsil ediyor: kime ait olduğu, dosya yolu, grubu, rolü ve varsa embedding satırı birlikte duruyor. CSV'den gelen değerler metin olarak okunuyor; örneğin satır numarası dosyada sayıya benzese de Python'a ilk geldiğinde metin oluyor.

### write_csv

Satır sözlüklerini CSV dosyasına yazıyor. Hem görüntü indeksi hem sonuç tabloları bunu kullanıyor. Sütun adları ayrıca verilmediyse ilk satırın anahtarlarını başlık kabul ediyor, sonra bütün satırları bu sırayla yazıyor. Üst klasörü gerektiğinde oluşturuyor; Windows'ta fazladan boş satırlar oluşmaması için CSV'ye uygun satır sonu ayarını kullanıyor.

### normalize

Vektörün uzunluğunu hesaplayıp her elemanını bu uzunluğa bölüyor; sonuçta uzunluğu 1 olan bir vektör elde ediyoruz. Bu işlemden sonra iki vektörün iç çarpımı doğrudan kosinüs benzerliğini veriyor. Fonksiyon 512 eleman bekliyor, sıfır vektörü ve geçersiz sayıları reddediyor. Uzunluğu hesaplarken `float64`, döndürdüğü vektörde `float32` kullanıyor.

### check_embeddings

Embedding üretiminin sonunda ortaya çıkan matrisin kullanılabilir olduğunu kontrol ediyor. Her görüntüye bir satır düşmesi, her satırın 512 tane `float32` sayı içermesi, görüntü kimliklerinin tekrarlanmaması ve vektör uzunluklarının yaklaşık 1 olması bekleniyor. Bu fonksiyon dosya sonradan değiştirilmiş mi diye bakmıyor; üretimin sayısal olarak düzgün sonuç verdiğini kontrol ediyor. Sorun bulursa depoyu kaydetmeden hata veriyor.

### load_data

İndeksi ve embedding deposunu okuyup vektörlerle doğru görüntü satırlarını eşleştiriyor. Depoda vektörlerin yanında görüntü kimlikleri de bulunduğu için her vektörün hangi indeks satırına ait olduğunu bu kimlikten buluyor. Tüm indeksi, vektör matrisini ve vektör sırasına dizilmiş indeks satırlarını döndürüyor. Sonraki işlemler bu son iki çıktıyı birlikte kullanarak bir vektörü yanlış kişiye atamadan ilerliyor.

### select

Ortak veri içinden istenen grubu veya rolü seçiyor. Örneğin `partition="P"` verilince saldırgan havuzu, `role="enrollment"` verilince kayıtlı galeri geliyor. Her satır için koşullara uyup uymadığını belirleyen bir maske oluşturuyor ve hem vektörleri hem karşılık gelen satırları aynı maskeyle seçiyor. Böylece G, P ve H için ayrı embedding dosyaları tutmuyoruz.

### validate_roles

Kişilerin yanlışlıkla iki farklı gruba dağıtılmadığını ve G/P/H kişi sayılarının beklenen değerlerde olduğunu kontrol ediyor. Hazırlığın son aşamasında ayrıca her G kişisinin bir enrollment vektörü ve en az bir held-out vektörü bulunmasını istiyor. Bunlardan biri yoksa o kişi için hedef sistemi ve recovery değerlendirmesini birlikte kuramayacağımızdan hata veriyor. Kontrol geçtiğinde grup kişi sayılarını döndürüyor.

## prepare.py

Gerçek fotoğrafların bulunduğu klasörle başlayıp kullanılabilir embedding deposunu oluşturuyor. `python prepare.py` çalıştırıldığında indeks yoksa hazırlanıyor, sonra görüntülerden vektörler çıkarılıyor. İndeks varsa mevcut indeks kullanılıyor.

### assign_initial_groups

Önce indeks satırlarını kişilere göre topluyor ve kişi listesini config'teki seed ile karıştırıyor. En az iki okunabilir fotoğrafı olan kişilerden G için gereken sayıda kişi seçiyor; çünkü birini sisteme kaydedip diğerini değerlendirmeye ayıracağız. G dışındaki kişilerden sırayla P ve H seçiliyor, kalanlar kullanılmıyor. Seçilen grupları görüntü satırlarına yazıp başlangıç rollerini atıyor. Seed aynı kaldığında aynı kişi sıralaması elde ediliyor.

### assign_roles

Bir fotoğrafın kendi grubu içinde ne işe yarayacağını belirliyor. P fotoğraflarına `pool`, H fotoğraflarına `calibration` rolü veriyor. G'de her kişinin kullanılabilir fotoğraflarını sıralayıp ilkini `enrollment`, diğerlerini `held_out` yapıyor. Embedding henüz üretilmemişse fotoğrafın okunabilirliğine, üretim bittiyse embedding'in başarıyla çıkmış olmasına bakıyor. Kullanılamayan fotoğrafları `unavailable` işaretliyor.

### finalize_roles

Bir fotoğraf okunabilse bile model yüz bulamayabilir. Bu yüzden embedding üretimi bittikten sonra G'deki herkesin en az iki kullanılabilir vektörü kalmış mı diye bakıyor. Yeterli vektörü kalmayan G kişisini, P'de en az iki vektörü bulunan bir kişiyle değiştiriyor; seçimler alfabetik sırayla yapılıyor. H aynı kalıyor. Ardından enrollment ve held-out rollerini yeniden atayıp yapılan kişi değişimlerini döndürüyor.

### build_index

Görüntü klasörünü gezip her fotoğraf için bir indeks satırı oluşturuyor. Fotoğrafı Pillow ile açmayı deniyor; açılamayanları da kaybetmeden başarısız olarak kaydediyor. Kişi adını klasörden, görüntü kimliğini dosya adından alıyor ve ikisinin uyuşmasını bekliyor. Başlangıçta embedding henüz bulunmadığından satır numarası `-1`, durumu `pending` oluyor. Sonra kişi gruplarını belirleyip indeksi CSV'ye, toplam sayıları metadata dosyasına yazıyor.

### extract

Yerel buffalo_l modelini açıp indeksteki okunabilir fotoğrafları tek tek işliyor. Fotoğrafta birden fazla yüz bulunursa en yüksek detection skoruna sahip olanı seçiyor; skor eşitliğinde yüz koordinatlarının sırasını kullanıyor. Seçilen yüzün 512 sayılık embedding'ini normalize edip görüntü kimliğiyle birlikte listeye ekliyor. Yüz bulunamayan veya işlenemeyen görüntüyü indekste başarısız işaretleyip devam ediyor. İşlem sonunda vektörleri kontrol ediyor, görüntülerin embedding satırlarını yazıyor, G için gereken son düzenlemeleri yapıyor ve tek NPZ deposunu kaydediyor. Model bilgileri ve başarısız görüntü sayıları da metadata'ya yazılıyor. Hazır depo varsa yanlışlıkla üzerine yazmak yerine duruyor.

### main

Hazırlık scriptinin başlangıç noktası. Config'i okuyup indeks dosyasının varlığına bakıyor; yoksa `build_index` ile oluşturuyor. Sonra `extract` çağırarak embedding üretimini başlatıyor. Böylece indeksleme ve vektör çıkarma için ayrı komut seçenekleriyle uğraşmadan `python prepare.py` çalıştırabiliyoruz.

## comparator.py

Sistemin bir aday vektöre nasıl cevap verdiği burada tanımlanıyor. Gerçek skor ve eşleşen kişi bu modülde biliniyor; algoritmaya ise yalnız seçtiğimiz geri bildirim gönderiliyor.

### compare

Gelen vektörü galerideki bütün enrollment vektörleriyle `gallery @ vector` işlemiyle karşılaştırıyor. Vektörler normalize olduğu için çıkan sayılar kosinüs benzerliği oluyor. En yüksek skoru ve o skorun galeri içindeki konumunu döndürüyor; tam eşitlik varsa ilk kaydı seçiyor. Yuvarlama nedeniyle skorun çok az 1'i aşması gibi durumları `[-1,1]` aralığına kırpıyor. Galeri konumu algoritmaya verilmek için değil, kabul olursa hangi kişinin held-out verisinin kullanılacağını bulmak için gerekiyor.

### feedback

Karşılaştırıcının bildiği gerçek skordan algoritmanın göreceği cevabı üretiyor. Float seçiliyse skoru, 1-bit seçiliyse yalnız kabul veya ret anlamına gelen 1/0 değerini veriyor. 4-bit ve 8-bit seçeneklerinde skor aralığını kutulara ayırıp kutu numarasını döndürüyor. Kutuların yarısı ret, yarısı kabul tarafına ayrıldığı için ayrıntı azalırken kabul kararı değişmiyor: 4 bitte 8 ve üzeri, 8 bitte 128 ve üzeri kabul demek. Tam eşik kabul tarafında kalıyor; aralık dışındaki skorlar uç kodlara yerleşiyor.

### recovery

İlk kabul edilen vektörü alıp eşleşen kişinin kayıt dışında tuttuğumuz diğer fotoğraf vektörleriyle karşılaştırıyor. Bu benzerliklerin medyanı H üzerinden belirlenen recovery eşiğini geçerse recovery başarılı sayılıyor. Böylece yalnız tek kayıt fotoğrafına benzeyen bir vektörle, kişinin diğer fotoğraflarına da benzeyen bir vektörü ayırıyoruz. Fonksiyon medyanı ve başarı durumunu döndürüyor; bunlar yalnız sonuç kaydına gidiyor, algoritmaya gönderilmiyor ve sorgu bütçesinden düşülmüyor.

### run_loop

Bir saldırı koşusunu baştan sona yürütüyor. Algoritmadan bir aday alıyor, sorgu sayısını artırıyor, adayı normalize edip galeriyle karşılaştırıyor ve ret durumunda görünür cevabı algoritmaya göndererek sıradaki adayı istiyor. İlk kabul geldiği anda aynı aday için recovery hesaplayıp koşuyu bitiriyor; recovery başarısız olsa da yeni aday aranmıyor. Başlangıç ve tekrar sorguları dahil her aday sayılıyor. Bütçe biterse `budget_exhausted`, aday listesi biterse `pool_exhausted`, geçersiz vektör gelirse `invalid_candidate` kaydı oluşuyor. Koşu bittiğinde kabul sorgusu, toplam sorgu sayısı ve özel değerlendirme sonucu tek sözlük olarak dönüyor.

## calibrate.py

Saldırıya başlamadan önce hangi skorun kabul sayılacağını H üzerinde belirliyor. Sistem için iki kabul eşiği, recovery değerlendirmesi için ayrı bir eşik çıkarılıyor.

### threshold_from_tail

Elimizdeki farklı-kişi skorlarından hedef yanlış kabul oranına uygun sınırı seçiyor. Toplam karşılaştırma sayısıyla hedef oranı çarpıp en fazla kaç yanlış kabule izin verileceğini buluyor. Sonra skorları tamamen sıralamak yerine `np.partition` ile ilgili sınır skorunu seçiyor. Kabul kuralımız `skor >= eşik` olduğu için sınır skorunun bir `float32` adım üstünü alıyor; böylece aynı sınır skoruna sahip çok sayıda gözlem hedef oranı aşırmıyor. Eşiği, gerçekten oluşan yanlış kabul oranını ve kullanılan karşılaştırma sayısını döndürüyor.

### system_scores

Sistemde kayıtlı olmayan kişilerin, 1.000 kişilik bir galeriye karşı ne kadar yüksek skor alabildiğini ölçüyor. H'den 1.000 kişi seçip her birinin tek vektörünü kayıt kabul ediyor; H'nin geri kalan kişilerini bu galeriyle karşılaştırıp her sorgunun en yüksek skorunu topluyor. Bunu config'teki sayı kadar, varsayılan 200 farklı galeri seçimiyle tekrarlıyor. Bu skorlar daha sonra %1 ve %0,1 yanlış kabul eşiklerini seçmekte kullanılıyor. Ayrıca seçilen kişilerin diğer fotoğraflarından aynı-kişi skorları topluyor; onlar dağılım ve ROC/DET grafiklerine gidiyor. G'deki hedef kişileri eşik seçmekte kullanmamak için bu işlem H'de yapılıyor.

### pairwise_scores

Recovery sırasında bir adayın tek bir kişinin diğer fotoğraflarına benzeyip benzemediğine baktığımız için burada farklı kişilerin tek tek vektör benzerliklerini topluyor. H içindeki her benzersiz çifti bir kez karşılaştırıyor; aynı görüntüyü kendisiyle karşılaştırmıyor ve ters sıradaki çifti yeniden saymıyor. Büyük matrisi tek seferde tutmamak için bloklarla çalışıyor. Recovery eşiğini belirlemek için yalnızca en yüksek farklı-kişi skorlarını bellekte saklıyor; ayrıca aynı ve farklı kişi histogramlarını güncelliyor. Sonunda %0,01 yanlış eşleşme eşiğini ve histogramları döndürüyor.

### calibrate

Kalibrasyonun bütün adımlarını bir araya getiriyor. Depodan H vektörlerini seçiyor, `system_scores` ile sistem skorlarını ve `pairwise_scores` ile recovery eşiğini hesaplıyor. Sistem skorlarının 0,1'inci yüzdeliğini kuantizerin alt sınırı olarak alıyor; iki hedef FMR için de kabul eşiğini çıkarıyor. Eşikleri ve gerçekleşen oranları `calibration.json`, dağılım sayımlarını `h_score_histogram.csv` içine yazıyor. `python calibrate.py` bu fonksiyonu config ile çağırıyor.

## attacks.py

Bu dosyada yalnız aday üretimi var. Algoritmalar başlangıçta P vektörlerini alıyor ve sorgularına gelen skor veya kodlarla ilerliyor. G kayıtlarını, eşleşen kişiyi ya da held-out sonucunu görmüyorlar. Sorgu sayımı ve durdurma kararını `run_loop` veriyor.

### zero_effort

P'deki vektörleri verilen sırayla, hiç değiştirmeden sisteme gönderiyor. Optimizasyon yapmadan yalnız hazır vektörleri denemek ne kadar başarılı oluyor sorusunun cevabı bu baseline. Geri bildirimden öğrenmiyor; havuzdaki vektörler bittiğinde aday üretimi sona eriyor. Buradaki havuz büyüklüğü kişi sayısı değil, kullanılabilir fotoğraf embedding'i sayısı.

### hill

P'den bir başlangıç vektörü seçip çevresinde küçük rastgele değişiklikler deniyor. Her adımda rastgele bir yön oluşturuyor, mevcut vektöre config'teki adım büyüklüğüyle ekliyor ve sonucu normalize ediyor. Yeni adayın görünür skoru veya kodu daha yüksekse onu mevcut aday yapıyor; eşit veya düşükse eskisini koruyor. Art arda yeterince başarısız deneme olursa sıradaki P vektörüyle yeniden başlıyor. Kabul veya bütçe sonuna kadar bu aday akışını `run_loop` ilerletiyor.

### simplex

Varsayılan olarak P'den 20 aday alıp her birinin cevabını öğreniyor ve bunları iyiden kötüye sıralıyor. En kötü aday dışındaki vektörlerin ortalamasını merkez alıp en kötü adayı merkezin ötesine yansıtıyor. Bu aday çok iyiyse aynı yönde biraz daha uzağa gitmeyi, yeterince iyi değilse merkeze yaklaşmayı deniyor. Bunlar da işe yaramazsa en iyi aday korunup diğerleri ona yaklaştırılıyor. Üretilen her aday normalize edilip ayrı sorgulanıyor. En iyi cevap belirli sayıda iterasyon boyunca gelişmezse yeni P adaylarıyla başlıyor. Buradaki 20 köşe, 512 boyutun tamamını kapsayan bir simplex değil; az sayıda vektörle kullanılan bir arama yaklaşımı.

### spherical_crossover

Genetik algoritmada iki ebeveyn vektörden bir çocuk yönü oluşturuyor. Vektörlerin uzunluğu 1 olduğu için bunları küre üzerindeki iki nokta gibi düşünüp aralarındaki yaydan bir konum seçiyor; `fraction` bu konumun hangi ebeveyne yakın olacağını belirliyor. İki yön çok yakınsa basit ağırlıklı ortalamayı normalize ediyor. Neredeyse zıtlarsa tek bir kısa yay seçilemediğinden ilk ebeveyni kopyalıyor. Yalnız vektör hesaplıyor; kendisi sisteme sorgu göndermiyor.

### genetic

P'den varsayılan 20 vektörle bir popülasyon kurup hepsini sorgulatıyor. Her nesilde görünür cevaplara göre sıralama yapıyor ve en iyi iki bireyi olduğu gibi koruyor. Yeni birey üretmek için iyi sıralanan ebeveynlerden ikisini seçiyor, `spherical_crossover` ile karıştırıyor, rastgele Gaussian değişiklik ekliyor ve normalize ediyor. Çocuğu sorgulayıp popülasyona ekliyor; popülasyon tamamlanınca yeni nesle geçiyor. En iyi cevap uzun süre gelişmezse yeni P vektörleriyle baştan başlıyor. Kuantize deneyde bütün bu seçimler yalnız kodlara göre yapılıyor.

### candidates

Seçilen yöntem adına göre doğru aday üretecini döndürüyor. Önce seed ile P vektörlerinin deneneceği sırayı oluşturuyor; böylece aynı tekrar farklı yöntem ve hassasiyetlerde aynı başlangıç sırasını kullanıyor. Yöntemin kendi rastgele adımları için ayrı bir rastgele sayı üreteci kuruyor. Sonra `zero_effort`, `hill`, `simplex` veya `genetic` fonksiyonunu uygun ayarlarla seçiyor. Bu noktada henüz bir sorgu yapılmıyor; adaylar `run_loop` üreteci ilerlettikçe geliyor.

## yield ve send nasıl birlikte çalışıyor

Algoritmalardaki `value = yield candidate` satırı, adayı dışarı verip cevabı beklemek demek. `run_loop` bu adayı karşılaştırdıktan sonra `send(visible)` çağırınca gelen cevap `value` değişkenine yerleşiyor ve algoritma devam ediyor. İlk aday `next` ile alınıyor; sonraki reddedilen adaylarda bu iki yönlü iletişim kullanılıyor. Kabul geldiğinde üreteç kapatılıyor. Dolayısıyla algoritmanın bir sonraki adımı recovery sonucundan etkilenemiyor.

## experiment.py

Config'teki yöntem, hassasiyet ve eşik seçimlerini çalıştırıyor. Tek bir koşuyu `run_loop` yönetirken bu dosya tekrarları ve farklı koşulları sıraya koyuyor.

### experiment_data

Ortak embedding deposundan deneyin kullanacağı parçaları ayırıyor. Enrollment satırları G galerisi, P satırları başlangıç havuzu oluyor. Held-out vektörlerini ise kişi adına göre bir sözlükte topluyor; böylece bir aday kabul edildiğinde doğru kişinin diğer fotoğrafları bulunabiliyor. Galeri sırasıyla aynı sıradaki kişi listesini de hazırlayıp galeri, kişi listesi, held-out sözlüğü ve P havuzunu döndürüyor.

### run

Embedding'leri ve H'den hesaplanan eşikleri okuyup yeni bir sonuç klasörü açıyor. Kullanılan config'i ve eşikleri bu klasöre yazdıktan sonra seçilen eşik, yöntem, hassasiyet ve tekrarların üzerinden geçiyor. Her tekrarda `candidates` ile algoritmayı kuruyor, `run_loop` ile koşuyu yürütüyor ve sonucu JSONL dosyasına ekliyor. Tekrar seed'i başlangıç seed'ine tekrar numarası eklenerek belirleniyor; farklı koşullarda aynı tekrar böyle eşleştiriliyor. Zero-effort her eşik için bir kez çalıştırılıyor, çünkü kabul kararı float veya kuantize cevap verilmesine bağlı değil. Başarısız koşular da dosyada kalıyor.

### main

Config'in `experiment` bölümündeki `methods`, `granularities` ve `targets` listelerini okuyup `run` fonksiyonuna veriyor. Örneğin yalnız hill ve float seçiliyse sadece o koşulları çalıştırıyor. Tam karşılaştırma için bu listelere diğer yöntemler, hassasiyetler ve iki hedef FMR ekleniyor. Bütçe ve tekrar sayısı da config'te duruyor; komut her iki durumda da `python experiment.py`.

## lfw_check.py

Bu script saldırıların kaç sorguda kabul bulduğundan ayrı olarak modelin standart LFW çiftlerindeki doğruluğunu ölçüyor. Çiftler aynı kişinin iki fotoğrafı veya farklı kişilerin fotoğrafları olarak etiketli geliyor.

### image_id

Çift dosyasındaki kişi adını ve fotoğraf numarasını, embedding deposunda kullandığımız görüntü kimliğine dönüştürüyor. Numarayı dört haneye sıfırla tamamlayıp `kişi_0001` biçiminde birleştiriyor. Bu küçük dönüşüm sayesinde çiftte sözü edilen fotoğrafın vektörünü depoda bulabiliyoruz.

### read_pairs

Desteklenen TXT veya CSV çift dosyasını okuyup her kaydı iki görüntü kimliği ve aynı kişi olup olmadığını gösteren etiketle döndürüyor. Üç alanlı satırlar aynı kişinin iki fotoğrafını, dört alanlı satırlar farklı kişileri belirtiyor. Toplam 6.000 çift bekliyor; her 600 çiftlik bölümde önce 300 aynı, sonra 300 farklı kişi çifti bulunmasını kontrol ediyor. Bu sıra korunmalı, çünkü sonraki değerlendirme on ayrı bölüm üzerinden yapılıyor.

### cross_validate

6.000 çifti on bölüm halinde değerlendiriyor. Her seferinde bir bölümü sonuç ölçmek için ayırıyor, kalan dokuz bölümde en iyi aynı/farklı kişi ayrımını veren eşiği seçiyor. Sonra bu eşiği ayırdığı bölümde kullanıp doğruluğu kaydediyor; böylece sonuç ölçülen çiftlerden eşik seçilmemiş oluyor. Karşılaştırmada kosinüsü `d² = 2 - 2*cosine` ile karesel uzaklığa çeviriyor ve uzaklık eşikten küçükse aynı kişi diyor. On bölümün eşik ve doğruluk listesini döndürüyor.

### check

Embedding deposunu ve çift listesini okuyup her çiftin vektörlerini buluyor. Gerekli bir fotoğrafın embedding'i yoksa o çifti atarak daha kolay bir değerlendirme yapmak yerine eksikliği kaydedip doğruluk sonucunu boş bırakıyor. Hepsi varsa çiftlerin kosinüslerini hesaplayıp `cross_validate` çağırıyor. Fold sonuçlarını, ortalama doğruluğu ve fold'lar arasındaki standart sapmayı JSON dosyasına yazıyor. `python lfw_check.py` çalıştırıldığında bu fonksiyon çağrılıyor.

## report.py

Kaydedilmiş koşuları okuyup sonuçları tablo ve grafik haline getiriyor. Yeni saldırı sorgusu yapmıyor. Script H histogramı bulursa onun grafiklerini, deney kaydı bulursa en son deney klasörünün özetini üretiyor.

### success_quantile

Koşuların yarısının veya %90'ının kaç sorguda başarıya ulaştığını hesaplıyor. Başarılı koşuların sorgularını sıralıyor ama gereken başarı sayısını bütün koşular üzerinden belirliyor. Örneğin koşuların yarısı bile başarılı olamamışsa başarılı birkaç koşunun medyanını vermiyor; medyana bütçe içinde ulaşılamadığını söylüyor. Recovery için de başarı zamanı ilk kabul sorgusu, çünkü o vektörden sonra arama devam etmiyor.

### summarize

Bir deney klasörünün ayarlarını ve koşu kayıtlarını okuyup sonuçları eşik, hassasiyet ve yönteme göre grupluyor. Her grup için kabul ve recovery oranlarını, sorgu medyanı/p90'ını ve başarısızlık sayılarını tabloya yazıyor. Recovery'yi hem bütün koşulara hem yalnız kabul edilenlere göre ayrı hesaplıyor. Ardından her sorgu sayısına kadar kaç koşunun başarılı olduğunu çizerek kabul ve recovery eğrilerini oluşturuyor; baseline aynı grafiklere ekleniyor. Başarısız koşular oranların paydasında kalıyor. Çok sayıda koşu bütçeyi tüketmişse bunu yorum dosyasına not ediyor; bütçeyi otomatik değiştirmiyor.

### h_report

Kalibrasyonun kaydettiği aynı-kişi ve farklı-kişi histogramlarını okuyup skor dağılımlarını çiziyor. Sonra farklı eşiklerde yanlış kabul ve yanlış ret oranlarını histogramların birikimli sayılarından çıkararak ROC ve DET grafiklerini oluşturuyor. Sistem ve tek çift karşılaştırmalarını ayrı gösteriyor. Grafikler kutulanmış skor sayılarından hesaplandığı için yaklaşık gösterimler; saldırılarda kullanılan kabul eşikleri kalibrasyon sırasında gerçek skorlarla belirlenmiş durumda. Sonuçları CSV ve PNG olarak kaydediyor.

## Ayarlar ve çıktı dosyaları

Yollar, kişi sayıları, seed'ler, yöntem parametreleri, bütçe ve tekrar sayısı `config.json` içinde. Kurulum bağımlılıkları `requirements.txt` dosyasında. `agent.md` ve `reconstruct.md` tasarım kararlarını, `VALIDATION.md` mevcut veri ve çalıştırma durumunu anlatıyor.

`work/data_index.csv` her görüntünün kişi, yol, grup ve rol bilgisini tutuyor. `work/embeddings.npz` vektörleri ve görüntü kimliklerini birlikte saklıyor. `metadata.json` model bilgileri ve hazırlık sayılarını, `calibration.json` eşikleri, `h_score_histogram.csv` dağılım sayımlarını içeriyor. Bir görüntünün okunabilir olması embedding'inin de çıktığı anlamına gelmiyor; bu nedenle indekste görüntü geçerliliği ve embedding durumu ayrı alanlar.

Her deneyin `results/pilot-.../` klasöründeki `resolved.json` kullanılan ayarlar, `private_records.jsonl` tek tek koşular içindir. Özet tablolar ve grafikler bunlardan üretiliyor. Özel kayıtta eşleşen kişi ve held-out medyanı bulunabiliyor; bunlar raporlama bilgisi, algoritmanın sorgu cevabı değil.

## Komut sırası

```powershell
python prepare.py
python calibrate.py
python experiment.py
python report.py
```

Embedding üretiminden sonra standart model doğruluğu için ayrıca `python lfw_check.py` çalıştırılabilir. Hazır embedding deposu varsa tekrar hazırlık yapmak gerekmez. Veriyi veya kalibrasyon ayarlarını değiştirirsen eşikleri `calibrate.py` ile yeniden hesaplamalısın.

Varsayılan deney tek hill/float/%1 ayarı, 20 tekrar ve koşu başına 1.000 sorgu. Tam 18 ayar için config'te yöntem listesi `["hill", "simplex", "genetic"]`, hassasiyet listesi `["float", "8", "4"]`, hedef listesi `[0.01, 0.001]` yapılır. Bu hedefler doğrudan kosinüs sayıları değil, H'den eşik seçmekte kullanılan yanlış kabul oranlarıdır.

Bir koşu kabul bulduğunda her durumda biter. `accepted=true, recovered=false` mümkün: vektör sisteme girmiştir ama aynı kişinin diğer fotoğraflarına yeterince benzememiştir. `budget_exhausted` bütçede kabul bulunmadığını, `pool_exhausted` aday havuzunun bittiğini belirtir. Bunlar silinecek hatalar değil, deneyin sonuçlarıdır.
