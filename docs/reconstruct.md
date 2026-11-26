# Biometrics yeniden kurulum planı

Bu belge yeni projenin uygulanacak kapsamını tanımlar. `agent.md` aynı planın İngilizce karşılığıdır. Eski kod referans olarak kullanılabilir; eski mimari, README ve yol haritasındaki çelişen kararlar taşınmayacak. Bu aşamada yalnızca plan yazılıyor; mevcut uygulama yeniden kurulmayacak.

## 1. Amaç ve deney tanımı

Yüz embedding'leriyle çalışan bir biyometrik sistemde, saldırganın yalnızca sistemin verdiği geri bildirimi kullanarak kabul edilen bir vektör bulmasını ölçüyoruz. Saldırgan fotoğraf göndermiyor; doğrudan embedding gönderiyor. Model eğitimi, fine-tuning ve görüntü yeniden oluşturma kapsam dışında.

İki sonuç ayrı kaydedilecek:

- Giriş başarısı: Gönderilen vektörün 1:N doğrulayıcı tarafından kabul edilmesi.
- Kimlik geri kazanımı (recovery): İlk kabul edilen aynı vektörün, o anda eşleştiği kişinin kayıt dışında tutulan fotoğraflarının embedding'lerine de yeterince benzemesi. Bu bir yüz görüntüsü üretme işlemi değildir.

İlk aşamadaki saldırı koşusu ilk kabulde durur. Recovery yalnızca bu ilk kabul edilen vektör için bir kez değerlendirilir. İlk kabulden sonra optimizasyona devam edilmez. Kabul yoksa recovery başarılı sayılmaz; bütçe dolan koşu sonuçlardan çıkarılmaz.

## 2. Mimari ve dosya sadeliği

- Doğrudan çalıştırılan, anlaşılır Python scriptleri kullanılacak. HTTP API, sunucu, mikroservis veya genel amaçlı framework kurulmayacak.
- Tek config; tek veri indeksi; tek embedding deposu kullanılacak.
- Küçük ortak fonksiyonlar gerektiğinde paylaşılabilir. Aynı işi tekrar yazmamak için yardımcı dosya kullanmak serbest; gereksiz sınıf ve katman eklenmeyecek.
- Veri indeksinde görüntü kimliği, kişi kimliği, göreli yol, geçerlilik, grup ve rol tutulacak. Embedding üretimindeki başarısızlıklar da bu indeksle ilişkilendirilecek.
- Embedding satırları görüntü kimlikleriyle açıkça eşleştirilecek. G/P/H ve enrollment/held-out seçimleri bu eşleşmeden yapılacak.
- Gruplar için ayrı embedding kopyaları, ikinci enrollment veritabanı, ayrı attacker_pool/image_roles/split_report çıktıları üretilmeyecek. Görünümler indeks üzerinden seçilecek.
- CSV, JSON ve NPZ formatları ihtiyaçlarına göre seçilebilir; aynı bilgiyi farklı formatlarda tekrar kaydetmek amaç olmayacak.
- Deney kayıtları ile bu kayıtlardan üretilen final tablo/grafikler farklı amaçlara hizmet eder ve tutulabilir.

## 3. Katman 1 — Ayarlar ve ortam

- Python bağımlılıkları tek ortam/bağımlılık dosyasında belirtilecek.
- Tek config içinde veri yolları, seed, model ayarları, kalibrasyon ayarları, saldırı parametreleri, tekrar sayısı ve sorgu bütçesi tutulacak.
- Aynı karşılaştırmalarda aynı seed'ler ve başlangıç adayları kullanılacak. Popülasyon/simplex yöntemleri için aynı seed'e bağlı havuz sırası kullanılacak.
- Her deney batch'i kullanılan config'i bir kez kaydedecek. Komutlar doğrudan çalışır; ayarlar config.json içindedir.

## 4. Katman 2 — Veri hazırlama

- LFW kullanılacak; hazır çıkarılmış görüntü klasörü okunacak. Uygulamanın parçası olarak kapsamlı arşiv yönetimi geliştirilmeyecek.
- Görüntülerin okunabilirliği, kişi bilgileri ve gerçek sayılar kontrol edilecek.
- Seed ile kişi bazında ayrık gruplar oluşturulacak:

| Grup | Kişi | Amaç |
|---|---:|---|
| G | 1.000 | Kayıtlı hedefler |
| P | 2.000 | Saldırganın başlangıçları ve zero-effort |
| H | 2.500 | Çalışma ve recovery eşiklerinin kalibrasyonu |
| Kalan | Nominal 249 | Kullanılmayacak |

- Ayrı development grubu ve hiperparametre araması ilk kapsamda yok.
- G'deki her kişinin en az iki kullanılabilir görüntüsü olacak: biri enrollment, kalanlar held-out.
- Embedding çıkarılamayan görüntüler açıkça işaretlenecek. G'de kullanılabilir enrollment veya held-out kalmazsa, H'yi değiştirmeden P'den en az iki kullanılabilir görüntüsü olan kişilerle deterministik değişim yapılabilir. Değişimler ve son sayılar kaydedilecek.
- P saldırgan tarafından kullanılabilir; G enrollment ve held-out verileri saldırgana verilmeyecek. H kalibrasyon içindir.

## 5. Katman 3 — Model ve embedding

- Model: ArcFace tabanlı InsightFace `buffalo_l` tanıma modeli. iResNet100'e geçilmeyecek; model karşılaştırması yapılmayacak.
- Paketle uyumlu yüz bulma/hizalama ve ön işleme kullanılacak; görüntü başına seçilen yüz politikası açık ve sabit olacak.
- 512 boyutlu embedding'ler float32 olarak saklanacak, L2 normalize edilecek.
- NaN, sonsuz değer, sıfır norm ve görüntü/satır eşleşmesi kontrol edilecek.
- Model ve ön işleme değişmedikçe mevcut embedding'ler tekrar kullanılacak.

Tek küçük metadata kaydında şunlar yeterli:

- Model adı ve tanıma checkpoint dosyasının adı.
- InsightFace sürümü ve yüz seçme/ön işleme ayarları.
- Veri kaynağı ve kullanılan LFW sürümü/hizalama varyantı.
- Split seed'i, grup sayıları ve embedding çıkarılamayan görüntü sayısı.

Her görüntü için ayrı hash envanteri veya modelin eğitim kimlikleriyle LFW örtüşmesi auditi yapılmayacak. İncelenmeyen örtüşme raporda sınırlılık olarak belirtilecek. Ham yüz görüntüleri ve model ağırlıkları dağıtılmayacak; kaynak ve kullanım koşulları kısaca kaydedilecek.

## 6. Katman 4 — Doğrulayıcı ve kalibrasyon

- Ana sistem 1:N: Her G kişisi için bir enrollment vektörü; toplam 1.000 vektör.
- Exact cosine karşılaştırması NumPy matris çarpımıyla yapılacak. FAISS ve ayrı vektör veritabanı yok.
- Skor tüm galeri karşılaştırmalarının maksimumudur. Kabul koşulu `score >= threshold`.
- Top-1 kişi kimliği yalnızca iç değerlendirme ve gerektiğinde savunma sayaçları için kullanılacak; saldırgana dönmeyecek.
- 1:1 karşılaştırma recovery değerlendirmesi için uygulanacak. Ayrı bir 1:1 saldırı deney matrisi ilk kapsamda yok.
- H üzerinden 200 seed'li geçici galeri kurulacak. Her galeride 1.000 kişi ve kişi başına bir enrollment olacak; diğer H kişilerinin görüntüleri impostor probe olarak kullanılacak.
- Bu 1:N top-1 skorlardan sistem FMR %1 ve %0,1 eşikleri hesaplanacak. Hedef ve gerçekleşen ampirik FMR kaydedilecek.
- Recovery için H içindeki farklı kişilerin 1:1 skorlarından FMR %0,01 eşiği hesaplanacak. Büyük karşılaştırmalar gerekirse bloklar halinde yapılacak.
- Güven aralıkları ve bootstrap hesaplanmayacak. Nadir hata oranları ve bağımlı skorlar hakkındaki belirsizlik raporda belirtilecek.
- Temel model kontrolü olarak standart LFW 6.000 çift / 10-fold doğrulama protokolü ayrı değerlendirilecek. Yayımlanan model başarısı ölçülmüş yerel sonuç gibi sunulmayacak.

Geri bildirim:

- Float: cosine skoru.
- 8-bit ve 4-bit: eşik altında ve üstünde ayrı eşit genişlikli kutular. `lo`, H top-1 impostor skorlarının %0,1 yüzdeliğinden alınacak. Kodların yarısı ret, yarısı kabul bölgesinde olacak; karar değişmeyecek.
- 1-bit: yalnızca kabul/ret.
- Eşik, alt/üst sınırlar, doygunluk ve sınırdaki kararlar açıkça uygulanacak. Kalibrasyon ayarları her çalışma eşiğiyle ilişkilendirilecek.

## 7. Katman 5 — Ortak deney döngüsü

- Saldırgan yalnızca P'den başlangıçlar ve izin verilen oracle yanıtını kullanacak.
- Sorgu sayımı, bütçe, durdurma ve kayıt tek ortak yerde yönetilecek.
- Başlangıç, popülasyon, simplex, yeniden başlatma ve aday değerlendirmelerinin her oracle çağrısı bütçeye dahil edilecek. Aynı aday tekrar sorgulanırsa o da sayılacak.
- Saklı recovery karşılaştırmaları oracle sorgusu sayılmayacak ve saldırı güncellemesine girdi olmayacak.
- İlk kabulde iç top-1 kimliği bulunacak; aynı kabul edilen vektörün bu kişinin held-out embedding'lerine cosine skorlarının medyanı hesaplanacak. Medyan H'den gelen recovery eşiğine eşit veya yüksekse recovery başarılıdır.
- İlk kabul skoru, kabul sorgu numarası, recovery sonucu ve toplam sorgu sayısı kaydedilecek. Bütçe/engel nedeniyle başarısız koşular da tutulacak.
- Saldırganın görebildiği yanıtlarla değerlendiricinin özel kayıtları ayrı tutulacak; bütün galeri veya özel skorlar her sorguda dosyaya dökülmeyecek.

Tekrar ve bütçe:

- Başlangıç bütçesi koşu başına 1.000 oracle sorgusu.
- İlk uygulama varsayılanı ayar başına 20 tekrar olacak. Bu düşük maliyetli başlangıç değeridir; kullanıcı tarafından kesin sayı ayrıca belirlenmedi. Büyük tekrar sayılarına otomatik geçilmeyecek.
- Önce gerçek görüntülerin embedding'leri üretilecek, ardından küçük bir gerçek veri pilotu ve başlangıç batch'i çalıştırılacak.
- Çok sayıda koşu bütçede başarısız kalırsa sorgu bütçesini artırma kararı sonuçlara göre verilecek. Önceden 10.000 sorguya veya 1.000 tekrara yükseltilmeyecek.
- Yöntemler aynı bütçe altında karşılaştırılacak. Farklı bütçeli batch'ler etiketlenmeden birleştirilmeyecek.

## 8. Katman 6 — Saldırılar ve baseline

İlk aşama:

1. Random-perturbation hill climbing: P başlangıcından normalize edilmiş pertürbasyonlarla görünen skor/seviyeyi iyileştirir.
2. Uphill-simplex: Pool başlangıçlarıyla simplex kurar; görünen skor/seviyeye göre reflection, expansion ve contraction uygular. Her aday normalize edilir.
3. Genetik algoritma: P'den 20 adayla başlar; görünen fitness üzerinden seçim, spherical crossover ve Gaussian mutation uygular.

- Bu üç saldırı float, 8-bit ve 4-bit üzerinde, iki eşikte çalışacak: `3 × 3 × 2 = 18 ayar`.
- Quantized saldırılar saklı float skoru kullanmayacak. Eşit seviyelerde davranış, yeniden başlatma ve adım ayarları config'te açık olacak.
- Decision-based ertelendiği için ona otomatik fallback ilk aşamada yapılmayacak.
- Zero-effort baseline P'nin mevcut embedding'lerini seed'li rastgele sırayla, değiştirmeden dener. İlk kabulde recovery ölçülüp durulur; kabul yoksa bütçe veya havuz bitiminde durur. Havuz büyüklüğü kişi değil mevcut embedding sayısıdır.
- Aynı zero-effort kabul kararları geri bildirim hassasiyetinden bağımsız olduğundan baseline her eşik için çalıştırılıp ilgili karşılaştırmalarda paylaşılabilir.

Decision-based sonraki aşamadır:

- Boundary/HSJA tarzı yöntem kabul edilen başlangıç vektörüne ihtiyaç duyar; esas arama bu başlangıçtan sonra yapılır.
- Başlangıcın bulunması/verilmesi, başlangıç maliyetinin hesabı, hedef kimlik ve kabul sonrası durdurma kuralı ayrıca kararlaştırılacak.
- Diğer saldırıların ilk kabulde durma kuralı bu yönteme otomatik uygulanmayacak.
- 1-bit saldırı deneyleri bu aşamaya bırakılacak; oracle'ın 1-bit desteği ilk aşamada bulunacak.
- Tam kapsam daha sonra `18 + 4 × 2 = 26 ayar` olabilir. İlk aşama için 26 ayarın tamamlandığı iddia edilmeyecek.

Savunmalar temel deneyler çalıştıktan sonra eklenecek: skor kabalaştırma karşılaştırması, hesap başına ve iç top-1 şablon başına sorgu sınırı, benzer sorgu dizilerini tespit etme. Ayrı development grubu kullanılmadığı için dedektör ayarları/verisi ve adaptive deney kapsamı bu aşamaya geçerken ayrıca kararlaştırılacak; G held-out ile ayar seçilmeyecek.

## 9. Katman 7 — Sonuçlar ve doğrulama

- Kabul ve ilk kabulde recovery başarıları ayrı raporlanacak; recovery oranında paydanın tüm koşular mı yalnızca kabul edilenler mi olduğu açık yazılacak.
- Sorgu sayısına karşı iki başarı eğrisi üretilecek; zero-effort karşılaştırması eklenecek.
- Başarısız koşular korunacak. Medyan/p90 bütçe içinde erişilmiyorsa `> budget / not reached` denecek; yalnız başarılı koşuların ortancası tüm koşuların sonucu gibi sunulmayacak.
- Güven aralıkları olmayacak; az tekrarın küçük farkları yorumlamayı sınırladığı belirtilecek.
- H baseline skor dağılımları ve DET/ROC çıktıları ile standart LFW doğrulama sonucu raporlanacak.
- Sonuçlar ölçülmüş kayıtlardan üretilecek; yayımlanan sonuçlar ve planlanan deneyler gerçek ölçümlerden ayrı tutulacak.

Kısa ve anlamlı kontroller: kişi ayrılığı, enrollment/held-out ayrılığı, embedding eşleşmesi ve normları, 1:N/1:1 skor doğruluğu, quantizer karar eşdeğerliği, özel bilgi sızmaması, tam sorgu sayımı ve ilk kabulde durma. Çok sayıda uygulamayı tekrar eden test veya gereksiz test altyapısı kurulmayacak.

## 10. Uygulama sırası

1. Ortam/config ve tek veri indeksi.
2. Embedding üretimi ve G/P/H rollerinin sonlandırılması.
3. NumPy doğrulayıcı, H kalibrasyonu, quantizer ve LFW baseline kontrolü.
4. Ortak döngü, zero-effort ve ilk kabulde recovery.
5. Random hill climbing ile uçtan uca küçük deney.
6. Uphill-simplex ve genetik algoritma; 18 ayarlık başlangıç batch'i.
7. Sonuçları inceleme; gerekirse sorgu bütçesini artırma.
8. Decision-based protokolünü ayrıca netleştirip uygulama.
9. Savunmalar ve final rapor.

Her aşama küçük çalışan bir sonuçla bitecek. Büyük deneyler, model indirmeleri veya yeni kapsamlar bu plan dosyalarını yazma isteğinin parçası olarak başlatılmayacak.
