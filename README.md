# Biometrics Reconstructed

Bu ders projesinde yüz fotoğraflarını buffalo_l modeliyle 512 sayılık vektörlere çevirip normalize ediyoruz. Kişileri G/P/H olarak ayırıyoruz: G sisteme kayıtlı hedefler, P saldırıların başlangıç havuzu, H kabul eşiklerini seçtiğimiz kişiler. Algoritmalar P'den başlayıp sisteme aday vektör gönderiyor; ilk kabulün kaçıncı sorguda geldiğini kaydediyoruz. Kabul edilen aynı vektörün, eşleşen kişinin diğer fotoğraflarına da benzeyip benzemediğini ayrıca ölçüyoruz. Buna recovery diyoruz.

## Kurulum

Python 3.11 veya 3.12 ortamında bağımlılıkları kur:

```powershell
python -m pip install -r requirements.txt
```

InsightFace 0.7.3 Windows'ta C++ derleme araçları gerektirebilir. Config'teki model kökünün altında `models/buffalo_l/w600k_r50.onnx` ve `models/buffalo_l/det_10g.onnx` dosyaları bulunmalı. Kod model indirmiyor. Bütün ayarlar `config.json` içinde; komutlara argüman vermiyoruz.

| Girdi | Config'teki mevcut yol |
|---|---|
| Görüntüler | `C:\dev\biometrics-project\biyometriData\lfw-deepfunneled\lfw-deepfunneled` |
| Model kökü | `C:\dev\biometrics-project\.insightface` |
| LFW çiftleri | `C:\dev\biometrics-project\biyometriData\pairs.csv` |

## Çalıştırma

```powershell
python prepare.py
python calibrate.py
python experiment.py
python report.py
```

`prepare.py` görüntü indeksini oluşturup embedding'leri çıkarıyor. İndeks varsa onu kullanıyor; hazır embedding deposunun üzerine yazmıyor. G/P/H kişi sayıları 1.000/2.000/2.500. G'deki her kişinin bir vektörü kayıt olarak seçiliyor, diğerleri recovery değerlendirmesine ayrılıyor. Yüz bulunamayan görüntüler işaretleniyor; gerektiğinde yeterli vektörü kalmayan G kişisi uygun bir P kişisiyle değiştiriliyor. Baştan hazırlık yapmak istersen config'teki work çıktı yollarını yeni bir klasöre yönlendir.

`calibrate.py` hangi skorun kabul sayılacağını hesaplıyor. H'den 1.000 kişilik bir galeri seçip geride kalan kişileri bu galeriye sorgulatıyor. Her sorgunun en yüksek skorunu topluyor ve bunu 200 galeri seçimiyle tekrarlıyor. Böylece kayıtlı olmayan kişileri yaklaşık %1 ve %0,1 oranında yanlış kabul eden iki eşik seçiliyor. Recovery için de H'deki farklı kişilerin tek tek vektör karşılaştırmalarından ayrı bir eşik çıkarılıyor. Veriyi veya bu ayarları değiştirirsen kalibrasyonu yeniden çalıştır.

`experiment.py` seçilen yöntemleri P'deki vektörlerden başlatıyor. Her aday, G'nin kayıtlarıyla karşılaştırılıyor; algoritmaya yalnız seçili hassasiyetteki cevap dönüyor. İlk kabulde koşu bitiyor ve aynı vektör için recovery sonucu kaydediliyor. Başlangıç adayları, tekrar sorguları ve yeniden başlamalar bütçeye dahil. Kabul bulamayan koşular da sonuç dosyasında kalıyor.

`report.py` en son deneyin kabul/recovery oranlarını, sorgu sayılarını ve başarı eğrilerini üretiyor. H histogramı varsa onun dağılım, ROC ve DET grafiklerini de çiziyor. Modelin standart LFW doğruluğunu saldırılardan ayrı ölçmek için embedding üretiminden sonra `python lfw_check.py` çalıştırılabilir; bu 6.000 çift ve on bölüm üzerinden yapılan bir değerlendirme.

## Deney ayarları

Varsayılan seçim hill yöntemi, float cevap ve %1 hedef FMR; koşu başına 1.000 sorgu ve 20 tekrar. Config'in `experiment` bölümü şöyle:

```json
{
  "seed": 2026,
  "query_budget": 1000,
  "repetitions": 20,
  "methods": ["hill"],
  "granularities": ["float"],
  "targets": [0.01]
}
```

18 ayarın tamamı için listeleri `["hill", "simplex", "genetic"]`, `["float", "8", "4"]`, `[0.01, 0.001]` yap. Aynı tekrarlar aynı P başlangıç sırasını kullanıyor. Zero-effort baseline, P vektörlerini değiştirmeden deniyor; her eşik için bir kez çalıştırılıp hassasiyet karşılaştırmalarında kullanılıyor. Float cevap gerçek skor, 8-bit ve 4-bit cevaplar skorun düştüğü kutunun numarası. Kutulama kabul kararını koruyor.

## Dosyalar ve sonuçlar

İndeks `work/data_index.csv`, vektörler ve görüntü kimlikleri `work/embeddings.npz`, kısa model/veri bilgileri `work/metadata.json` içinde. Eşikler `work/calibration.json`, H dağılım sayımları `work/h_score_histogram.csv` dosyasına yazılıyor. Her deneyin `results/pilot-.../` klasöründe kullanılan ayarlar `resolved.json`, koşular `private_records.jsonl` içinde saklanıyor. CSV özetleri ve PNG grafikler bu kayıtlardan üretiliyor. Recovery hem bütün koşulara hem kabul edilen koşulara göre raporlanıyor; medyan veya p90'a bütçede ulaşılamadıysa bu belirtiliyor.

Fonksiyonların açıklamaları [Türkçe kod rehberinde](REHBER_TR.md). `common.py` ortak dosya ve vektör işlemlerini, `attacks.py` aday üretimini, `comparator.py` karşılaştırma ve sorgu döngüsünü içeriyor. Diğer Python dosyaları yukarıdaki komutlarla çalıştırılan aşamalar.

## Mevcut durum

İndeks 13.233 görüntü ve 5.749 kişiyi içeriyor. Yerel model dosyaları tamamlanmadığı için gerçek embedding üretimi ve sonraki deneyler henüz çalıştırılmadı. İncelenen diğer ortam InsightFace 2.1 içeriyor; proje için requirements.txt içindeki 0.7.3 ortamı kurulmalı. Karar tabanlı saldırılar, 1-bit saldırı deneyleri ve savunmalar sonraki aşamalara bırakıldı. Model eğitimi veya yüz görüntüsü üretimi yapılmıyor. Ham görüntüleri ve ağırlıkları dağıtmadan, kaynaklarının kullanım koşullarına uygun yerel ders deneyi olarak kullanılmalı.
