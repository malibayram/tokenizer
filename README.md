# Tokenizer

Dilbilim kurallarını temel alarak, çok dilli metinleri işlemek ve anlam bütünlüğünü korumak için gelişmiş bir tokenizer altyapısı.

## İlk Versiyon
- [x] Kelime köklerinin ses olayına uğramış olan hallerinin ses olayına uğramamış olan halleri ile aynı id ile temsil edilmesi
- [x] İlkHarfBüyük tokeni oluşturulması ve tüm tokenlerin ilk harfinin küçük harfe çevrilmesi
- [x] Çoğul tokeni oluşturulması ve ler - lar eklerinin silinmesi
- [x] Tamamen aynı olan ama sesleri farklı olan eklerin özel tokenler ile temsil edilmesi

---

## Projenin Amacı ve Kapsamı

Bu projenin amacı, metin analizi ve doğal dil işleme (NLP) süreçlerinde kullanılabilecek, dilbilgisel yapıları ve anlam bütünlüğünü dikkate alan bir tokenizer geliştirmektir. Proje, Türkçe dilbilgisi kurallarını referans alarak başlamış olsa da, evrensel dil kuralları doğrultusunda çok dilli bir yapıya sahip olacak şekilde genişletilecektir.

## Temel Özellikler

- Dilbilim kurallarına dayalı tokenizasyon
- Morfolojik analiz desteği
- Çok dilli destek altyapısı
- Genişletilebilir mimari
- Yüksek performanslı işleme
- Paralel işleme desteği

## Mevcut Implementasyonlar

### Türkçe Morfolojik Tokenizer

`turkish_tokenizer` dizininde Python ve Rust ile geliştirilmiş, Türkçe morfolojik analiz yapabilen tokenizer'lar bulunmaktadır. Her iki implementasyon da aynı özelliklere sahiptir:

- Kök kelime ve ek tespiti
- Büyük harf hassasiyeti
- Türkçe karakter desteği
- BPE (Byte-Pair Encoding) yedekleme sistemi

#### Python Implementasyonu
```python
from turkish_tokenizer import tokenize

text = "Kitabı ve defterleri getirn, YouTube"
result = tokenize(text)
print(result)
```

#### Rust Implementasyonu
```bash
cd turkish_tokenizer
cargo build --release
./target/release/turkish_tokenizer "Kitabı ve defterleri getirn, YouTube"
```

Her iki implementasyon da aynı çıktıyı üretir:
```json
{
  "tokens": [
    "<UPCL>",
    "kitab",
    "ı",
    "ve",
    "defter",
    "ler",
    "i",
    "getir",
    "n",
    ",",
    "<UPCL>",
    "yo",
    "u",
    "<UPCL>",
    "tu",
    "be"
  ],
  "ids": [
    0,
    385,
    19936,
    19901,
    2001,
    19934,
    19935,
    159,
    19950,
    20022,
    0,
    643,
    19937,
    0,
    21941,
    21383
  ]
}
```

Detaylı bilgi ve kullanım kılavuzu için: [Turkish Tokenizer Documentation](turkish_tokenizer/README.md)

### Implementasyon Farklılıkları

1. **Performans**:
   - Rust implementasyonu `rayon` ile paralel işleme yapabilir
   - Rust versiyonu daha verimli bellek yönetimi sunar
   - Python versiyonu daha basit ve değiştirmesi kolaydır

2. **String İşleme**:
   - Rust UTF-8 farkındalıklı string işlemleri kullanır
   - Python doğal Unicode desteğine sahiptir

3. **Bellek Kullanımı**:
   - Rust sıfır maliyetli soyutlamalar kullanır
   - Python referans sayımı ve çöp toplama kullanır

## Geliştirme ve Katkıda Bulunma

Projeye katkıda bulunmak için:

1. Repository'yi fork edin
2. Yeni bir branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request açın

## Lisans

MIT

---

**Not:** Proje aktif geliştirme aşamasındadır. Detaylı dokümantasyon için [Wiki](wiki) sayfasını ziyaret edebilirsiniz.