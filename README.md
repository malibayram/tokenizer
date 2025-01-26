# Tokenizer

Dilbilim kurallarını temel alarak, çok dilli metinleri işlemek ve anlam bütünlüğünü korumak için gelişmiş bir tokenizer altyapısı.

## İlk Versiyon
- [ ] Kelime köklerinin ses olayına uğramış olan hallerinin ses olayına uğramamış olan halleri ile aynı id ile temsil edilmesi
- [ ] İlkHarfBüyük tokeni oluşturulması ve tüm tokenlerin ilk harfinin küçük harfe çevrilmesi
- [ ] Çoğul tokeni oluşturulması ve ler - lar eklerinin silinmesi
- [ ] Tamamen aynı olan ama sesleri farklı olan eklerin özel tokenler ile temsil edilmesi

---

## Projenin Amacı ve Kapsamı

Bu projenin amacı, metin analizi ve doğal dil işleme (NLP) süreçlerinde kullanılabilecek, dilbilgisel yapıları ve anlam bütünlüğünü dikkate alan bir tokenizer geliştirmektir. Proje, Türkçe dilbilgisi kurallarını referans alarak başlamış olsa da, evrensel dil kuralları doğrultusunda çok dilli bir yapıya sahip olacak şekilde genişletilecektir.

## Temel Özellikler

- Dilbilim kurallarına dayalı tokenizasyon
- Morfolojik analiz desteği
- Çok dilli destek altyapısı
- Genişletilebilir mimari
- Yüksek performanslı işleme

## Mevcut Implementasyonlar

### Türkçe Morfolojik Tokenizer (Rust)

`turkish_tokenizer` dizininde Rust ile geliştirilmiş, Türkçe morfolojik analiz yapabilen bir tokenizer bulunmaktadır. Bu tokenizer:

- Kök kelime ve ek tespiti
- Büyük harf hassasiyeti
- Türkçe karakter desteği
- BPE (Byte-Pair Encoding) yedekleme sistemi

gibi özelliklere sahiptir.

Detaylı bilgi ve kullanım kılavuzu için: [Turkish Tokenizer Documentation](turkish_tokenizer/README.md)

Örnek kullanım:
```bash
cd turkish_tokenizer
cargo build --release
./target/release/turkish_tokenizer "Kitabı ve defterleri getirn, YouTube"
```

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