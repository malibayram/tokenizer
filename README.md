# Tokenizer

Dilbilim kurallarını temel alarak, çok dilli metinleri işlemek ve anlam bütünlüğünü korumak için gelişmiş bir tokenizer altyapısı.

## İlk Versiyon
- [x] Kelime köklerinin ses olayına uğramış olan hallerinin ses olayına uğramamış olan halleri ile aynı id ile temsil edilmesi
- [x] İlkHarfBüyük tokeni oluşturulması ve tüm tokenlerin ilk harfinin küçük harfe çevrilmesi
- [x] Çoğul tokeni oluşturulması ve ler - lar eklerinin silinmesi
- [x] Tamamen aynı olan ama sesleri farklı olan eklerin özel tokenler ile temsil edilmesi
- [x] Boşluk, satır sonu ve tab karakterlerinin özel tokenler ile temsil edilmesi

## Gelecek Özellikler
- [ ] Çok dilli destek
- [ ] Performans optimizasyonları
- [ ] Daha kapsamlı test senaryoları
- [ ] Web API desteği
- [ ] Docker entegrasyonu

---

## Projenin Amacı ve Kapsamı

Bu projenin amacı, metin analizi ve doğal dil işleme (NLP) süreçlerinde kullanılabilecek, dilbilgisel yapıları ve anlam bütünlüğünü dikkate alan bir tokenizer geliştirmektir. Proje, Türkçe dilbilgisi kurallarını referans alarak başlamış olsa da, evrensel dil kuralları doğrultusunda çok dilli bir yapıya sahip olacak şekilde genişletilecektir.

## Temel Özellikler

- Dilbilim kurallarına dayalı tokenizasyon
- Morfolojik analiz desteği
- Çok dilli destek altyapısı
- Genişletilebilir mimari
- Yüksek performanslı işleme
- Özel karakter ve boşluk işleme desteği

## Dosya Yapısı

Tokenizer üç temel sözlük dosyası kullanır:
- `kokler_v05.json`: Kök kelimeler ve özel tokenler (0-20000 arası ID'ler)
- `ekler_v05.json`: Ekler (22268-22767 arası ID'ler)
- `bpe_v05.json`: BPE token'ları

### Özel Tokenler
```json
{
    "<uppercase>": 0,    // Büyük harf işareti
    "<space>": 1,       // Boşluk karakteri
    "<newline>": 2,     // Satır sonu
    "<tab>": 3,         // Tab karakteri
    "<unknown>": 4      // Bilinmeyen token
}
```

## Kullanım

### Python Implementasyonu
```python
from turkish_tokenizer import tokenize

text = "Kitabı ve defterleri getirn,\nYouTube\t"
result = tokenize(text)
print(result)
```

### Rust Implementasyonu
```rust
use turkish_tokenizer::TurkishTokenizer;

fn main() {
    let mut tokenizer = TurkishTokenizer::new().unwrap();
    let text = "Kitabı ve defterleri getirn,\nYouTube\t";
    let result = tokenizer.tokenize(text).unwrap();
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
```

## Implementasyon Özellikleri

### Python Versiyonu
1. **Temel Özellikler**:
   - Basit ve anlaşılır kod yapısı
   - Kolay entegrasyon
   - Hızlı prototipleme için uygun
   - Dinamik tip sistemi

2. **Performans Özellikleri**:
   - Sıralı işleme
   - Bellek dostu veri yapıları
   - Yorumlanmış dil avantajları

### Rust Versiyonu
1. **Temel Özellikler**:
   - Güvenli bellek yönetimi
   - Statik tip sistemi
   - Thread-safe veri yapıları
   - Sıfır maliyetli soyutlamalar

2. **Performans Özellikleri**:
   - Paralel işleme desteği (Rayon)
   - Verimli UTF-8 karakter işleme
   - Düşük seviye optimizasyonlar
   - Önbellekleme mekanizmaları

3. **Teknik Detaylar**:
   - Arc ile thread-safe paylaşımlı veri
   - Regex ile gelişmiş kelime bölümleme
   - Lazy static ile verimli statik kaynaklar
   - Zero-copy string işlemleri

## Geliştirme ve Katkıda Bulunma

### Geliştirme Ortamı Kurulumu

1. Repository'yi klonlayın:
```bash
git clone <repository-url>
cd tokenizer
```

2. Python ortamını hazırlayın:
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# veya
.\venv\Scripts\activate  # Windows
```

3. Rust toolchain'i kurun:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# veya
rustup update
```

### Geliştirme Süreci

1. Yeni bir branch oluşturun:
```bash
git checkout -b feature/yeni-ozellik
```

2. Testleri çalıştırın:
```bash
# Python testleri
python -m pytest tests/

# Rust testleri
cargo test
```

3. Kod stilini kontrol edin:
```bash
# Python
flake8 .
black .

# Rust
cargo fmt
cargo clippy
```

4. Değişikliklerinizi commit edin:
```bash
git add .
git commit -m "feat: yeni özellik eklendi"
```

### Pull Request Süreci

1. Branch'inizi push edin:
```bash
git push origin feature/yeni-ozellik
```

2. GitHub üzerinden pull request açın
3. Code review sürecini takip edin
4. Gerekli düzeltmeleri yapın
5. PR'ınız onaylandığında main branch'e merge edilecektir

### Geliştirme Gereksinimleri

#### Python
- Python 3.6+
- pytest
- black
- flake8
- JSON desteği
- UTF-8 karakter desteği

#### Rust
- Rust 1.50+
- Cargo paket yöneticisi
- rustfmt
- clippy
- Bağımlılıklar:
  - serde (JSON işleme)
  - rayon (paralel işleme)
  - regex (kelime bölümleme)
  - lazy_static (statik kaynaklar)

## Lisans

MIT

---

**Not:** Proje aktif geliştirme aşamasındadır. Detaylı dokümantasyon için [Wiki](wiki) sayfasını ziyaret edebilirsiniz.