---
license: cc-by-nc-4.0
language:
- tr
---

# TR Tokenizer README

## Tokenizer Summary
TR Tokenizer, Türkçe kelimeleri anlam bütünlüklerine göre parçalara bölen, hem güncel doğal dil işleme yöntemlerini hem de Türk dilbilgisi kurallarını kullanan yenilikçi bir araçtır. Bu tokenizer, kelimeleri morfolojik ve anlamsal olarak analiz ederek doğru ve detaylı sonuçlar sağlar. Örneğin, "akademisyenler ve aileleri ile birlikte aktif çalışıyorlar" cümlesi, şu parçalara ayrılır:

```json
[‘akademisyen’, ‘ler’, ‘ve’, ‘aile’, ‘leri’, ‘ile’, ‘birlikte’, ‘aktif’, ‘çalış’, ‘ı’, ‘yor’, ‘lar’]
```

## Supported Tasks and Applications
TR Tokenizer, aşağıdaki NLP görevleri için kullanılabilir:
- **Morfolojik Ayrıştırma**: Kelimelerin kök ve ek yapılarını ayırarak analiz eder.
- **Dil Modeli Eğitimi ve İnce Ayarı**: Türkçe dil modeli eğitiminde ön işleme aşamasında kelimeleri anlam bütünlüklerine göre işler.
- **Frekans Analizi**: Metinlerdeki kelime frekanslarının tespitinde yardımcı olur.
- **Doğal Dil İşleme (NLP) Araştırmaları**: Türkçe dilin morfolojik yapısını ve kelime yapılarını inceleyen araştırmalarda kullanılır.

## Languages
Bu tokenizer, **Türkçe** diline odaklanır ve Türkçe'nin zengin morfolojik yapısını desteklemek için tasarlanmıştır.

## Tokenizer Details
TR Tokenizer, kelimeleri kök ve eklerine ayırmak için Türk dilbilgisi kurallarını ve güncel NLP yöntemlerini birleştirir. Tokenizer, önceden tanımlanmış kelime ve ek listeleriyle eğitilmiştir ve Türkçe'nin anlam bütünlüğünü koruyarak kelimeleri analiz eder.

### Örnek Kullanım
```python
from tr_tokenizer import TRTokenizer

tokenizer = TRTokenizer()
sentence = "akademisyenler ve aileleri ile birlikte aktif çalışıyorlar"
tokens = tokenizer.tokenize(sentence)
print(tokens)
# Çıktı: ['akademisyen', 'ler', 've', 'aile', 'leri', 'ile', 'birlikte', 'aktif', 'çalış', 'ı', 'yor', 'lar']

## Licensing Information
TR Tokenizer, CC BY-NC 4.0 lisansı ile sunulmuştur ve ticari olmayan amaçlarla serbestçe kullanılabilir. Kullanıcılar, bu tokenizer'ı araştırma ve eğitim amaçları için kullanabilir, ancak ticari kullanımlar için ek izin gerekmektedir.

## Citation Information
Bu tokenizer'ı kullanan araştırmacıların aşağıdaki gibi atıfta bulunmaları önerilir:

```bibtex
@software{bayram_2024_tr_tokenizer,
  author = {Bayram, M. Ali},
  title = {{TR Tokenizer: Anlam Bütünlüklerine Göre Türkçe Kelime Ayrıştırma Aracı}},
  year = 2024,
  publisher = {Hugging Face},
  url = {https://huggingface.co/alibayram/tr_tokenizer}
}

## Contact and Contributions
Proje hakkında daha fazla bilgi almak, katkıda bulunmak veya geri bildirim sağlamak için [Ali Bayram](https://github.com/malibayram) ile iletişime geçebilirsiniz. Her türlü geri bildirim ve katkı, projenin gelişimi açısından değerlidir.

Katkıda bulunmak isteyenler, GitHub deposunu ziyaret ederek projeye katkıda bulunabilirler. Projenin geliştirilmesine ve açık kaynak topluluğuna katkılarınızdan dolayı şimdiden teşekkür ederiz!