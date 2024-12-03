# Tokenizer Benchmark

### Bu klasör bir tokenizer benchmarkı oluşturma ve bu yolla tokenizer standartları oluşturma amaçlı çalışmaları içerir.

## Benchmark Kriterleri 

1. Verilen metnin kaç tokene dönüştürüldüğü. 
2. Ne kadar hızlı dönüştürüldüğü.
3. Dönüştürülen tokenlerin kaçının anlamlı tokenler olduğu. 
4. Dönüştürülen tokenlerin kaçının anlam kaybı olmadan daha küçük tokenlerle ifade edilebileceği.
5. Token sözlüğünün boyutu. 

## Maddelerin Uygulanması

*İlgili maddelerin uygulanması ile ilgili fikirler.*

- Tokenlerin anlamlı olup olmadığı isTurkish() fonksiyonu ile kontrol edilebilir. 
- Eğer token bir predefined_word ise isTurkish() fonksiyonunu çalıştırmaya gerek yok.