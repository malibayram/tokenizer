# Ocak 2025 Tokenizer Projesi İçin Strateji 

### Token Sınıflandırma

Bu aşamada beş token tipimiz var:

    1. Kök
    2. Ek
    3. Semantik Token
    4. Other
    5. Unknown 

Şu an tüm ekleri semantik olarak sınıflandırmadığımız için ekler ve semantik tokenler olarak iki ayrı sınıfımız var. Ve semantik tokenler sınıfında şu an kesin olarak ayırdıklarımız "çoğulluk" ve "bulunma" ekleri, bunları semantik token olarak ayırdık. Diğer tüm ekler "ek" olarak sınıflandırıldı. Ayrıca ilk 3 maddeye ait olmayan kelime parçacıkları için "Other" sınıfı ve bilinmeyen kelimeler için "Unknown" tokeni var.

### Kelimeyi Kök ve Eklerine Ayırma

Bu bölümde adımları şöyle sıralayabiliriz:

    1. Baştan en uzun kökü bulma
    2. Kökü bulduktan sonra kalan kısmı ek olarak ayırma
    3. Varsa semantik tokenleri belirleme ve ek kısmını bunlara göre bölme
    4. Ek kısımlarını en uzun ekleri bularak anlamlı eklere ayırma

##### Örnek

*Burnumdakilerden* kelimesini ele alalım:

    1. Burnumdakilerden -> Burn (kök) + umdakilerden (ekler)
    2. Burn (kök) + umdakilerden (ekler) -> Burn (kök) + um (ek) + da (semantik token -bulunma-) + ki (ek) + ler (semantik token -çoğul-) + den (ek)

Bu örnekte şansımıza semantik tokenlerden ayırmak her bir ek grubunu bir eke düşürmemizi sağladı ama aksi durumlarda şunu yapabiliriz:

    1. Kitapçılıkta -> Kitap (kök) + çılıkta (ekler)
    2. Kitap (kök) + çılıkta (ekler) -> Kitap (kök) + çılık (ekler) + ta (semantik token -bulunma-)
    3. Kitap (kök) + çılık (ekler) + ta (semantik token -bulunma-) -> Kitap (kök) + çı (ek) + lık (ek) + ta (semantik token -bulunma-)

Bu aşamaları daha detaylı anlatmak istediğimizde ortaya şöyle bir şey çıkar:

1. aşamada kelimenin kökünü bulmak için sondan başlayarak kelimenin kök listesinde bir eşleşmesi olup olmadığına bakılır.
Örneğin *Burnumdakilerden* kelimesi için direkt kelime kök listesinde mi diye baktık. Sonra sondan bir harf çıkarıp (Burnumdakilerde) tekrar kontrol ettik. 
Olumsuz dönüş aldığımız için bir harf daha çıkardık ve bu şekilde kök listesiyle ilk eşleşmede durduk ve kelimeyi kök ve ekler olarak ayırdık.

2. aşamada ise direkt olarak ekler kısmında herhangi bir semantik token var mı diye baktık, eğer varsa ekler kısmını semantik tokenden split ettik.

3. aşamada ise ekler kısmını en uzun ekleri bulmaya çalıştık. Köklerde yaptığımız gibi sondan başlayarak ekler listesinde bir eşleşme olup olmadığına baktık. Eğer varsa ekleri ayırdık.

### Ek
    
Ayrıca bu işlemlerin hepsinde başarılı olmamızın sebebi şu an elimizde bulunan kök-id listesine her kökün -uğruyorsa- ses olaylarına uğramış hallerini ekleyip orijinal kök ile aynı id'ye atadık böylece *burun* ve *burn* kelimeleri aynı id'ye sahip olacak ve bu sayede *burnumdakilerden* kelimesinde *burn* kısmı kök olarak bulunabilecek ve embeddingte *burun* ile aynı anlama sahip olacak. 