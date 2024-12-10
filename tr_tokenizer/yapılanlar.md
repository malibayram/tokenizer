1. [Kelime Böl ve Kök-Ek Ayrıştırma `kelime_bol.ipynb`](kelime_bol.ipynb)
```python
# %%
print("hello")

# %%
YUMUSAT = {'ç': 'c', 't': 'd', 'p': 'b', 'k': 'ğ', 'g':'ğ'}
SERTLES = {'c': 'ç', 'd': 't', 'b': 'p', 'g': 'k'}
SESLILER = {'ı', 'O', 'e', 'o', 'u', 'Ö', 'Ü', 'I',
            'ü', 'E', 'A', 'a', 'İ', 'ö', 'U', 'i'}
KALINSESLILER = {'ı', 'O', 'o', 'u', 'I', 'A', 'a', 'U'}
INCESESLILER = {'i', 'Ö', 'e', 'Ü', 'E', 'İ', 'ö', 'ü'}
kokler_dict = {}
ekler_dict = {}
dusenler = {}

# %%
"""
sonsesli() ve kontrol_simdiki_zaman() fonksiyonlarını ekledim
"""
def sonsesli(s):
    len_s = len(s)
    for i in range(len_s, 0, -1):
        if s[-i] in SESLILER:
            return s[-i]
    return None

def kok_fiil_mi(kok):
    return (kok in kokler_dict.keys()) and ('FI' in kokler_dict[kok])



# %%
def kontrol_simdiki_zaman(kelime):
    pos = kelime.find('yor')
    arases = kelime[pos-1]
    if arases not in ["ı","i","u","ü"]:
        return None,None

    kok = kelime[:pos-1]
    if len(kok)==4 and kok=='söyl':
        kok='söyle'
    elif len(kok)>2:
        # olumsuzlama eki olup olmadığını kontrol et
        if (kelime[pos-2]=='m') and (kelime[pos-3] in ['e','a']):
            pos1 = kelime.find('emiyor')
            if pos1>0:
                pos = pos1
                kok = kelime[:pos]
                if kok=='d':
                    kok='de'
                    pos+=1
                elif kok=='y':
                    kok='ye'
                    pos+=1
                elif kok =='diy':
                    kok = 'de'
                    pos=1
                elif kok =='yiy':
                    kok = 'ye'
                    pos=1
                if kok_fiil_mi(kok)==False:
                    pos+=1
                    kok =kelime[:pos]
                    pos+=1
                else:
                    pos+=1
            else:
                pos1 = kelime.find('amıyor')
                if pos1>0:
                    pos = pos1
                    kok = kelime[:pos]
                    if kok_fiil_mi(kok)==False:
                        pos+=1
                        kok =kelime[:pos]
                        pos+=1
                    elif kok_fiil_mi(kok+'a')==True:
                        kok+='a'
                        pos+=2
                    else:
                        pos+=1
    elif len(kok)==2:
        if kok=='ed': kok='et'
    if kok == 'diy': kok = 'de'
    elif kok == 'yiy': kok = 'ye'
    elif kok=='d': kok='de'
    elif kok=='y': kok = 'ye'
    ek = kelime[pos-1:]
    # ek sözlükte varsa işlemlere devam et
    if ek in ekler_dict.keys():
        if 'FI' in ekler_dict[ek]:
            if kok in kokler_dict.keys():
                if 'FI' in kokler_dict[kok]:
                   return kok,ek
    if ek in ekler_dict.keys():
        sses = sonsesli(kok)
        if sses in KALINSESLILER:
            if arases=='ı' or arases=='u':
                kok1=kok+'a'
                if kok1 in kokler_dict.keys():
                    return kok1, ek
        elif sses in INCESESLILER:
            if arases=='i' or arases=='ü':
                if kok not in ['de','ye']:
                    kok1=kok+'e'
                else:
                    kok1 = kok
                if kok1 in kokler_dict.keys():
                    return kok1,ek
    return None,None


# %%
def kokoku():
    for ab in [1,2]:
        if ab==1:
            ff = "veri/KOKLER.txt"
        else:
            ff = "veri/KOKOZLER.txt"
        with open(ff, "r", encoding="utf-8") as f:
            for sat in f.readlines():
                sat = sat.strip().split()
                len_sat = len(sat)
                if len_sat <= 0:
                    continue
                kelime = sat[0].strip()
                try:
                    tip = sat[1].strip()
                except:
                    print("Hatalı kelime: ",kelime)

                for ek in range(2, len_sat):
                    tip += ' ' + sat[ek].strip()

                if kelime in kokler_dict.keys():
                    if tip not in kokler_dict[kelime]:
                        kokler_dict[kelime] += ' '+tip
                else:
                    kokler_dict[kelime] = tip

                # Ünlü düşmesi varsa, gereğini yap
                if 'DUS' in tip:
                    if len(kelime) > 2:
                        kelime0 = kelime
                        kelime = kelime[:-2]+kelime[-1]
                        dusenler[kelime0] = kelime
                        if kelime not in kokler_dict.keys():
                            kokler_dict[kelime] = tip + ' DUS'
                        else:
                            if 'DUS' not in kokler_dict[kelime]:
                                kokler_dict[kelime] = tip + ' DUS'


# %%
def ekoku():
    with open("veri/EKLER.txt", "r", encoding="utf-8") as f:
        for sat in f.readlines():
            sat = sat.strip().split()
            if len(sat) >= 2:
                kelime = sat[0].strip()
                tip = sat[1].strip()
                if kelime in ekler_dict.keys():
                    if tip not in ekler_dict[kelime]:
                        ekler_dict[kelime] += ' '+tip
                else:
                    ekler_dict[kelime] = tip

def ekkaydet():
    with open("veri/EKLER2.txt", "w", encoding="utf-8") as f:
        for e in ekler_dict.keys():
            print("{} {}".format(e, ekler_dict[e]), file=f, flush=True)


# %%
def uzatma_temizle(kelime):
    if kelime in ['www','ııı','vııı','xııı']:
        return kelime
    k=''
    kk=kelime+'.'
    while len(kk)>1:
        c=kk[0]
        k+=c
        for say in range(1,len(kk)+1):
            if kk[say]!=c:
                kk = kk[say:]
                break
        if say ==2: k += c

    return k

# %%
def kok_tara(kelime1):
    """
    İşlev: Verilen kelimeyi kök ve ek adaylarına ayırır. En uzun kökü döndürür.
    Return: tamam ve kök ikilisini döndürür. Kök tipi stringdir. tamam ise kök
    ve ek adaylarından oluşan bir listedir.
    """
    if len(kokler_dict)<1: kokoku()
    if len(ekler_dict)<1: ekoku()
    koklen = len(kokler_dict)
    eklen = len(ekler_dict)
    tamam = []
    tip=''

    # Kelime uzatmalarla deforme edilmişse, orijinal haline çevir
    kelime = uzatma_temizle(kelime1)
    # kelime kök listesinde varsa işlem tamam
    if kelime in kokler_dict.keys():
        tamam.append(kelime)
        tip = kokler_dict[kelime]
        return tamam, kelime, tip
    elif kelime in ekler_dict.keys():
        #tamam.append("YALIN EK: "+kelime)
        #tip = ekler_dict[kelime]
        #return tamam, kelime, tip
        pass
    else:   # Kelime aslında bir ek ise
        #if kelime in ekler_dict.keys():
        #    tamam.append(":" + kelime)
        #    return tamam, None
        pass

    if len(kelime)<3: # kök de değil, ek de değil
        if tamam == []: tamam.append(kelime)
        return tamam, kelime, tip
    # Kelimenin tamamı kök sözlüğünde varsa başka şey aramaya gerek yok

    enuzunkok=''    # en uzun kökü belirlemek için
    basla=1
    # yor- eki varsa ses düşmesi ihtimalini değerlendir
    if "yor" in kelime:
        k1, e1 = kontrol_simdiki_zaman(kelime)
        if k1 != None and e1 != None:
            tamam.append(k1 + ":" + e1)
            tip='FI'
            # demek ve yemek fiilleri için özel durum
            if k1 in['de','ye']:
                enuzunkok = k1
                #return tamam, enuzunkok, tip
            basla=len(k1)
            if len(k1)>len(enuzunkok): enuzunkok=k1
            return tamam, enuzunkok, tip

    # de ve ye için özel durumlar
    if kelime[0] in ['d','y']:
        for dyek in ['iyebil','iyeceğ','iyecek','iyerek','iyeme','iyen','iyip','iyince']:
            if dyek in kelime:
                e1 = kelime[1:]
                if e1 in ekler_dict.keys():
                    k1 = kelime[0]+'e'
                    tamam.append(k1+':'+e1)
                    tip='FI'
                    if len(k1)>len(enuzunkok): enuzunkok=k1
                    return tamam, enuzunkok, tip

    ll = len(kelime)
    for i in range(basla,len(kelime)+1):
        l=ll-i
        kok = kelime[:l]
        ek = kelime[l:]
        tip=''
        etip=''
        # Önce eki kontrol et. Çünkü kökte değişiklik olabiliyor.
        if ek in ekler_dict.keys():
            etip=ekler_dict[ek]
            if kok in kokler_dict.keys():
                tip = kokler_dict[kok]
                if ('KIS' in tip or 'OZ' in tip) and ('IS' not in tip):
                    tip += ' IS'
                # Aynı ek farklı kök tiplerine bağlanıyor olabilir
                etipler = etip.split(' ')
                for etp in etipler:
                    if etp in tip: # tipler uyuşuyor -> işlem tamam
                        # Ünlü düşmesi varsa kontrol et
                        if 'DUS' in tip:
                            if kok in dusenler.values():
                                for kik in dusenler.keys():
                                    if dusenler[kik]==kok:
                                        kok=kik
                                        break
                        tamam.append(kok + ":" + ek)
                        if len(kok) > len(enuzunkok): enuzunkok = kok
                        return tamam, enuzunkok, tip
        if tip>'':
            # kökte yumuşama olmuş mu?
            if 'EKSI' in tip:
                try:
                    y = kok[-1]
                    if y in YUMUSAT.values():
                        for z in YUMUSAT.keys():
                            if YUMUSAT[z] == y:
                                kok = kok[:-1] + z
                                break
                except Exception as e:
                    print("YUM: {} {} : {}".format(kelime, kok, e))

            # kökte ünlü düşmüş olabilir. Kontrol et.
            if 'DUS' in tip:
                if kok in dusenler.keys():
                    kok = dusenler[kok]
        elif etip>'' and len(kok)>0:
            #try:
            y=kok[-1]
            if y in SERTLES.keys():
                for z in SERTLES.keys():
                    if z==y:
                        kok = kok[:-1]+SERTLES[z]
                        if kok in kokler_dict.keys():
                            # tipler uyuşuyor mu?
                            tip = kokler_dict[kok]
                            if 'YUM' in tip:
                                if ('KIS' in tip or 'OZ' in tip) and ('IS' not in tip):
                                    tip += ' IS'
                                etipler = etip.split(' ')
                                for etp in etipler:
                                    if etp in tip:  # tipler uyuşuyor -> işlem tamam
                                        tamam.append(kok + ":" + ek)
                                        if len(kok) > len(enuzunkok): enuzunkok = kok
                                        return tamam, enuzunkok, tip

            #except Exception as e:
            #    print("SERT: {} {} : {}".format(kelime, kok, e))
            # Yumuşama olmuş mu?
            y=kok[-1]
            if y in YUMUSAT.values():
                for z in YUMUSAT.keys():
                    if YUMUSAT[z]==y:
                        kok = kok[:-1]+z
                        if kok in kokler_dict.keys():
                            # tipler uyuşuyor mu?
                            tip = kokler_dict[kok]
                            if ('KIS' in tip or 'OZ' in tip) and ('IS' not in tip):
                                tip += ' IS'
                            etipler = etip.split(' ')
                            for etp in etipler:
                                if etp in tip:  # tipler uyuşuyor -> işlem tamam
                                    tamam.append(kok + ":" + ek)
                                    if len(kok) > len(enuzunkok): enuzunkok = kok
                                    return tamam, enuzunkok, tip

    if len(kok) > len(enuzunkok): enuzunkok = kok
    if tamam==[]: tamam.append(kelime)
    return tamam, enuzunkok, tip


# %%
ornek = "kalemi"
print(kok_tara(ornek))

# %%
kelime_dosya = "veri/Birleştirilmiş_Sözlük_Kelime_Listesi.txt"

with open(kelime_dosya, "r", encoding="utf-8") as f:
    satirlar = f.readlines()

len(satirlar), satirlar[:5]


# %%
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.tr")

# %%
import pandas as pd

df = pd.DataFrame(ds["train"])

df

# %%
satirlar = df["title"] + " " + df["text"]
len(satirlar), satirlar[:5]

# %%
%pip install cloudpickle

# %%
import sys
from multiprocessing import Pool
from multiprocessing.reduction import ForkingPickler
from types import FunctionType
import cloudpickle

assert sys.version_info >= (3, 8), 'python3.8 or greater required to use reducer_override'

def reducer_override(obj):
    if type(obj) is FunctionType:
        return (cloudpickle.loads, (cloudpickle.dumps(obj),))
    else:
        return NotImplemented

# Monkeypatch our function reducer into the pickler for multiprocessing.
# Without this line, the main block will not work on windows or macOS.
# Alterntively, moving the defintionn of foo outside of the if statement
# would make the main block work on windows or macOS (when run from
# the command line).
ForkingPickler.reducer_override = staticmethod(reducer_override)



# %%
if __name__ == '__main__':
    tum_kokler = set()

    def kok_ek_bul(satirlar):
      kokler = set()
      i = 0
      for satir in satirlar:
          i += 1
          satir = satir.strip()
          satir = satir.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
          while '  ' in satir:
              satir = satir.replace('  ', ' ')
          kelimeler = satir.split(' ')
          for kelime in kelimeler:
            kelime = kelime.strip()
            kelime = kelime.strip('.,?!')
            kelime = kelime.lower()
            if len(kelime) < 3:
                continue
            kok_tuple = kok_tara(kelime)
            if len(kok_tuple) < 3:
                continue
            kok = kok_tuple[1]
            kokler.add(kok)
          print(i, "Satır bitti")
          print("Kelime: {}, Kök: {}".format(kelime, kok))
      return kokler
    
    with Pool(40) as p:
        chunks = [satirlar[i:i+10000] for i in range(0, len(satirlar), 10000)]

        kokler = p.map(kok_ek_bul, chunks)
        for k in kokler:
            tum_kokler.update(k)
        print(tum_kokler)

# %%
len(tum_kokler)

# %%
kokler = set()
i = 0
for satir in satirlar:
    i += 1
    satir = satir.strip()
    satir = satir.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    while '  ' in satir:
        satir = satir.replace('  ', ' ')
    kelimeler = satir.split(' ')
    for kelime in kelimeler:
      kelime = kelime.strip()
      kelime = kelime.strip('.,?!')
      kelime = kelime.lower()
      if len(kelime) < 3:
          continue
      kok_tuple = kok_tara(kelime)
      if len(kok_tuple) < 3:
          continue
      kok = kok_tuple[1]
      kokler.add(kok)
    print(i, "Satır bitti")
    print("Kelime: {}, Kök: {}".format(kelime, kok))

len(kokler)

# %%
len(kokler)

# %%
with open("veri/kokler3.txt", "w", encoding="utf-8") as f:
    kok_sirali_liste = sorted(list(tum_kokler))
    for kok in kok_sirali_liste:
        print(kok, file=f, flush=True)

# %%
kokler1 = "veri/KOKLER.txt"
kokler2 = "veri/kokler2.txt"
kokler3 = "veri/kokler3.txt"

kokler_set = set()

with open(kokler1, "r", encoding="utf-8") as f:
    for satir in f.readlines():
        kokler_set.add(satir.strip().split(' ')[0])

with open(kokler2, "r", encoding="utf-8") as f:
    for satir in f.readlines():
        kokler_set.add(satir.strip().split(' ')[0])

with open(kokler3, "r", encoding="utf-8") as f:
    for satir in f.readlines():
        kokler_set.add(satir.strip())

len(kokler_set)

# %%
with open("veri/tum_kokler.txt", "w", encoding="utf-8") as f:
    kok_sirali_liste = sorted(list(kokler_set))
    for kok in kok_sirali_liste:
        print(kok, file=f, flush=True)



```
2. [İTÜ Kelime Böl ve Kök-Ek Ayrıştırma `itu_ek_kok_ayir.ipynb`](itu_ek_kok_ayir.ipynb)
```python
# %%
import requests

cookies = {
    'JSESSIONID': 'D3F87FDD7257B2052A21CE4D2305CDBE',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9,tr;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    # 'Cookie': 'JSESSIONID=D3F87FDD7257B2052A21CE4D2305CDBE',
    'Origin': 'http://tools.nlp.itu.edu.tr',
    'Referer': 'http://tools.nlp.itu.edu.tr/MorphAnalyzer',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
}


# %%
data = {
    'input': 'şıpsevdiler\r\nşıpıdıkçı\r\nşıpır\r\nşıpırtı\r\nşıpşıp\r\nşıra\r\nşırak\r\nşırakkadak\r\nşırfıntı\r\nşırla\r\nşırlağan\r\nşırıl\r\nşırıltı\r\nşırınga\r\nşıvgın',
    'output': '',
}

response = requests.post('http://tools.nlp.itu.edu.tr/MorphAnalyzer', cookies=cookies, headers=headers, data=data, verify=False)

# %%
from bs4 import BeautifulSoup

soup = BeautifulSoup(response.text, 'html.parser')

lines = soup.find('textarea', {'name': 'output'}).text.split('\n')

for line in lines:
    if line.strip() == '' or len(line.split()) < 2:
        continue
    kok = line.split()[1].split('+')[0]
    print(kok)

# %%
def kokleri_getir(text):
    data = {
        'input': text,
        'output': '',
    }
    response = requests.post('http://tools.nlp.itu.edu.tr/MorphAnalyzer', cookies=cookies, headers=headers, data=data, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    lines = soup.find('textarea', {'name': 'output'}).text.split('\n')
    kokler = set()
    for line in lines:
        if line.strip() == '' or len(line.split()) < 2:
            continue
        kok = line.split()[1].split('+')[0]
        kokler.add(kok)
    return kokler

# %%
kokler_set = set()

with open("veri/tum_kokler.txt", "r", encoding="utf-8") as f:
    for satir in f.readlines():
        kokler_set.add(satir.strip().split(' ')[0])

len(kokler_set)

# %%
yeni_kokler = set()

chunk_size = 161
chunks = [list(kokler_set)[i:i + chunk_size] for i in range(0, len(kokler_set), chunk_size)]

text = " ".join(chunks[10])
yeni_kokler_i = kokleri_getir(text)

len(yeni_kokler_i), len(chunks[10])

# %%
for i, chunk in enumerate(chunks):
    text = " ".join(chunk)
    yeni_kokler_i = kokleri_getir(text)
    print(i, len(yeni_kokler_i), len(chunk))
    yeni_kokler.update(yeni_kokler_i)

len(yeni_kokler)

# %%
with open("veri/after_itu_kokler.txt", "w", encoding="utf-8") as f:
    kok_sirali_liste = sorted(list(yeni_kokler))
    for kok in kok_sirali_liste:
        print(kok, file=f, flush=True)

# %%
```
3. [İTÜ isTurkish `itu_is_turkish.ipynb`](itu_is_turkish.ipynb)
```python
# %%
import requests

cookies = {
    'JSESSIONID': 'B7265EAA789B2654DBC7005531EFB13A',
}

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'en-US,en;q=0.9,tr;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded',
    # 'Cookie': 'JSESSIONID=B7265EAA789B2654DBC7005531EFB13A',
    'Origin': 'http://tools.nlp.itu.edu.tr',
    'Referer': 'http://tools.nlp.itu.edu.tr/IsTurkish',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
}


# %%
from bs4 import BeautifulSoup


# %%
data = {
    'input': 'şıpsevdiler\r\nşıpıdıkçı\r\nşıpır\r\nşıpırtı\r\nşıpşıp\r\nşıra\r\nşırak\r\nşırakkadak\r\nşırfıntı\r\nşırla\r\nşırlağan\r\nşırıl\r\nşırıltı\r\nşırınga\r\nşıvgın',
    'output': '',
}

response = requests.post('http://tools.nlp.itu.edu.tr/IsTurkish', cookies=cookies, headers=headers, data=data, verify=False)
response.status_code

# %%

soup = BeautifulSoup(response.text, 'html.parser')

# %%
lines = soup.find('textarea', {'name': 'output'}).text.split('\n')
lines

# %%
lines = soup.find('textarea', {'name': 'output'}).text.split('\n')

is_turkish_map = {}
for line in lines:
    if line.strip() == '' or len(line.split()) < 2:
        continue
    parts = line.split(':')
    word = parts[0].strip()
    
    is_turkish_map[word] = parts[1].startswith(' t')

is_turkish_map

# %%
def get_is_turkish(text):
  is_turkish_map = {}

  data = {
      'input': text,
      'output': '',
  }

  response = requests.post('http://tools.nlp.itu.edu.tr/IsTurkish', cookies=cookies, headers=headers, data=data, verify=False)

  soup = BeautifulSoup(response.text, 'html.parser')

  lines = soup.find('textarea', {'name': 'output'}).text.split('\n')

  for line in lines:
      if line.strip() == '' or len(line.split()) < 2:
          continue
      parts = line.split(':')
      word = parts[0].strip()

      is_turkish_map[word] = parts[1].startswith(' t')

  return is_turkish_map

# %%
kokler_set = set()

with open("veri/after_itu_kokler.txt", "r", encoding="utf-8") as f:
    for satir in f.readlines():
        kokler_set.add(satir.strip().split(' ')[0])

len(kokler_set)

# %%
chunk_size = 1361
chunks = [list(kokler_set)[i:i + chunk_size] for i in range(0, len(kokler_set), chunk_size)]

text = " ".join(chunks[10])

get_is_turkish(text)

# %%
is_turkish_map = {}

for i, chunk in enumerate(chunks):
    text = " ".join(chunk)
    is_turkish_map.update(get_is_turkish(text))
    print(f"{i}/{len(chunks)}")

len(is_turkish_map)

# %%
# turkish words

turkish_words = {k: v for k, v in is_turkish_map.items() if v}
len(turkish_words)

# %%
import pandas as pd

df = pd.DataFrame(is_turkish_map.items(), columns=['word', 'is_turkish'])
df.to_csv("veri/itu_turkish_words.csv", index=False, encoding="utf-8")
df

# %%
with open("veri/itu_turkish_kokler.txt", "w", encoding="utf-8") as f:
    kok_sirali_liste = sorted(list(turkish_words.keys()))
    for kok in kok_sirali_liste:
        print(kok, file=f, flush=True)

# %%



```
4. [Frekans Listesi Oluşturma `frekans_bul.ipynb`](frekans_bul.ipynb)
```python
# %%
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.tr")

# %%
import pandas as pd



# %%
df = pd.DataFrame(ds["train"])

df

# %%
satirlar = df["title"] + " " + df["text"]
len(satirlar), satirlar[:5]

# %%
import sys
from multiprocessing import Pool
from multiprocessing.reduction import ForkingPickler
from types import FunctionType
import cloudpickle

assert sys.version_info >= (3, 8), 'python3.8 or greater required to use reducer_override'

def reducer_override(obj):
    if type(obj) is FunctionType:
        return (cloudpickle.loads, (cloudpickle.dumps(obj),))
    else:
        return NotImplemented

# Monkeypatch our function reducer into the pickler for multiprocessing.
# Without this line, the main block will not work on windows or macOS.
# Alterntively, moving the defintionn of foo outside of the if statement
# would make the main block work on windows or macOS (when run from
# the command line).
ForkingPickler.reducer_override = staticmethod(reducer_override)



# %%
from kelime_bol import kok_tara

# %%
ornek = "kalemi"
print(kok_tara(ornek))

# %%
kelime_dosya = "veri/itu_turkish_kokler.txt"

kelime_freq = {}

with open(kelime_dosya, "r", encoding="utf-8") as f:
    satirla = f.readlines()
    for satir in satirla:
        kelime = satir.strip().replace("\n", "")
        kelime_freq[kelime] = 0

kelime_freq

# %%
if __name__ == '__main__':
    def kok_ek_bul(satirlar):
      kokler = dict()
      i = 0
      for satir in satirlar:
          i += 1
          satir = satir.strip()
          satir = satir.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
          while '  ' in satir:
              satir = satir.replace('  ', ' ')
          kelimeler = satir.split(' ')
          for kelime in kelimeler:
            kelime = kelime.strip()
            kelime = kelime.strip('.,?!')
            kelime = kelime.lower()
            if len(kelime) < 3:
                continue
            kok_tuple = kok_tara(kelime)
            if len(kok_tuple) < 3:
                continue
            kok = kok_tuple[1]
            if kok in kelime_freq:
                if kok not in kokler:
                    kokler[kok] = 1
                else:
                    kokler[kok] += 1
          print(i, "Satır bitti")
          print("Kelime: {}, Kök: {}".format(kelime, kok))
      return kokler
    
    with Pool(40) as p:
        chunks = [satirlar[i:i+10000] for i in range(0, len(satirlar), 10000)]

        kokler_n = p.map(kok_ek_bul, chunks)
        for k in kokler_n:
            # update frequencies kelimeler_dict
            for kok in k:
                if kok in kelime_freq:
                    kelime_freq[kok] += k[kok]


# %%
kelime_freq_df = pd.DataFrame(kelime_freq.items(), columns=["Kelime", "Frekans"])
kelime_freq_df = kelime_freq_df.sort_values(by="Frekans", ascending=False)
kelime_freq_df.to_csv("veri/kelime_freq.csv", index=False)
kelime_freq_df

# %%
kelime_freq_df = pd.read_csv("veri/kelime_freq.csv")
kelime_freq_df

# %%
first_1250 = kelime_freq_df[:12500]['Kelime'].tolist()
first_1250.sort()
first_1250[:10]

# %%
with open("veri/first_12500.txt", "w", encoding="utf-8") as f:
    for kelime in first_1250:
        f.write(kelime + "\n")

# %%
with open("veri/first_12500_titled.txt", "w", encoding="utf-8") as f:
    for kelime in first_1250:
        f.write(kelime + "\n")
        f.write(kelime.title() + "\n")

# %%



```
5. [Huggingface Tokenizer Oluşturma `hf_tokenizer.ipynb`](hf_tokenizer.ipynb)
6. [Huggingface FastTokenizer Oluşturma ve Yayınlama `hf_auto_tokenizer.ipynb`](hf_auto_tokenizer.ipynb)
```python
# %%
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# %%
# load gemma2 tokenizer from huggingface

from transformers import AutoTokenizer

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")



# %%
import pandas as pd
# get string, int pairs from gemma_tokenizer.vocab

gemma_vocab = gemma_tokenizer.get_vocab()

vocab_df = pd.DataFrame(gemma_vocab.items(), columns=['token', 'id'])
vocab_df = vocab_df.sort_values('id')
vocab_df[1000:1040]

# %%
gemma_tokenizer.vocab_size

# %%


# %%
vocab = {}

unk_token = '<unk>'
spl_tokens = ['<bos>', '<eos>', '<unk>', '<pad>', '<start_of_turn>', '<end_of_turn>']

vocab['<pad>'] = 0
vocab['<bos>'] = 1
vocab['<eos>'] = 2
vocab['<unk>'] = 3
vocab['<mask>'] = 4
vocab['<2mass>'] = 5
vocab['[@BOS@]'] = 6
for i in range(50):
    vocab[f'<unused{i}>'] = i + 7
vocab['<start_of_turn>'] = 57
vocab['<end_of_turn>'] = 58

vocab

#spl_tokens

# %%


# %%
predefined_words = []
with open("veri/first_12500_titled.txt") as f:
    for line in f:
        # remove newline character
        if line.strip() != "":
          predefined_words.append(line.strip())
predefined_words[:10]

# %%
ekler = [
    "lık", "lik", "luk", "lük",
    "lı", "li", "lu", "lü",
    "sız", "siz", "suz", "süz",
    "cı", "ci", "cu", "cü",
    "çı", "çi", "çu", "çü",
    "cık", "cik", "cuk", "cük",
    "çık", "çik", "çuk", "çük",
    "ca", "ce",
    "ça", "çe",
    "daş", "deş",
    "taş", "teş",
    "ncı", "nci", "ncu", "ncü",
    "ar", "er",
    "şar", "şer",
    "sal", "sel",
    "tı", "ti", "tu", "tü",
    "aç", "eç",
    "ak", "ek",
    "an", "en",
    "cıl", "cil", "cul", "cül",
    "çıl", "çil", "çul",
    "cileyin",
    "ç",
    "gil", "gül", "kıl", "kil",
    "ge", "ka",
    "kan", "ken",
    "kek",
    "man", "men",
    "la",
    "lak", "lek",
    "layın", "leyin",
    "msı", "msi", "msu",
    "mtırak",
    "rak", "rek",
    "sak", "sek",
    "sı", "si", "su", "sü",
    "şın", "şin",
    "t",
    "z",
    "la", "le",
    "al", "el",
    "l",
    "a", "e",
    "ar", "er",
    "da", "de",
    "k",
    "kır", "kir", "kur", "kür",
    "msa", "mse",
    "r",
    "sa", "se",
    "l",
    "ma", "me",
    "n",
    "r",
    "t",
    "ş",
    "dır", "dir", "dur", "dür",
    "tır", "tir", "tur", "tür",
    "a", "e",
    "ala", "ele",
    "ar", "er",
    "ı", "ü",
    "k",
    "k",
    "p",
    "a", "e",
    "ağan", "eğen",
    "ak", "ek",
    "alak", "elek",
    "amak", "emek",
    "anak", "enek",
    "ca", "ce",
    "ç",
    "aç", "eç",
    "dı", "di", "du", "dü",
    "tı", "ti", "tu", "tü",
    "ga", "ge",
    "gaç", "geç", "kaç",
    "gıç", "giç", "guç",
    "maca", "mece",
    "maç", "meç",
    "man", "men",
    "mık", "mik", "muk",
    "n",
    "t",
    "a", "e",
    "ga", "ge",
    "gıç", "giç", "guç",
    "n",
    "t",
    "lar", "ler", # Çokluk eki
    "mı", "mi", "mu", "mü", # Soru eki
    "m", "n", "ı", "i", "u", "ü", "sı", "si", "su", "sü", # İyelik ekleri
    "da", "de", "ta", "te", # Bulunma hali eki
    "dan", "den", "tan", "ten", # Ayrılma hali eki
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün", # İlgi hali eki
    "y", # Koruyucu ünsüz
    "n", # Koruyucu ünsüz
    "ki", # Aitlik eki
    "yor", "makta", "mede", # Şimdiki zaman ekleri
    "acak", "ecek", "acağ", "eceğ", # Gelecek zaman ekleri
    "r", "ar", "er", # Geniş zaman ekleri
    "malı", "meli", # Gereklilik kipi eki
    "sa", "se", # Dilek-şart kipi eki
    "a", "e", # İstek kipi eki
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü", # Görülen geçmiş zaman eki
    "mış", "miş", "muş", "müş", # Öğrenilen geçmiş zaman eki
    "im", "sin", "dir", "iz", "siniz", "dirler", # Ek-fiilin şimdiki/geniş zaman eki
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü", # Ek-fiilin görülen geçmiş zaman eki
    "mış", "miş", "muş", "müş", # Ek-fiilin öğrenilen geçmiş zaman eki
    "sa", "se", # Ek-fiilin şart eki
    "yacak", "yecek", # Katmerli birleşik çekim eki
    "miş", "miş", "muş", "müş", # Katmerli birleşik çekim eki
    "dı", "di", "du", "dü", "tı", "ti", "tu", "tü" # Katmerli birleşik çekim eki
]

print(len(set(ekler)))

# %%
edatlar = ["gibi", "için", "kadar", "ile", "üzere", "sanki", "den", "dolayı", "doğru", "önce", "sonra", "karşı", "göre", "rağmen", "beri", "üzere", "yönelik", "ait"]
bağlaçlar = ["ve", "ama", "ile", "de", "çünkü", "eğer", "fakat", "lakin", "ki", "ya", "yahut", "ya da", "veya", "hem", "oysa", "oysa ki", "nitekim", "sanki", "zira", "öyle ki", "şu var ki", "lâkin", "üstelik", "ne ... ne", "hem ... hem", "ister ... ister", "gerek ... gerek(se)", "ancak", "bari"]
ünlemler = ["ah", "ey", "oha", "öf", "hah", "ya", "aman", "hey", "hadi", "vah", "eyvah", "uff", "of", "oh", "heyecanla", "tamam", "çüş"]
cekim_ekleri = ["ler", "lar", "i", "e", "den", "de", "im", "in", "i", "imiz", "iniz", "leri", "ın", "in", "un", "ün", "ca", "ce", "le", "la", "r", "yor", "mekte", "di", "miş", "ecek", "meli", "malı", "se", "sa", "ayım", "ydi", "ymiş", "yse", "dir", "m", "n", "k", "iz", "siniz", "ler", "mak", "mek"]
yapim_ekleri = ["lık", "li", "sız", "ci", "ce", "daş", "üncü", "msı", "cık", "tı", "cıl", "deki", "la", "al", "l", "a", "ar", "da", "ık", "ımsa", "laş", "sa", "ca", "acak", "ak", "ge", "gı", "ı", "ıcı", "ık", "ım", "nç", "ıntı", "ır", "ış", "ma", "dır", "l", "n", "t"]

# doldurulacaklar = predefined_words + edatlar + bağlaçlar + ünlemler + cekim_ekleri + yapim_ekleri
# TypeError: can only concatenate list (not "tuple") to list
doldurulacaklar = set()
doldurulacaklar.update(predefined_words)
doldurulacaklar.update(edatlar)
doldurulacaklar.update(bağlaçlar)
doldurulacaklar.update(ünlemler)
doldurulacaklar.update(cekim_ekleri)
doldurulacaklar.update(yapim_ekleri)
doldurulacaklar.update(ekler)
doldurulacaklar = list(doldurulacaklar)
len(doldurulacaklar)

# %%
25072

# %%
tokenizer = Tokenizer(BPE(unk_token=unk_token))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.add_tokens(list(doldurulacaklar))
tokenizer

# %%
trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=7001, limit_alphabet=1200, min_frequency=16)

trainer

# %%
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.tr")
df = ds['train'].to_pandas()
df

# %%
texts_with_title = df['title'] + " " + df['text']
texts = texts_with_title.to_list()
len(texts)

# %%
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.save("custom_bpe_tokenizer.json")
tokenizer.get_vocab_size()


# %%
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="custom_bpe_tokenizer.json")


# %%
encoded = tokenizer.encode("akademi akademisyen akademisyenler ve aileleri ile birlikte aktif çalışıyorlar")
print(tokenizer.decode(encoded.ids))
print(fast_tokenizer.tokenize("akademi akademisyen akademisyenler ve aileleri ile birlikte aktif çalışıyorlar"))

# %%
""" 
akademisyen Ada er ve aile Ada er ad il e birlikte aktif çalış Ahize yor la Adalet
['akademisyen', 'l', 'er', 've', 'aile', 'l', 'er', 'i', 'il', 'e', 'birlikte', 'aktif', 'çalış', 'ı', 'yor', 'la', 'r']

akade mi akademisyen akademisyenler ve aileleri ile birlikte aktif çalışıyor lar
['akade', 'mi', 'akademisyen', 'akademisyenler', 've', 'aileleri', 'ile', 'birlikte', 'aktif', 'çalışıyor', 'lar']
 """

# %%
fast_tokenizer.push_to_hub("alibayram/tr_tokenizer")

# %%
fast_tokenizer.tokenize(texts[0])


````
