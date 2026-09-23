# Recording Guide — clean voice sample kivabe banabe

Clone er quality **90% depend kore recording er quality er upor.** Noisy/echoey audio = kharap clone.

## Kototuku record korbo?

| Purpose | Length |
|---------|--------|
| Zero-shot (XTTS / Chatterbox-Bangla) | **10-30 second** clean audio-i enough |
| Fine-tune (GPT-SoVITS) — best quality | **5-10 minute** |

Tumi 5 min record korle dutai kaj korbe (fine-tune er jonno pura, zero-shot er jonno best 20 sec kete newa jabe).

## Setup (equipment lagbe na, phone-o cholе)

- **Quiet room** — fan/AC off, door bondho, kono echo na (chhoto room, curtain thakle better).
- **Mic distance** — mukh theke ~15-20 cm (khub kache na, breathing/pop ashbe).
- **Format** — WAV preferred (MP3 o cholе). Mono, 22050 Hz or higher, 16-bit.
- Phone er "Voice Recorder" / laptop er built-in mic-o cholе, kintu external mic hole onek better.

## Kotha bolar niyom

- **Normal, natural tone** e poro — jemon voice clone chao thik temon (news-reading style hole news-style clone hobe).
- **Consistent volume ও speed** — hoot kore joré/aste kora jabe na.
- Ek nagade poro, boro pause diyo na. Vul hole thamiye abar oi line theke shuru koro (por e edit kore felo).
- **Bangla + English dutai** clone korte chaile — recording e **dutai** rakho (kichu Bangla line, kichu English line). Tahole model dutar accent/timbre bujhbe.

## Ki poro? (script)

Nicher script ~5 min. Nijer moto boro/choto koro. Bangla + English mixed rakha holo bhalo.

### Bangla portion
> Amar naam [naam]. Ami ekhon ekta voice sample record korchi jate amar konthosor clone kora jay.
> Aajke akasher obostha bhalo, roud utheche. Boshonto kaal amar khub priyo ekta somoy.
> Prithibir shobcheye boro mohadesh holo Asia. Bangladesher rajdhani Dhaka.
> Ekta, dui, tin, char, panch, chhoy, saat, aat, noy, dosh.
> Bigyan o projukti amader jibon ke shohoj kore diyeche.
> Boi porar obhyash ekjon manushke gyani kore tole.
> Ami protidin sokale hete jai ebong tarpor kaj shuru kori.

### English portion
> Hello, my name is [name] and I am recording a sample of my voice.
> The quick brown fox jumps over the lazy dog near the river bank.
> Today the weather is clear and the sky is bright blue.
> One, two, three, four, five, six, seven, eight, nine, ten.
> Technology has transformed the way we live, work, and communicate.
> Reading books every day is one of the best habits a person can build.
> I usually wake up early, go for a short walk, and then start my work.

*(Emotion/expression variation add korte chaile kichu line khushi, kichu serious tone e poro — model expressive hobe.)*

## Record korar por

1. File ta `voices/` folder e rakho, naam `sample.wav`.
2. Beginning/end er silence ba noise Audacity (free) diye trim koro.
3. Beshi background noise thakle Colab notebook er "denoise" cell run korte paro.
