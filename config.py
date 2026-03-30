ACCESS_CODES: dict[str, int] = {
    "anotator-01": 0,
    "anotator-02": 1,
    "anotator-03": 2,
    "anotator-04": 3,
    "anotator-05": 4,
    "anotator-06": 5,
    "anotator-07": 6,
    "anotator-08": 7,
    "anotator-09": 8,
    "anotator-10": 9,
    # Speed-controlled lab annotators (50 tweets, 15-min timer)
    "scl-01": 10,
    "scl-02": 11,
    "scl-03": 12,
    "ff-01": 13,
    "ff-02": 14,
    "ff-03": 15,
    "ff-04": 16,
    "ff-05": 17,
    "ff-test": 18,
    **{
        f"ff{code}": 18 + offset // 7
        for offset, code in enumerate(range(101, 178))
    },
}

SCL_CODES: set[str] = {"scl-01", "scl-02", "scl-03"}

EMOTIONS: list[str] = [
    "poverenje",
    "bes",
    "tuga",
    "iznenađenje",
    "strah",
    "gađenje",
    "radost",
    "iščekivanje",
]

SPECIAL_LABELS: list[str] = [
    "Emocionalno neutralno",
    "Ne mogu da razumem",
]

LABEL_NORM: dict[str, str] = {
    "gadjenje": "gađenje",
    "anticipacija": "iščekivanje",
    "iznenadjenje": "iznenađenje",
    "bes": "bes",
    "tuga": "tuga",
    "strah": "strah",
    "poverenje": "poverenje",
    "radost": "radost",
}

EXAMPLES: list[dict] = [
    {
        "emotion": "bes",
        "label": "bes",
        "tweet": "Nisam prijavila jer mi je rečeno da ću ja biti ta koja će biti osuđena. Sistem štiti nasilnike! #NisamPrijavila",
        "explanation": "Direktan napad na sistem koji štiti počinioce; izražen gnev i frustracija.",
    },
    {
        "emotion": "bes",
        "label": "bes",
        "tweet": "Policajac mi je rekao: 'Šta si očekivala kada si izašla sama noću?' Neverovatno. #NisamPrijavila",
        "explanation": "Ogorčenost zbog victim-blaming stavova institucija.",
    },
    {
        "emotion": "tuga",
        "label": "tuga",
        "tweet": "Godinama nosim ovu tajnu sama. Nisam imala koga da pitam za pomoć. #NisamPrijavila",
        "explanation": "Osećaj usamljenosti i bola zbog nošenja traume bez podrške.",
    },
    {
        "emotion": "tuga",
        "label": "tuga",
        "tweet": "Plačem čitajući vaše priče. Toliko nas je. Zašto moramo da prolazimo kroz ovo? #NisamPrijavila",
        "explanation": "Tuga izazvana empatijom prema drugima i osećajem kolektivne patnje.",
    },
    {
        "emotion": "strah",
        "label": "strah",
        "tweet": "Nisam prijavila jer mi je zapretio da će doći po mene ako progovorim. Živim u strahu. #NisamPrijavila",
        "explanation": "Eksplicitno pominjanje straha od odmazde kao razloga za ćutanje.",
    },
    {
        "emotion": "strah",
        "label": "strah",
        "tweet": "I danas se bojim da izađem napolje ako ga vidim. Nikada se nisam oporavila. #NisamPrijavila",
        "explanation": "Trajni strah koji utiče na svakodnevni život žrtve.",
    },
    {
        "emotion": "poverenje",
        "label": "poverenje",
        "tweet": "Konačno sam progovorila i porodica me je podržala. Hvala im što su mi verovali. #NisamPrijavila",
        "explanation": "Izražavanje zahvalnosti i poverenja prema bliskim osobama koje su pružile podršku.",
    },
    {
        "emotion": "poverenje",
        "label": "poverenje",
        "tweet": "Moja terapeutkinja mi je pomogla da shvatim da nisam kriva. Ima dobrih ljudi. #NisamPrijavila",
        "explanation": "Poverenje u stručnu osobu koja je pružila podršku i razumevanje.",
    },
    {
        "emotion": "gađenje",
        "label": "gađenje",
        "tweet": "Odvratno mi je što takvi ljudi slobodno hodaju ulicom dok mi se skrivamo. #NisamPrijavila",
        "explanation": "Moralna odbojnost prema nepravdi i slobodi počinilaca.",
    },
    {
        "emotion": "gađenje",
        "label": "gađenje",
        "tweet": "Mučno mi je kad vidim kako ga svi hvale a znam šta je uradio. #NisamPrijavila",
        "explanation": "Fizički opis gađenja uz moralno negodovanje zbog kontrasta između javnog imidža i stvarnosti.",
    },
    {
        "emotion": "radost",
        "label": "radost",
        "tweet": "Progovorila sam i osećam se slobodnom prvi put posle godina. Hvala svima na podršci! #NisamPrijavila",
        "explanation": "Oslobađanje i radost nakon što je žrtva odlučila da progovori.",
    },
    {
        "emotion": "radost",
        "label": "radost",
        "tweet": "Vidim da nas ima toliko i da nismo same — to mi daje snagu i nadu. #NisamPrijavila",
        "explanation": "Pozitivna emocija nastala iz solidarnosti i osećaja zajednice.",
    },
    {
        "emotion": "iznenađenje",
        "label": "iznenađenje",
        "tweet": "Nisam mogla da verujem kad sam videla ko je — niko ne bi pomislio na njega. #NisamPrijavila",
        "explanation": "Šok zbog identiteta počinioca koji se nije uklapao u očekivanja.",
    },
    {
        "emotion": "iznenađenje",
        "label": "iznenađenje",
        "tweet": "Nisam očekivala da će toliko žena podeliti svoje priče. Ovaj pokret me je zapanjio. #NisamPrijavila",
        "explanation": "Iznenađenost obimom i snagom kolektivnog svjedočenja.",
    },
    {
        "emotion": "iščekivanje",
        "label": "iščekivanje",
        "tweet": "Nadam se da će ovo konačno dovesti do promene zakona. Pratim svaki korak. #NisamPrijavila",
        "explanation": "Iščekivanje sistemske promene i aktivno praćenje razvoja situacije.",
    },
    {
        "emotion": "iščekivanje",
        "label": "iščekivanje",
        "tweet": "Čekamo da vidimo hoće li institucije ovog puta stvarno reagovati ili opet ćutati. #NisamPrijavila",
        "explanation": "Napeto iščekivanje odgovora institucija uz implicitnu skepsu.",
    },
]
