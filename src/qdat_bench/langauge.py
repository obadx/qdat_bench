from pydantic import BaseModel


SIFAT_ATTR_TO_ARABIC = {
    "hams": "همس",
    "jahr": "جهر",
    "shadeed": "شديد",
    "between": "بين الشدة والرخاوة",
    "rikhw": "رخو",
    "mofakham": "مفخم",
    "moraqaq": "مرقق",
    "low_mofakham": "أدنى المفخم",
    "monfateh": "منفتح",
    "motbaq": "مطبق",
    "safeer": "صفير",
    "no_safeer": "لا صفير",
    "moqalqal": "مقلقل",
    "not_moqalqal": "لا قلقلة",
    "mokarar": "مكرر",
    "not_mokarar": "لا تكرار",
    "motafashie": "متفشي",
    "not_motafashie": "لا تفشي",
    "mostateel": "مستطيل",
    "not_mostateel": "لا إستطالة",
    "maghnoon": "مغن",
    "not_maghnoon": "لا غنة",
}


class Language(BaseModel):
    title: str
    subtitle: str


class ArabicLang(Language): ...


class EnglsihLang(Language): ...
