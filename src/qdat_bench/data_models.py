from typing import Literal
from enum import IntEnum


from quran_transcript import SifaOutput
from pydantic import BaseModel, Field

MADD_LEN = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]


class NoonMoshaddahLen(IntEnum):
    PARTIAL = 0
    COMPLETE = 1


class NoonMokhfahLen(IntEnum):
    NOON = 0
    PARTIAL = 1
    COMPLETE = 2


class Qalqalah(IntEnum):
    NO_QALQALH = 0
    HAS_QALQALAH = 1


class QdataBenchItem(BaseModel):
    # premetives
    id: str = Field(description="Unique hash id for every element")
    original_id: str = Field(description="The item's id in the original dataset")
    gender: Literal["male", "female"]
    age: int

    # General Tajeweed
    qalo_alif_len: MADD_LEN = Field(
        default=2, description="The lengths of the normal madd alif form word `قالوا`"
    )
    qalo_waw_len: MADD_LEN = Field(
        default=2, description="The length of the normal madd waw form word `قالوا`"
    )
    laa_alif_len: MADD_LEN = Field(
        default=2, description="The length of the normal madd alif form word `لا`"
    )
    separate_madd: MADD_LEN = Field(
        default=4, description="The length of separate madd for word `لنا إنك`"
    )
    noon_moshaddadah_len: NoonMoshaddahLen = Field(
        default=NoonMoshaddahLen.COMPLETE,
        description="The length of noon moshaddah for word `إنَّك`",
    )
    noon_mokhfah_len: NoonMokhfahLen = Field(
        default=NoonMokhfahLen.COMPLETE,
        description="The length of noon mokhfah for word ` أنت`",
    )
    allam_alif_len: MADD_LEN = Field(
        default=2, description="The lengths of the normal madd alif form word `علام`"
    )

    madd_aared_len: MADD_LEN = Field(
        default=4, description="The length of the مد العارض للسكون "
    )
    qalqalah: Qalqalah = Field(
        default=Qalqalah.HAS_QALQALAH,
        description="The existance of qalqalah for not for word `الغيوب`",
    )

    # quran-transcript
    phonetic_transcript: str = Field(
        description="The phoetic transcript using `quran-transcript` package guidlenes"
    )
    sifat: list[SifaOutput] = Field(
        description="The sifat  transcript level using `quran-transcript` package guidlenes"
    )
