from pydantic import BaseModel
from typing import Dict, Type, Literal
from enum import Enum


class Language(BaseModel):
    # General
    title: str
    subtitle: str
    
    # Navigation
    previous: str
    next: str
    save_annotation: str
    edit_annotation: str
    download_json: str
    clear_annotations: str
    
    # Status messages
    annotated: str
    not_annotated: str
    annotation_saved: str
    no_annotations: str
    all_annotations_cleared: str
    
    # Sections
    audio_sample: str
    quran_reference: str
    uthmani_script: str
    phonetic_transcription: str
    sifat_table: str
    qdat_bench_annotation: str
    annotations_management: str
    
    # Fields
    id: str
    source: str
    original_id: str
    progress: str
    sura: str
    enter_search_text: str
    generate_phonetic_transcription: str
    phonetic_script: str
    add_row_at_end: str
    insert_after_row: str
    insert_row: str
    select_row_to_delete: str
    delete_selected_row: str
    gender: str
    madd_lengths: str
    ghonnah: str
    qalqalah: str
    
    # Gender options
    male: str
    female: str
    
    # Madd lengths
    qalo_alif_len: str
    qalo_waw_len: str
    laa_alif_len: str
    separate_madd: str
    allam_alif_len: str
    madd_aared_len: str
    
    # Ghonnah options
    noon_moshaddadah_len: str
    noon_mokhfah_len: str
    
    # Qalqalah options
    qalqalah_field: str
    
    # Sifat attributes
    hams: str
    jahr: str
    shadeed: str
    between: str
    rikhw: str
    mofakham: str
    moraqaq: str
    low_mofakham: str
    monfateh: str
    motbaq: str
    safeer: str
    no_safeer: str
    moqalqal: str
    not_moqalqal: str
    mokarar: str
    not_mokarar: str
    motafashie: str
    not_motafashie: str
    mostateel: str
    not_mostateel: str
    maghnoon: str
    not_maghnoon: str
    
    # Sifat column names
    phoneme: str
    hams_or_jahr: str
    shidda_or_rakhawa: str
    tafkheem_or_taqeeq: str
    itbaq: str
    safeer_col: str
    qalqla: str
    tikraar: str
    tafashie: str
    istitala: str
    ghonna: str


class EnglishLang(Language):
    title: str = "Quran Audio Transcription Annotation Tool"
    subtitle: str = ""
    
    # Navigation
    previous: str = "Previous"
    next: str = "Next"
    save_annotation: str = "Save Annotation"
    edit_annotation: str = "Edit Annotation"
    download_json: str = "Download JSON"
    clear_annotations: str = "Clear All Annotations"
    
    # Status messages
    annotated: str = "✓ Annotated"
    not_annotated: str = "Not annotated"
    annotation_saved: str = "Annotation saved"
    no_annotations: str = "No annotations have been saved yet."
    all_annotations_cleared: str = "All annotations cleared"
    
    # Sections
    audio_sample: str = "Audio Sample"
    quran_reference: str = "Quran Reference"
    uthmani_script: str = "Uthmani Script"
    phonetic_transcription: str = "Phonetic Transcription"
    sifat_table: str = "Sifat Table"
    qdat_bench_annotation: str = "QdataBenchItem Annotation"
    annotations_management: str = "Annotations Management"
    
    # Fields
    id: str = "ID"
    source: str = "Source"
    original_id: str = "Original ID"
    progress: str = "Progress"
    sura: str = "Sura"
    enter_search_text: str = "Enter search text"
    generate_phonetic_transcription: str = "Generate Phonetic Transcription"
    phonetic_script: str = "Phonetic Script"
    add_row_at_end: str = "Add Row at End"
    insert_after_row: str = "Insert after row"
    insert_row: str = "Insert Row"
    select_row_to_delete: str = "Select row to delete"
    delete_selected_row: str = "Delete Selected Row"
    gender: str = "Gender"
    madd_lengths: str = "Madd Lengths"
    ghonnah: str = "Ghonnah"
    qalqalah: str = "Qalqalah"
    
    # Gender options
    male: str = "male"
    female: str = "female"
    
    # Madd lengths
    qalo_alif_len: str = "Qalo Alif Len"
    qalo_waw_len: str = "Qalo Waw Len"
    laa_alif_len: str = "Laa Alif Len"
    separate_madd: str = "Separate Madd"
    allam_alif_len: str = "Allam Alif Len"
    madd_aared_len: str = "Madd Aared Len"
    
    # Ghonnah options
    noon_moshaddadah_len: str = "Noon Moshaddadah Len"
    noon_mokhfah_len: str = "Noon Mokhfah Len"
    
    # Qalqalah options
    qalqalah_field: str = "Qalqalah"
    
    # Sifat attributes
    hams: str = "hams"
    jahr: str = "jahr"
    shadeed: str = "shadeed"
    between: str = "between"
    rikhw: str = "rikhw"
    mofakham: str = "mofakham"
    moraqaq: str = "moraqaq"
    low_mofakham: str = "low_mofakham"
    monfateh: str = "monfateh"
    motbaq: str = "motbaq"
    safeer: str = "safeer"
    no_safeer: str = "no_safeer"
    moqalqal: str = "moqalqal"
    not_moqalqal: str = "not_moqalqal"
    mokarar: str = "mokarar"
    not_mokarar: str = "not_mokarar"
    motafashie: str = "motafashie"
    not_motafashie: str = "not_motafashie"
    mostateel: str = "mostateel"
    not_mostateel: str = "not_mostateel"
    maghnoon: str = "maghnoon"
    not_maghnoon: str = "not_maghnoon"
    
    # Sifat column names
    phoneme: str = "Phoneme"
    hams_or_jahr: str = "hams_or_jahr"
    shidda_or_rakhawa: str = "shidda_or_rakhawa"
    tafkheem_or_taqeeq: str = "tafkheem_or_taqeeq"
    itbaq: str = "itbaq"
    safeer_col: str = "safeer"
    qalqla: str = "qalqla"
    tikraar: str = "tikraar"
    tafashie: str = "tafashie"
    istitala: str = "istitala"
    ghonna: str = "ghonna"


class ArabicLang(Language):
    title: str = "أداة تدوين تسجيلات القرآن الكريم"
    subtitle: str = ""
    
    # Navigation
    previous: str = "السابق"
    next: str = "التالي"
    save_annotation: str = "حفظ التدوين"
    edit_annotation: str = "تعديل التدوين"
    download_json: str = "تحميل JSON"
    clear_annotations: str = "مسح كل التدوينات"
    
    # Status messages
    annotated: str = "✓ تم التدوين"
    not_annotated: str = "لم يتم التدوين"
    annotation_saved: str = "تم حفظ التدوين"
    no_annotations: str = "لا توجد تدوينات محفوظة بعد"
    all_annotations_cleared: str = "تم مسح جميع التدوينات"
    
    # Sections
    audio_sample: str = "عينة الصوت"
    quran_reference: str = "مرجع القرآن"
    uthmani_script: str = "النص العثماني"
    phonetic_transcription: str = "النص الصوتي"
    sifat_table: str = "جدول الصفات"
    qdat_bench_annotation: str = "تدوين QdataBenchItem"
    annotations_management: str = "إدارة التدوينات"
    
    # Fields
    id: str = "المعرف"
    source: str = "المصدر"
    original_id: str = "المعرف الأصلي"
    progress: str = "التقدم"
    sura: str = "السورة"
    enter_search_text: str = "أدخل نص البحث"
    generate_phonetic_transcription: str = "إنشاء النص الصوتي"
    phonetic_script: str = "النص الصوتي"
    add_row_at_end: str = "إضافة صف في النهاية"
    insert_after_row: str = "إدراج بعد الصف"
    insert_row: str = "إدراج صف"
    select_row_to_delete: str = "اختر صفًا للحذف"
    delete_selected_row: str = "حذف الصف المحدد"
    gender: str = "الجنس"
    madd_lengths: str = "أطوال المد"
    ghonnah: str = "الغنة"
    qalqalah: str = "القلقلة"
    
    # Gender options
    male: str = "ذكر"
    female: str = "أنثى"
    
    # Madd lengths
    qalo_alif_len: str = "طول مد الألف في قالوا"
    qalo_waw_len: str = "طول مد الواو في قالوا"
    laa_alif_len: str = "طول مد الألف في لا"
    separate_madd: str = "مد المنفصل"
    allam_alif_len: str = "طول مد الألف في علام"
    madd_aared_len: str = "طول مد العارض للسكون"
    
    # Ghonnah options
    noon_moshaddadah_len: str = "طول النون المشددة"
    noon_mokhfah_len: str = "طول النون المخفاة"
    
    # Qalqalah options
    qalqalah_field: str = "القلقلة"
    
    # Sifat attributes - Using the original mapping
    hams: str = "همس"
    jahr: str = "جهر"
    shadeed: str = "شديد"
    between: str = "بين الشدة والرخاوة"
    rikhw: str = "رخو"
    mofakham: str = "مفخم"
    moraqaq: str = "مرقق"
    low_mofakham: str = "أدنى المفخم"
    monfateh: str = "منفتح"
    motbaq: str = "مطبق"
    safeer: str = "صفير"
    no_safeer: str = "لا صفير"
    moqalqal: str = "مقلقل"
    not_moqalqal: str = "لا قلقلة"
    mokarar: str = "مكرر"
    not_mokarar: str = "لا تكرار"
    motafashie: str = "متفشي"
    not_motafashie: str = "لا تفشي"
    mostateel: str = "مستطيل"
    not_mostateel: str = "لا إستطالة"
    maghnoon: str = "مغن"
    not_maghnoon: str = "لا غنة"
    
    # Sifat column names
    phoneme: str = "الصوت"
    hams_or_jahr: str = "همس أو جهر"
    shidda_or_rakhawa: str = "شدة أو رخاوة"
    tafkheem_or_taqeeq: str = "تفخيم أو ترقيق"
    itbaq: str = "إطباق"
    safeer_col: str = "صفير"
    qalqla: str = "قلقلة"
    tikraar: str = "تكرار"
    tafashie: str = "تفشي"
    istitala: str = "استطالة"
    ghonna: str = "غنة"


# Language mapping
LANGUAGES = {
    "english": EnglishLang,
    "arabic": ArabicLang
}


def get_language(lang: Literal["english", "arabic"]) -> Language:
    return LANGUAGES[lang]()
