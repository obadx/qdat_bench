import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes, search, SifaOutput
from datasets import load_dataset, Dataset

from qdat_bench.langauge import get_language, get_all_languages
from qdat_bench.data_models import (
    QdataBenchItem,
    Qalqalah,
    NoonMokhfahLen,
    NoonMoshaddahLen,
)
from build_qdat_bench_audio_source import chose_single_source


def initialize():
    # Initialize session state
    if "lang" not in st.session_state:
        st.session_state.lang = "arabic"
    if "index" not in st.session_state:
        st.session_state.index = 0
    if "sura_idx" not in st.session_state:
        st.session_state.sura_idx = 1
    if "start_aya" not in st.session_state:
        st.session_state.start_aya = Aya(1, 1)
    if "uthmani_script" not in st.session_state:
        st.session_state.uthmani_script = ""
    if "phonetic_script" not in st.session_state:
        st.session_state.phonetic_script = ""
    if "sifat_df" not in st.session_state:
        st.session_state.sifat_df = pd.DataFrame()
    if "last_editor_key" not in st.session_state:
        st.session_state.last_editor_key = ""
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "item_to_edit" not in st.session_state:
        st.session_state.item_to_edit = None
    if "gen_ph_script_pressed" not in st.session_state:
        st.session_state.gen_ph_script_pressed = False

    # Initialize QdataBenchItem fields
    qdat_item_fields = set(QdataBenchItem.model_fields.keys()) - {"id", "original_id"}
    for field_name in qdat_item_fields:
        if field_name not in st.session_state:
            st.session_state[field_name] = QdataBenchItem.model_fields[
                field_name
            ].default

    # Initialize annotations
    if not st.session_state.annotations:
        st.session_state.annotations = load_annotations()


@st.cache_data
def load_language_dict():
    return get_all_languages()


def load_language_settings():
    st.session_state.lang_sett = load_language_dict()[st.session_state.lang]


# Load dataset
@st.cache_resource
def load_audio_dataset():
    seed = 42
    qdata_path = "/home/abdullah/Downloads/qdat/"
    ds = load_dataset("audiofolder", data_dir=qdata_path)
    print("Before Filteration")
    print(ds)

    ds = chose_single_source(ds["train"], seed=seed)
    print("After Filteration")
    print(ds)

    return ds, list(range(len(ds)))

    # ds = load_dataset("obadx/ood_muaalem_test", split="train")
    # rng = np.random.default_rng(seed=42)
    # ids = np.arange(len(ds))
    # rng.shuffle(ids)
    # ids = [int(i) for i in ids]  # Convert to integers
    # return ds, ids


# Cache sura information
@st.cache_data
def get_sura_info():
    sura_idx_to_name = {}
    sura_to_aya_count = {}
    start_aya = Aya()
    for sura_idx in range(1, 115):
        start_aya.set(sura_idx, 1)
        sura_idx_to_name[sura_idx] = start_aya.get().sura_name
        sura_to_aya_count[sura_idx] = start_aya.get().num_ayat_in_sura
    return sura_idx_to_name, sura_to_aya_count


# Load existing annotations
def load_annotations() -> dict:
    annotations_file = Path("annotations.json")
    if annotations_file.exists():
        with open(annotations_file, "r", encoding="utf-8") as f:
            raw_annotations = json.load(f)
            return raw_annotations
    return {}


# Function to save annotation to JSON file
def save_annotation(item_id, annotation):
    # Update session state
    st.session_state.annotations[item_id] = annotation
    # validatiing annotation before saveing
    saved_annotations = {
        k: QdataBenchItem(**v) for k, v in st.session_state.annotations.items()
    }
    saved_annotations = {k: v.model_dump() for k, v in saved_annotations.items()}

    # Save to file
    with open("annotations.json", "w", encoding="utf-8") as f:
        json.dump(saved_annotations, f, ensure_ascii=False, indent=2)


def load_item_to_session_state(item):
    # Check if current item has existing annotation
    if item["id"] in st.session_state.annotations and not st.session_state.edit_mode:
        annotation_data = st.session_state.annotations[item["id"]]
        # Load QdataBenchItem fields
        st.session_state.phonetic_script = annotation_data.get(
            "phonetic_transcript", ""
        )
        # Convert sifat to DataFrame
        sifat_list = annotation_data.get("sifat", [])
        st.session_state.sifat_df = pd.DataFrame(
            [
                sifa.model_dump() if isinstance(sifa, SifaOutput) else sifa
                for sifa in sifat_list
            ]
        )
        # Add row index column
        if not st.session_state.sifat_df.empty:
            st.session_state.sifat_df.insert(
                0, "row_index", range(1, len(st.session_state.sifat_df) + 1)
            )
        # Load other fields
        # Always use the gender from the dataset item, not from annotation
        st.session_state.gender = item.get("gender", "male")
        st.session_state.qalo_alif_len = annotation_data.get("qalo_alif_len")
        st.session_state.qalo_waw_len = annotation_data.get("qalo_waw_len")
        st.session_state.laa_alif_len = annotation_data.get("laa_alif_len")
        st.session_state.separate_madd = annotation_data.get("separate_madd")
        st.session_state.noon_moshaddadah_len = annotation_data.get(
            "noon_moshaddadah_len"
        )
        st.session_state.noon_mokhfah_len = annotation_data.get("noon_mokhfah_len")
        st.session_state.allam_alif_len = annotation_data.get("allam_alif_len")
        st.session_state.madd_aared_len = annotation_data.get("madd_aared_len")
        st.session_state.qalqalah = annotation_data.get("qalqalah")


# Display audio and metadata
def display_item(item, ids):
    st.header(st.session_state.lang_sett.audio_sample)
    st.audio(item["audio"]["array"], sample_rate=item["audio"]["sampling_rate"])

    # First row: 3 columns
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.metric(st.session_state.lang_sett.id, item["id"])
    with row1_col2:
        st.metric(st.session_state.lang_sett.original_id, item["original_id"])
    with row1_col3:
        st.metric(
            st.session_state.lang_sett.progress,
            f"{st.session_state.index + 1}/{len(ids)}",
        )
    
    # Second row: 3 columns
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        # Display gender
        gender_display = {
            "male": st.session_state.lang_sett.male,
            "female": st.session_state.lang_sett.female,
        }.get(item.get("gender", ""), item.get("gender", ""))
        st.metric(st.session_state.lang_sett.gender, gender_display)
    with row2_col2:
        # Display age
        st.metric(st.session_state.lang_sett.age, item.get("age", "N/A"))

    # Show annotation status
    if item["id"] in st.session_state.annotations:
        st.success(st.session_state.lang_sett.annotated)
    else:
        st.info(st.session_state.lang_sett.not_annotated)

    # Navigator to jump to specific source_id
    st.subheader(st.session_state.lang_sett.navigator)
    item_idx = st.number_input(
        st.session_state.lang_sett.enter_reciter_id,
        min_value=0,
        max_value=len(ids) - 1,
        value=0,
        step=1,
    )

    if st.button(st.session_state.lang_sett.jump_to_reciter):
        # Find the item with the matching source_id pattern
        target_id = f"s{item_idx}_"
        found_index = None
        for idx, original_idx in enumerate(ids):
            item_id = ds[original_idx]["id"]
            if item_id.startswith(target_id):
                found_index = idx
                break
        
        if found_index is not None:
            st.session_state.index = found_index
            reset_item_session_state()
            st.rerun()
        else:
            st.error(f"{st.session_state.lang_sett.no_item_found} {item_idx}")


def select_quran_text(sura_idx_to_name, sura_to_aya_count) -> str:
    """Select Quran Text

    Retusn:
        str: Uthmani text
    """
    # Quran reference selection
    st.header(st.session_state.lang_sett.quran_reference)

    sura_idx = st.selectbox(
        st.session_state.lang_sett.sura,
        options=list(range(1, 115)),
        format_func=lambda x: f"{x}. {sura_idx_to_name[x]}",
        # index=st.session_state.sura_idx - 1,
    )
    max_aya = sura_to_aya_count[sura_idx]
    st.session_state.start_aya.set(sura_idx, max_aya // 2)
    search_text = st.text_input(
        st.session_state.lang_sett.enter_search_text,
        value=st.session_state.start_aya.set_new(sura_idx, 1).get().imlaey,
    )
    search_out = search(
        search_text,
        start_aya=st.session_state.start_aya,
        window=max_aya + 1,
        remove_tashkeel=True,
        ignore_hamazat=True,
    )
    if len(search_out) == 0:
        st.error("No search results found")
        uthmani_script = ""
    else:
        uthmani_script = search_out[0].uthmani_script

    st.subheader(st.session_state.lang_sett.uthmani_script)
    st.write(uthmani_script)

    # Update session state
    st.session_state.sura_idx = sura_idx
    st.session_state.uthmani_script = uthmani_script
    return uthmani_script


# Phonetization
def annotate_phonetic_script(uthmani_script: str, default_moshaf: MoshafAttributes):
    st.subheader(st.session_state.lang_sett.phonetic_transcription)
    if st.button(st.session_state.lang_sett.generate_phonetic_transcription):
        # Toggle the flag to force re-render of the text area
        st.session_state.gen_ph_script_pressed = (
            not st.session_state.gen_ph_script_pressed
        )
        phonetizer_out = quran_phonetizer(
            st.session_state.uthmani_script, default_moshaf
        )
        st.session_state.phonetic_script = phonetizer_out.phonemes

        # Create sifat table
        # TODO: aligment here
        sifat_data = []
        for sifa in phonetizer_out.sifat:
            sifat_data.append(sifa.model_dump())

        st.session_state.sifat_df = pd.DataFrame(sifat_data)
        # No need to call st.rerun() here as the button press will trigger a rerun

    # Always show the phonetic script editor, and update its value from session state
    # Use a key that changes when the button is pressed to force re-render
    if st.session_state.phonetic_script:
        phonetic_script_value = st.text_area(
            st.session_state.lang_sett.phonetic_script,
            value=st.session_state.phonetic_script,
            width=300,
            key=f"phonetic_script_editor_{st.session_state.gen_ph_script_pressed}",
        )
        # Update session state whenever the text area content changes
        if phonetic_script_value != st.session_state.phonetic_script:
            st.session_state.phonetic_script = phonetic_script_value

    annotate_sifat()


def annotate_sifat():
    # Editable sifat table
    if not st.session_state.sifat_df.empty:
        st.subheader(st.session_state.lang_sett.sifat_table)

        # Add row index column if it doesn't exist
        if "row_index" not in st.session_state.sifat_df.columns:
            st.session_state.sifat_df.insert(
                0, "row_index", range(1, len(st.session_state.sifat_df) + 1)
            )

        # Add row operations
        col_add, col_add_pos, col_del = st.columns(3)
        with col_add:
            if st.button("➕ " + st.session_state.lang_sett.add_row_at_end):
                # Add a new empty row at the end
                new_row = {
                    col: ""
                    for col in st.session_state.sifat_df.columns
                    if col != "row_index"
                }
                new_row["row_index"] = len(st.session_state.sifat_df) + 1
                st.session_state.sifat_df = pd.concat(
                    [st.session_state.sifat_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                st.rerun()

        with col_add_pos:
            if len(st.session_state.sifat_df) > 0:
                insert_position = st.selectbox(
                    st.session_state.lang_sett.insert_after_row,
                    options=list(range(len(st.session_state.sifat_df))),
                    format_func=lambda x: f"After row {x + 1}",
                )
                if st.button("➕ " + st.session_state.lang_sett.insert_row):
                    # Split the dataframe and insert new row
                    new_row = {
                        col: ""
                        for col in st.session_state.sifat_df.columns
                        if col != "row_index"
                    }
                    new_row["row_index"] = (
                        insert_position + 1.5
                    )  # Placeholder, will be renumbered
                    # Insert the new row
                    st.session_state.sifat_df = pd.concat(
                        [
                            st.session_state.sifat_df.iloc[: insert_position + 1],
                            pd.DataFrame([new_row]),
                            st.session_state.sifat_df.iloc[insert_position + 1 :],
                        ],
                        ignore_index=True,
                    )
                    # Renumber all rows
                    st.session_state.sifat_df["row_index"] = range(
                        1, len(st.session_state.sifat_df) + 1
                    )
                    st.rerun()

        with col_del:
            if len(st.session_state.sifat_df) > 0:
                row_to_delete = st.selectbox(
                    st.session_state.lang_sett.select_row_to_delete,
                    options=list(range(len(st.session_state.sifat_df))),
                    format_func=lambda x: f"Row {x + 1}",
                )
                if st.button("➖ " + st.session_state.lang_sett.delete_selected_row):
                    st.session_state.sifat_df = st.session_state.sifat_df.drop(
                        st.session_state.sifat_df.index[row_to_delete]
                    ).reset_index(drop=True)
                    # Renumber remaining rows
                    st.session_state.sifat_df["row_index"] = range(
                        1, len(st.session_state.sifat_df) + 1
                    )
                    st.rerun()

        # Define options for each column with English values but display translated text
        column_options = {
            "hams_or_jahr": ["hams", "jahr"],
            "shidda_or_rakhawa": ["shadeed", "between", "rikhw"],
            "tafkheem_or_taqeeq": ["mofakham", "moraqaq", "low_mofakham"],
            "itbaq": ["monfateh", "motbaq"],
            "safeer": ["safeer", "no_safeer"],
            "qalqla": ["moqalqal", "not_moqalqal"],
            "tikraar": ["mokarar", "not_mokarar"],
            "tafashie": ["motafashie", "not_motafashie"],
            "istitala": ["mostateel", "not_mostateel"],
            "ghonna": ["maghnoon", "not_maghnoon"],
        }

        # Create a mapping from English values to their translated versions
        translation_map = {
            "hams": st.session_state.lang_sett.hams,
            "jahr": st.session_state.lang_sett.jahr,
            "shadeed": st.session_state.lang_sett.shadeed,
            "between": st.session_state.lang_sett.between,
            "rikhw": st.session_state.lang_sett.rikhw,
            "mofakham": st.session_state.lang_sett.mofakham,
            "moraqaq": st.session_state.lang_sett.moraqaq,
            "low_mofakham": st.session_state.lang_sett.low_mofakham,
            "monfateh": st.session_state.lang_sett.monfateh,
            "motbaq": st.session_state.lang_sett.motbaq,
            "safeer": st.session_state.lang_sett.safeer,
            "no_safeer": st.session_state.lang_sett.no_safeer,
            "moqalqal": st.session_state.lang_sett.moqalqal,
            "not_moqalqal": st.session_state.lang_sett.not_moqalqal,
            "mokarar": st.session_state.lang_sett.mokarar,
            "not_mokarar": st.session_state.lang_sett.not_mokarar,
            "motafashie": st.session_state.lang_sett.motafashie,
            "not_motafashie": st.session_state.lang_sett.not_motafashie,
            "mostateel": st.session_state.lang_sett.mostateel,
            "not_mostateel": st.session_state.lang_sett.not_mostateel,
            "maghnoon": st.session_state.lang_sett.maghnoon,
            "not_maghnoon": st.session_state.lang_sett.not_maghnoon,
        }

        # Create a unique key for this editor instance
        editor_key = f"sifat_editor_{st.session_state.index}"

        # Use the data editor with explicit value assignment
        edited_df = st.data_editor(
            st.session_state.sifat_df,
            column_config={
                "row_index": st.column_config.NumberColumn(
                    st.session_state.lang_sett.progress, width="small", disabled=True
                ),
                "phoneme": st.column_config.TextColumn(
                    st.session_state.lang_sett.phoneme, width="small"
                ),
                **{
                    col: st.column_config.SelectboxColumn(
                        getattr(st.session_state.lang_sett, col, col),
                        options=options,
                        format_func=lambda x: translation_map.get(x, x),
                    )
                    for col, options in column_options.items()
                },
            },
            width="stretch",
            num_rows="dynamic",  # Change to dynamic to allow adding rows through the editor
            key=editor_key,
            disabled=["row_index"],  # Make row index non-editable
        )

        # Update the session state if the dataframe has changed
        # Don't trigger a rerun here as it causes infinite loops
        if not edited_df.equals(st.session_state.sifat_df):
            st.session_state.sifat_df = edited_df
            # Update the last editor key to track changes
            st.session_state.last_editor_key = editor_key


def annotate_addional_qdabenc_fields(item):
    # QdataBenchItem fields
    st.header(st.session_state.lang_sett.qdat_bench_annotation)

    # Always use the gender from the dataset item
    st.session_state.gender = item.get("gender", "male")

    st.subheader(st.session_state.lang_sett.madd_lengths)
    cols = st.columns(4)
    with cols[0]:
        st.session_state.qalo_alif_len = st.slider(
            st.session_state.lang_sett.qalo_alif_len,
            0,
            8,
            st.session_state.qalo_alif_len,
        )
    with cols[1]:
        st.session_state.qalo_waw_len = st.slider(
            st.session_state.lang_sett.qalo_waw_len, 0, 8, st.session_state.qalo_waw_len
        )
    with cols[2]:
        st.session_state.laa_alif_len = st.slider(
            st.session_state.lang_sett.laa_alif_len, 0, 8, st.session_state.laa_alif_len
        )
    with cols[3]:
        st.session_state.separate_madd = st.slider(
            st.session_state.lang_sett.separate_madd,
            0,
            8,
            st.session_state.separate_madd,
        )

    cols = st.columns(3)
    with cols[0]:
        st.session_state.allam_alif_len = st.slider(
            st.session_state.lang_sett.allam_alif_len,
            0,
            8,
            st.session_state.allam_alif_len,
        )
    with cols[1]:
        st.session_state.madd_aared_len = st.slider(
            st.session_state.lang_sett.madd_aared_len,
            0,
            8,
            st.session_state.madd_aared_len,
        )

    st.subheader(st.session_state.lang_sett.ghonnah)
    ghonnah_cols = st.columns(2)
    with ghonnah_cols[1]:
        # Get current index for noon_moshaddadah_len
        if (
            hasattr(st.session_state, "noon_moshaddadah_len")
            and st.session_state.noon_moshaddadah_len is not None
        ):
            try:
                current_noon_moshaddadah_index = list(NoonMoshaddahLen).index(
                    st.session_state.noon_moshaddadah_len
                )
            except ValueError:
                current_noon_moshaddadah_index = 0
        else:
            current_noon_moshaddadah_index = 0
        st.session_state.noon_moshaddadah_len = st.selectbox(
            st.session_state.lang_sett.noon_moshaddadah_len,
            options=list(NoonMoshaddahLen),
            format_func=lambda x: {
                NoonMoshaddahLen.NO_GHONNAH: st.session_state.lang_sett.not_maghnoon,
                NoonMoshaddahLen.PARTIAL: st.session_state.lang_sett.partial,
                NoonMoshaddahLen.COMPLETE: st.session_state.lang_sett.complete,
            }[x],
            index=current_noon_moshaddadah_index,
        )

    with ghonnah_cols[0]:
        # Get current index for noon_mokhfah_len
        if (
            hasattr(st.session_state, "noon_mokhfah_len")
            and st.session_state.noon_mokhfah_len is not None
        ):
            try:
                current_noon_mokhfah_index = list(NoonMokhfahLen).index(
                    st.session_state.noon_mokhfah_len
                )
            except ValueError:
                current_noon_mokhfah_index = 0
        else:
            current_noon_mokhfah_index = 0
        st.session_state.noon_mokhfah_len = st.selectbox(
            st.session_state.lang_sett.noon_mokhfah_len,
            options=list(NoonMokhfahLen),
            format_func=lambda x: {
                NoonMokhfahLen.NOON: st.session_state.lang_sett.noon,
                NoonMokhfahLen.PARTIAL: st.session_state.lang_sett.partial,
                NoonMokhfahLen.COMPLETE: st.session_state.lang_sett.complete,
            }[x],
            index=current_noon_mokhfah_index,
        )

    st.subheader(st.session_state.lang_sett.qalqalah)
    # Get current index for qalqalah
    if hasattr(st.session_state, "qalqalah") and st.session_state.qalqalah is not None:
        try:
            current_qalqalah_index = list(Qalqalah).index(st.session_state.qalqalah)
        except ValueError:
            current_qalqalah_index = 0
    else:
        current_qalqalah_index = 0
    st.session_state.qalqalah = st.selectbox(
        st.session_state.lang_sett.qalqalah_field,
        options=list(Qalqalah),
        format_func=lambda x: {
            Qalqalah.NO_QALQALH: st.session_state.lang_sett.not_moqalqal,
            Qalqalah.HAS_QALQALAH: st.session_state.lang_sett.moqalqal,
        }[x],
        index=current_qalqalah_index,
    )


def reset_item_session_state():
    st.session_state.phonetic_script = ""
    st.session_state.sifat_df = pd.DataFrame()
    st.session_state.last_editor_key = ""
    st.session_state.edit_mode = False

    # Additional values
    for key in [
        "qalo_alif_len ",
        "qalo_waw_len ",
        "laa_alif_len ",
        "allam_alif_len ",
    ]:
        st.session_state[key] = 2
    st.session_state.separate_madd = 4
    st.session_state.madd_aared_len = 4

    st.session_state.gender = "female"
    st.session_state.noon_mokhfah_len = NoonMokhfahLen.COMPLETE
    st.session_state.noon_moshaddadah_len = NoonMoshaddahLen.COMPLETE
    st.session_state.qalqalah = Qalqalah.HAS_QALQALAH


def save_navigatoin_bar(item, ids):
    # Navigation and saving
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if (
            st.button(st.session_state.lang_sett.previous)
            and st.session_state.index > 0
        ):
            st.session_state.index -= 1
            # Reset for the new item
            reset_item_session_state()
            st.rerun()

    with col2:
        if st.button(st.session_state.lang_sett.save_annotation):
            # Remove the row_index column before saving
            sifat_df_to_save = st.session_state.sifat_df.drop(
                columns=["row_index"], errors="ignore"
            )
            # Convert sifat records to SifaOutput instances
            sifat_list = []
            for record in sifat_df_to_save.to_dict(orient="records"):
                # Filter out None values and empty strings
                filtered_record = {
                    k: v for k, v in record.items() if v is not None and v != ""
                }
                sifat_list.append(SifaOutput(**filtered_record))

            # Create QdataBenchItem instance
            # Always use the gender from the dataset item
            bench_item = QdataBenchItem(
                id=item["id"],
                original_id=item["original_id"],
                gender=item.get("gender", "male"),
                age=item.get("age", 0),  # Use age from dataset
                qalo_alif_len=st.session_state.qalo_alif_len,
                qalo_waw_len=st.session_state.qalo_waw_len,
                laa_alif_len=st.session_state.laa_alif_len,
                separate_madd=st.session_state.separate_madd,
                noon_moshaddadah_len=st.session_state.noon_moshaddadah_len,
                noon_mokhfah_len=st.session_state.noon_mokhfah_len,
                allam_alif_len=st.session_state.allam_alif_len,
                madd_aared_len=st.session_state.madd_aared_len,
                qalqalah=st.session_state.qalqalah,
                phonetic_transcript=st.session_state.phonetic_script,
                sifat=sifat_list,
            )

            # Save to JSON file
            save_annotation(item["id"], bench_item.model_dump())
            st.success(
                f"{st.session_state.lang_sett.annotation_saved} for ID: {item['id']}"
            )
            st.session_state.edit_mode = False

    with col3:
        # Edit button for current item (only shown if annotation exists)
        if item["id"] in st.session_state.annotations:
            if st.button(st.session_state.lang_sett.edit_annotation, type="secondary"):
                # Load the existing annotation
                annotation = st.session_state.annotations[item["id"]]
                st.session_state.phonetic_script = annotation.get(
                    "phonetic_transcript", ""
                )
                # Convert sifat to DataFrame
                sifat_list = annotation.get("sifat", [])
                st.session_state.sifat_df = pd.DataFrame(
                    [
                        sifa.model_dump() if isinstance(sifa, SifaOutput) else sifa
                        for sifa in sifat_list
                    ]
                )
                # Add row index column
                if not st.session_state.sifat_df.empty:
                    st.session_state.sifat_df.insert(
                        0, "row_index", range(1, len(st.session_state.sifat_df) + 1)
                    )
                # Load other fields
                st.session_state.gender = annotation.get("gender")
                st.session_state.qalo_alif_len = annotation.get("qalo_alif_len")
                st.session_state.qalo_waw_len = annotation.get("qalo_waw_len")
                st.session_state.laa_alif_len = annotation.get("laa_alif_len")
                st.session_state.separate_madd = annotation.get("separate_madd")
                st.session_state.noon_moshaddadah_len = annotation.get(
                    "noon_moshaddadah_len"
                )
                st.session_state.noon_mokhfah_len = annotation.get("noon_mokhfah_len")
                st.session_state.allam_alif_len = annotation.get("allam_alif_len")
                st.session_state.madd_aared_len = annotation.get("madd_aared_len")
                st.session_state.qalqalah = annotation.get("qalqalah")
                st.session_state.edit_mode = True
                st.success("Annotation loaded for editing")
        else:
            st.write("")  # Empty space for layout

    with col4:
        if (
            st.button(st.session_state.lang_sett.next)
            and st.session_state.index < len(ids) - 1
        ):
            st.session_state.index += 1
            # Reset for the new item
            reset_item_session_state()
            st.rerun()


def annotation_managment_view(ds, ids):
    # Export annotations
    st.divider()
    st.header(st.session_state.lang_sett.annotations_management)

    # Display current annotations with edit buttons
    if st.session_state.annotations:
        st.subheader(st.session_state.lang_sett.annotations_management)

        # Create a list of annotations with edit buttons
        annotations_list = []
        for item_id, annotation in st.session_state.annotations.items():
            # Find the index of this item in the dataset
            item_index = None
            for idx, original_idx in enumerate(ids):
                if ds[original_idx]["id"] == item_id:
                    item_index = idx
                    break

            # Use the correct key 'phonetic_transcript' instead of 'phonetic_script'
            phonetic_text = annotation.get("phonetic_transcript", "")
            truncated_phonetic = (
                phonetic_text[:50] + "..." if len(phonetic_text) > 50 else phonetic_text
            )

            # Get sifat count
            sifat_count = len(annotation.get("sifat", []))

            annotations_list.append(
                {
                    "id": item_id,
                    "phonetic_script": truncated_phonetic,
                    "sifat_count": sifat_count,
                    "item_index": item_index,
                }
            )

        # Create a dataframe for display
        annotation_df = pd.DataFrame(annotations_list)

        # Display the dataframe with edit buttons
        for _, row in annotation_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 5, 2, 2])
            with col1:
                st.write(f"**ID:** {row['id']}")
            with col2:
                st.write(f"**Phonetic:** {row['phonetic_script']}")
            with col3:
                st.write(f"**Sifat:** {row['sifat_count']}")
            with col4:
                if st.button("Edit", key=f"edit_{row['id']}"):
                    if row["item_index"] is not None:
                        st.session_state.index = row["item_index"]
                        st.session_state.edit_mode = True

                        # Load the annotation data
                        annotation = st.session_state.annotations[row["id"]]
                        st.session_state.phonetic_script = annotation.get(
                            "phonetic_transcript", ""
                        )
                        # Convert sifat to DataFrame
                        sifat_list = annotation.get("sifat", [])
                        st.session_state.sifat_df = pd.DataFrame(
                            [
                                sifa.model_dump()
                                if isinstance(sifa, SifaOutput)
                                else sifa
                                for sifa in sifat_list
                            ]
                        )
                        # Add row index column
                        if not st.session_state.sifat_df.empty:
                            st.session_state.sifat_df.insert(
                                0,
                                "row_index",
                                range(1, len(st.session_state.sifat_df) + 1),
                            )
                        # Load other fields
                        st.session_state.gender = annotation.get("gender")
                        st.session_state.qalo_alif_len = annotation.get("qalo_alif_len")
                        st.session_state.qalo_waw_len = annotation.get("qalo_waw_len")
                        st.session_state.laa_alif_len = annotation.get("laa_alif_len")
                        st.session_state.separate_madd = annotation.get("separate_madd")
                        st.session_state.noon_moshaddadah_len = annotation.get(
                            "noon_moshaddadah_len"
                        )
                        st.session_state.noon_mokhfah_len = annotation.get(
                            "noon_mokhfah_len"
                        )
                        st.session_state.allam_alif_len = annotation.get(
                            "allam_alif_len"
                        )
                        st.session_state.madd_aared_len = annotation.get(
                            "madd_aared_len"
                        )
                        st.session_state.qalqalah = annotation.get("qalqalah")
                        st.session_state.edit_mode = True
                        st.session_state.index = row["item_index"]
                        st.rerun()
                    else:
                        st.error(f"Could not find item {row['id']} in dataset")

        # Provide download link
        with open("annotations.json", "r", encoding="utf-8") as f:
            st.download_button(
                label=st.session_state.lang_sett.download_json,
                data=f,
                file_name="annotations.json",
                mime="application/json",
            )
    else:
        st.info(st.session_state.lang_sett.no_annotations)

    # Add a button to clear all annotations
    if st.button(st.session_state.lang_sett.clear_annotations):
        if st.session_state.annotations:
            st.session_state.annotations = {}
            Path("annotations.json").unlink(missing_ok=True)
            st.session_state.phonetic_script = ""
            st.session_state.sifat_df = pd.DataFrame()
            st.success(st.session_state.lang_sett.all_annotations_cleared)
            st.rerun()
        else:
            st.info(st.session_state.lang_sett.no_annotations)


def main():
    initialize()

    try:
        ds, ids = load_audio_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    sura_idx_to_name, sura_to_aya_count = get_sura_info()

    # Get current item
    current_id = ids[st.session_state.index]
    item = ds[current_id]
    load_item_to_session_state(item)

    # App layout
    select_en_lang = st.toggle(
        "Select English", value="english" == st.session_state.lang
    )
    if select_en_lang:
        st.session_state.lang = "english"
    else:
        st.session_state.lang = "arabic"
    load_language_settings()

    st.title(st.session_state.lang_sett.title)
    display_item(item, ids)

    uthmani_script = select_quran_text(
        sura_idx_to_name=sura_idx_to_name, sura_to_aya_count=sura_to_aya_count
    )

    default_moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=4,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=4,
    )
    annotate_phonetic_script(uthmani_script, default_moshaf=default_moshaf)

    annotate_addional_qdabenc_fields(item)

    save_navigatoin_bar(item, ids)

    annotation_managment_view(ds, ids)


if __name__ == "__main__":
    main()
