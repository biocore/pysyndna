from typing import Optional, Union, List

import biom
import pandas
import pandas as pd

DNA_BASEPAIR_G_PER_MOLE = 650
RNA_BASE_G_PER_MOLE = 340
NANOGRAMS_PER_GRAM = 1e9

# NB: sample_name instead of sample_id bc that's what qiita uses
SAMPLE_ID_KEY = 'sample_name'
SAMPLE_IN_ALIQUOT_MASS_G_KEY = 'calc_mass_sample_aliquot_input_g'
ELUTE_VOL_UL_KEY = 'vol_extracted_elution_ul'
REQUIRED_SAMPLE_INFO_KEYS = [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]


def _validate_sample_id_consistency(
        sample_ids_in_metadata: set,
        sample_ids_in_data: set,
        metadata_name: str,
        data_set_name: str) \
        -> Union[List[str], None]:
    """
    Checks that the sample ids in the metadata and data are consistent.

    Parameters
    ----------
    sample_ids_in_metadata: set
        A set of the sample ids in the metadata
    sample_ids_in_data: set
        A set of the sample ids in the data
    metadata_name: str
        A string identifying the metadata being checked, for use in error
        messages.
    data_set_name: str
        A string identifying the data set being checked, for use in error
        messages.

    Raises
    ------
    ValueError
        If there are sample ids in the data that aren't in the metadata

    Returns
    -------
    missing_sample_ids : set
        A set of sample ids that are in the metadata but not in the
        data.  Empty if all sample ids in the metadata were in the data.

    """

    # if there are sample ids in the data that are not in the metadata, raise
    # an error, since we don't know how to process that
    data_only_samples = sample_ids_in_data - sample_ids_in_metadata
    if len(data_only_samples) > 0:
        raise ValueError(
            f"Found sample ids in {data_set_name} that were "
            f"not in {metadata_name}: {data_only_samples}")

    # check if there are sample ids in the metadata that are not in the data
    # and if so, capture a list of them. Sometimes a sample just fails
    # sequencing and that shouldn't preclude processing the others that did
    # work, but we want to know about it.
    missing_sample_ids_set = sample_ids_in_metadata - sample_ids_in_data

    if len(missing_sample_ids_set) > 0:
        missing_sample_ids = list(missing_sample_ids_set)
    else:
        missing_sample_ids = None

    return missing_sample_ids


def validate_required_columns_exist(
        input_df: pd.DataFrame,
        required_cols_list: List[str],
        error_msg: str):

    """Checks that the input dataframe has the required columns.

    Parameters
    ----------
    input_df: pd.DataFrame
        A Dataframe to be checked.
    required_cols_list: list[str]
        List of column names that must be present in the dataframe.
    error_msg: str
        Error message to be raised if any of the required columns are missing.
    """

    missing_cols = set(required_cols_list) - set(input_df.columns)
    if len(missing_cols) > 0:
        missing_cols = sorted(missing_cols)
        raise ValueError(
            f"{error_msg}: {missing_cols}")


def validate_metadata_vs_reads_id_consistency(
        metadata_df: pd.DataFrame,
        reads_df: Union[pd.DataFrame, biom.Table]) \
        -> Union[List[str], None]:
    """
    Checks that the sample ids in the sample metadata and data are consistent.

    Parameters
    ----------
    metadata_df: pd.DataFrame
        A Dataframe containing at least SAMPLE_ID_KEY column
    reads_df: pd.DataFrame | biom.Table
        Either a Dataframe with a column for each SAMPLE_ID_KEY or a biom.Table
        with a column for each SAMPLE_ID_KEY

    Raises
    ------
    ValueError
        If there are sample ids in the data that aren't in the metadata df

    Returns
    -------
    missing_sample_ids : List[str] | None
        List of sample ids that are in the sample info but not in the
        data.  None if all sample ids in the experiment info were in the data.
    """

    sample_ids_in_metadata = set(metadata_df[SAMPLE_ID_KEY])
    if isinstance(reads_df, biom.Table):
        sample_ids_in_reads = set(reads_df.ids(axis='sample'))
    else:
        sample_ids_in_reads = set(reads_df.columns)
    missing_reads_ids = _validate_sample_id_consistency(
        sample_ids_in_metadata, sample_ids_in_reads, "sample info",
        "reads data")

    return missing_reads_ids


def validate_metadata_vs_prep_id_consistency(
        metadata_df: pd.DataFrame,
        prep_df: pd.DataFrame) \
        -> Union[List[str], None]:
    """
    Checks that sample ids in the sample metadata and prep info are consistent.

    Parameters
    ----------
    metadata_df: pd.DataFrame
        A Dataframe of sample metadata containing at least SAMPLE_ID_KEY column
    prep_df: pd.DataFrame
        A Dataframe of prep info with a column for SAMPLE_ID_KEY

    Raises
    ------
    ValueError
        If there are sample ids in prep info that aren't in sample metadata

    Returns
    -------
    missing_sample_ids : List[str] | None
        List of sample ids that are in the sample metadata but not in the
        prep info.  None if all sample ids in the sample metadata were in the
        prep info.
    """

    sample_ids_in_metadata = set(metadata_df[SAMPLE_ID_KEY])
    sample_ids_in_prep = set(prep_df[SAMPLE_ID_KEY])
    missing_prep_ids = _validate_sample_id_consistency(
        sample_ids_in_metadata, sample_ids_in_prep,
        "sample info",  "prep info")
    return missing_prep_ids


def cast_cols(params_df, float_col_names, cast_type=float):
    working_params_df = params_df.copy()

    # cast the contents of columns with input names to cast_type (e.g. float)
    # if they are in the dataframe and they aren't already that type
    for col in float_col_names:
        if col in working_params_df.columns:
            if working_params_df[col].dtype != cast_type:
                working_params_df[col] = \
                    working_params_df[col].astype(cast_type)

    return working_params_df


def calc_copies_genomic_element_per_g_series(
        genomic_elements_lengths_series: pd.Series,
        genomic_element_unit_avg_g_per_mole: float,
        is_test: Optional[bool] = False) -> pd.Series:

    """Calculates copies of genomic unit per gram of genomic element's unit.

    For example, get copies of OGU genomes per gram of double-stranded OGU gDNA
    or copies of OGU+ORF RNAs per gram of single-stranded OGU+ORF RNA.

    Parameters
    ----------
    genomic_elements_lengths_series: pd.Series
        A Series with index identifying each genomic element, containing length
        of each element in genomic element units.  For example, length in DNA
        basepairs for OGUs or length in (single-stranded) RNA bases for
        OGU+ORF RNAs.
    genomic_element_unit_avg_g_per_mole: float
        Average mass in grams per mole of a genomic element unit.  For example,
        650 g/mole for a DNA basepair or 340 g/mole for an RNA base.
    is_test: Optional[bool]
        Default is False.  If True, the function will use the less-precise
        value of Avogadro's number (6.022*(10^23)) used in cell [16] of the
        https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        notebook, rather than the more precise value (6.02214076*(10^23))
        calculation used if False.  This is True in testing ONLY.

    Returns
    -------
    ogu_genomes_series : pd.Series
        A Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU in the sequenced sample.

    Terminology:
    genomic_element: a distinct element measured on a genome such as an OGU
        (i.e., the whole genome) or an ORF on an OGU (called "OGU+ORF")
    genomic_element_unit: the units in which the genomic element is measured;
        in the case of OGUs, this is DNA basepairs, while in the case of
        OGU+ORFs, the units are RNA bases (i.e., single-stranded).

    This calculates the total number of copies of genomic element X per gram
    of genomic element units by the equation:

        Avogadro's number in (copies of genomic element X)/mole
    =	---------------------------------------------------------------
        length of genomic element X in genomic element units *
            average g/mole per genomic element unit

    Avogadro's number is 6.02214076 × 10^23 , and is the number of
    molecules--such as OGU genomes or OGU+ORF RNAs--in a mole of the genomic
    element.
    """

    # seems weird to make this a variable since it's famously a constant, but..
    avogadros_num = 6.02214076e23
    # this is done so we can test against Livia's results, which use
    # a truncated version of the constant. This should NOT be done in
    # production.  In testing, makes a difference of e.g., about 10 cells
    # out of 25K for the first OGU in the first sample in Livia's dataset.
    if is_test:
        avogadros_num = 6.022e23

    # TODO: do we have to worry about integer overflow here?
    #  Dan H. said, "if you use ints, the length * 650 * 10^9
    #  can overflow integers with very long genomes".  HOWEVER,
    #  the internet says that python *3* , "[o]nly floats have a hard
    #  limit in python. Integers are implemented as “long” integer
    #  objects of arbitrary size"(https://stackoverflow.com/a/52151786)
    #  HOWEVER HOWEVER, *numpy* integer types are fixed width, and
    #  "Some pandas and numpy functions, such as sum on arrays or
    #  Series return an np.int64 so this might be the reason you are
    #  seeing int overflows in Python3."
    #  (https://stackoverflow.com/a/58640340)
    #  What to do?

    denominator_series = \
        genomic_elements_lengths_series * genomic_element_unit_avg_g_per_mole

    copies_of_genomic_element_per_g_of_genomic_element_unit = \
        avogadros_num/denominator_series

    return copies_of_genomic_element_per_g_of_genomic_element_unit


def calc_gs_genomic_element_in_aliquot(
        genomic_elements_df: pd.DataFrame,
        genomic_element_conc_key: str,
        genomic_element_mass_key: str) -> pandas.DataFrame:

    working_df = genomic_elements_df.copy()

    # get the total grams of the genomic element that are in the elute after
    # extraction; this is sample-specific:
    # concentration of genomic element after extraction in ng/uL times
    # volume of elute from the extraction in uL, divided by 10^9 ng/g
    # (which is the same as multiplied by 1/10^9 g/ng)

    working_df[genomic_element_mass_key] = \
        working_df[genomic_element_conc_key] * \
        working_df[ELUTE_VOL_UL_KEY] / NANOGRAMS_PER_GRAM

    return working_df
