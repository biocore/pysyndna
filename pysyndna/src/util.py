from typing import Optional, Union, List

import biom
import numpy as np
import pandas
import pandas as pd

DNA_BASEPAIR_G_PER_MOLE = 650
RNA_BASE_G_PER_MOLE = 340
NANOGRAMS_PER_GRAM = 1e9

# NB: sample_name instead of sample_id bc that's what qiita uses
SAMPLE_ID_KEY = 'sample_name'
OGU_ID_KEY = 'ogu_id'
SAMPLE_IN_ALIQUOT_MASS_G_KEY = 'calc_mass_sample_aliquot_input_g'
ELUTE_VOL_UL_KEY = 'vol_extracted_elution_ul'


def get_ids_from_df_or_biom(
        df_or_biom: Union[pd.DataFrame, biom.Table],
        get_sample_ids=True) -> list[str]:
    """
    Gets sample or OGU ids from a dataframe or a biom table.

    Parameters
    ----------
    df_or_biom: pd.DataFrame | biom.Table
        For sample ids, a DataFrame with a SAMPLE_ID_KEY column or a column for
        each sample id, or a biom.Table with sample ids on the sample axis.
        For OGU ids, a DataFrame with an OGU_ID_KEY column or a biom.Table
        with OGU ids on the observation axis.
    get_sample_ids: bool
        If True, gets sample ids; if False, gets OGU ids. Default is True.

    Returns
    -------
    found_ids : list[str]
        A list of ids of the specified type in the dataframe or biom table.
    """

    if isinstance(df_or_biom, biom.Table):
        axis = 'sample' if get_sample_ids else 'observation'
        ids = [x for x in df_or_biom.ids(axis=axis)]
    else:
        col_name = SAMPLE_ID_KEY if get_sample_ids else OGU_ID_KEY
        col_name_present = col_name in df_or_biom.columns
        if col_name_present:
            # if the column is present, we just get the ids from that column
            ids = df_or_biom[col_name].tolist()
        else:
            if get_sample_ids:
                # if there isn't a sample id column, we assume that the
                # DataFrame has one column per sample id in addition to a
                # column for OGU ids, so we get the sample ids from the
                # DataFrame's columns
                ids = df_or_biom.columns.tolist()
                if OGU_ID_KEY in ids:
                    ids.remove(OGU_ID_KEY)
            else:
                # in a dataframe, OGU ids are always in a column named
                # OGU_ID_KEY, so if there isn't one, this is an error
                raise ValueError(
                    f"DataFrame does not have a column named '{col_name}'")
            # endif we are/aren't getting sample ids
        # endif there is/isn't an explicit column for the desired ids
    # endif df_or_biom is biom.Table or pd.DataFrame

    # convert to string in case the ids are integers or other types
    found_ids = [str(x) for x in ids]
    return found_ids


def _validate_id_consistency(
        superset_ids: set,
        subset_ids: set,
        superset_name: str,
        subset_name: str,
        id_type: str) \
        -> Union[List[str], None]:
    """
    Checks that the ids in the superset and subset are consistent.

    Parameters
    ----------
    superset_ids: set
        A set of the superset of required ids
    subset_ids: set
        A set of the subset of required ids
    superset_name: str
        A string identifying the superset dataset, for use in error messages.
    subset_name: str
        A string identifying the subset dataset, for use in error messages.
    id_type: str
        A string identifying the type of id being checked, for use in error
        messages.

    Raises
    ------
    ValueError
        If there are ids in the subset that aren't in the superset

    Returns
    -------
    missing_ids : set
        A set of ids that are in the superset but not in the
        subset.  Empty if all supersets ids are also in the subset.

    """

    # if there are ids in the subset that are not in the superset, raise
    # an error, since we don't know how to process that
    subset_only_ids = subset_ids - superset_ids
    if len(subset_only_ids) > 0:
        raise ValueError(
            f"Found {id_type} ids in {subset_name} that were "
            f"not in {superset_name}: {subset_only_ids}")

    # check if there are ids in the superset that are not in the subset
    # and if so, capture a list of them. For example, sometimes a sample just
    # fails sequencing and that shouldn't preclude processing the others that
    # did work, but we want to know about it.
    missing_superset_set = superset_ids - subset_ids

    if len(missing_superset_set) > 0:
        missing_ids = list(missing_superset_set)
    else:
        missing_ids = None

    return missing_ids


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


def validate_id_consistency_between_datasets(
        superset_df_or_biom: Union[pd.DataFrame, biom.Table],
        subset_df_or_biom: Union[pd.DataFrame, biom.Table],
        superset_name: str,
        subset_name: str,
        check_sample_ids=True) \
        -> Union[List[str], None]:
    """
    Checks sample or OGU ids are consistent in the superset and subset datasets

    Parameters
    ----------
    superset_df_or_biom: Union[pd.DataFrame, biom.Table]
        A Dataframe or biom.Table containing the superset of required ids of
        specified type (sample or OGU) either in a column or as column names.
    subset_df_or_biom: Union[pd.DataFrame, biom.Table]
        A Dataframe or biom.Table containing the subset of required ids of
        specified type (sample or OGU) either in a column or as column names.
    superset_name: str
        A string identifying the superset dataset, for use in error messages.
    subset_name: str
        A string identifying the subset dataset, for use in error messages.
    check_sample_ids: bool
        If True, checks sample ids; if False, checks OGU ids. Default is True.

    Raises
    ------
    ValueError
        If there are ids of the specified type in the subset that aren't in
        the superset.

    Returns
    -------
    missing_ids : List[str] | None
        List of ids of the specified type that are in the superset but not in
        the subset.  None if all superset ids are also in the subset.
    """

    ids_in_superset = set(
        get_ids_from_df_or_biom(superset_df_or_biom, check_sample_ids))
    ids_in_subset = set(
        get_ids_from_df_or_biom(subset_df_or_biom, check_sample_ids))

    id_type = "sample" if check_sample_ids else "OGU"
    missing_ids = _validate_id_consistency(
        ids_in_superset, ids_in_subset, superset_name, subset_name, id_type)

    return missing_ids


def cast_cols(params_df, numeric_col_names, force_float=False):
    working_params_df = params_df.copy()

    # cast the contents of columns with input names to cast_type (e.g. float)
    # if they are in the dataframe and they aren't already that type
    for col in numeric_col_names:
        if col in working_params_df.columns:
            working_params_df[col] = pandas.to_numeric(
                working_params_df[col], errors='coerce')
            if force_float:
                working_params_df[col] = \
                    working_params_df[col].astype(float)

    return working_params_df


def filter_data_by_sample_info(
        quant_params_per_sample_df: pandas.DataFrame,
        a_data_table: Union[biom.Table, pd.DataFrame],
        param_cols_to_filter_on: List[str]) -> \
        (Union[biom.Table, pd.DataFrame], List[str]):
    """Filter samples with NaNs in necessary param column(s) from table.

    Parameters
    ----------
    quant_params_per_sample_df : pandas.DataFrame
        A DataFrame containing combined sample and prep info, with sample id
        as the index.
    a_data_table : biom.Table | pd.DataFrame
        A biom.Table with the values for each sample or a dataframe with
        columns for each sample.
    param_cols_to_filter_on : list[str]
        A list of column names in quant_params_per_sample_df that are
        necessary and should not have NaNs or negatives in them.

    Returns
    -------
    filtered_table : biom.Table | pd.DataFrame
        If input was a biom, a biom.Table with values for each sample that is
        NOT NaN or negative in any of the necessary prep/sample columns.  If
        input was a dataframe, a dataframe with columns for each sample that
        is NOT NaN or negative in any of the necessary prep/sample columns.
    log_msgs_list: list[str]
        A list of log messages, if any, generated during the function's
        operation.  Empty if no log messages were generated.
    """

    log_msgs_list = []
    remaining_nans_msg = "There are NaNs remaining in the filtered table."

    # identify samples w/NaNs in sample/prep info columns we need to use
    # (such as, frequently, blanks)
    nan_samples = quant_params_per_sample_df[
        quant_params_per_sample_df[param_cols_to_filter_on].isna().any(
            axis=1)].index.tolist()

    neg_samples = quant_params_per_sample_df[
        quant_params_per_sample_df[param_cols_to_filter_on].lt(0).any(
            axis=1)].index.tolist()

    problem_samples = list(set(nan_samples) | set(neg_samples))

    if len(problem_samples) > 0:
        if isinstance(a_data_table, biom.Table):
            def is_problem(val, id_, _):
                return id_ in problem_samples
            filtered_table = a_data_table.filter(
                is_problem, invert=True, inplace=False)

            # check if there are any NaNs left in the biom table (presumably
            # from causes other than NaN sample/prep info); if so, error
            # (this check for NaNs within the biom table suggested by @wasade)
            if np.isnan(filtered_table.matrix_data.data).any():
                raise ValueError(remaining_nans_msg)
        elif isinstance(a_data_table, pd.DataFrame):
            filtered_table = a_data_table.drop(problem_samples, axis=1)
            if filtered_table.isnull().values.any():
                raise ValueError(remaining_nans_msg)
        else:
            raise ValueError(
                "a_data_table must be a biom.Table or a pd.DataFrame")

        if len(nan_samples) > 0:
            log_msgs_list.append(
                "Dropping samples with NaNs in necessary "
                "prep/sample column(s): " + ", ".join(nan_samples))

        if len(neg_samples) > 0:
            log_msgs_list.append(
                "Dropping samples with negative values in necessary "
                "prep/sample column(s): " + ", ".join(neg_samples))
    else:
        filtered_table = a_data_table

    return filtered_table, log_msgs_list


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
