import biom
import numpy as np
import pandas as pd
import yaml
from typing import Optional, Union, Dict, List
from pysyndna.src.util import calc_copies_genomic_element_per_g_series, \
    calc_gs_genomic_element_in_aliquot, \
    validate_required_columns_exist, \
    validate_id_consistency_between_datasets, filter_data_by_sample_info, \
    get_ids_from_df_or_biom, cast_cols, \
    DNA_BASEPAIR_G_PER_MOLE, NANOGRAMS_PER_GRAM, \
    SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY, ELUTE_VOL_UL_KEY, OGU_ID_KEY

from pysyndna.src.fit_syndna_models import INPUT_SYNDNA_POOL_MASS_NG_KEY, \
    SLOPE_KEY, INTERCEPT_KEY

DEFAULT_SYNDNA_MASS_FRACTION_OF_SAMPLE = 0.05
DEFAULT_READ_LENGTH = 150
DEFAULT_MIN_RSQUARED = 0.8

CELL_COUNT_RESULT_KEY = 'cell_count_biom'
CELL_COUNT_LOG_KEY = 'calc_cell_counts_log'

GDNA_CONCENTRATION_NG_UL_KEY = 'extracted_gdna_concentration_ng_ul'
GDNA_FROM_ALIQUOT_MASS_G_KEY = 'extracted_gdna_mass_g'
# NB: below is NOT the full mass of gDNA extracted from the sample (which can
# be calculated from GDNA_CONCENTRATION_NG_UL_KEY and ELUTE_VOL_UL_KEY
# and then stored in GDNA_FROM_ALIQUOT_MASS_G_KEY) but
# ONLY the mass of gDNA that was put into sequencing . This mass should
# NOT include the additional mass of the syndna pool added to sequencing.
SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY = 'sequenced_sample_gdna_mass_ng'
OGU_READ_COUNT_KEY = 'ogu_read_count'
LOG_10_OGU_READ_COUNT_KEY = 'log10_ogu_read_count'
OGU_PERCENT_COVERAGE_KEY = 'percent_coverage_of_ogu'
OGU_AGNOSTIC_COVERAGE_KEY = 'coverage_of_ogu'
LOG_10_OGU_GDNA_MASS_NG_KEY = 'log10_ogu_gdna_mass_ng'
OGU_LEN_IN_BP_KEY = 'ogu_len_in_bp'
OGU_GDNA_MASS_NG_KEY = 'ogu_gdna_mass_ng'
OGU_GENOMES_PER_G_OF_GDNA_KEY = 'ogu_genomes_per_g_of_gdna'
OGU_CELLS_PER_G_OF_GDNA_KEY = 'ogu_cells_per_g_of_gdna'
OGU_CELLS_PER_G_OF_SAMPLE_KEY = 'ogu_cells_per_g_of_sample'
# NB: below is based on the full mass of gDNA extracted from the sample
# (NOT limited to the amount of gDNA that was put into sequencing, unlike
# SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY)
GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY = 'gdna_mass_to_sample_mass_ratio'
SAMPLE_VOLUME_UL_KEY = "sample_volume_ul"
OGU_CELLS_PER_UL_OF_SAMPLE_KEY = "ogu_cells_per_ul_of_sample"
GDNA_MASS_TO_SAMPLE_VOL_RATIO_KEY = "gdna_mass_to_sample_vol_ratio"
SAMPLE_SURFACE_AREA_CM2_KEY = "sample_surface_area_cm2"
OGU_CELLS_PER_CM2_OF_SAMPLE_KEY = "ogu_cells_per_cm2_of_sample"
GDNA_MASS_TO_SAMPLE_SURFACE_AREA_RATIO_KEY = "gdna_mass_to_sample_surface_area_ratio"
REQUIRED_DNA_PREP_INFO_KEYS = [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
                               ELUTE_VOL_UL_KEY]
RATIO_NAME_KEY = "ratio_key"
DENOMINATOR_KEY = "denom_key"
SAMPLE_LEVEL_METRICS_DICT = {
    OGU_CELLS_PER_G_OF_SAMPLE_KEY: {
        DENOMINATOR_KEY: SAMPLE_IN_ALIQUOT_MASS_G_KEY,
        RATIO_NAME_KEY: GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY},
    OGU_CELLS_PER_UL_OF_SAMPLE_KEY: {
        DENOMINATOR_KEY: SAMPLE_VOLUME_UL_KEY,
        RATIO_NAME_KEY: GDNA_MASS_TO_SAMPLE_VOL_RATIO_KEY},
    OGU_CELLS_PER_CM2_OF_SAMPLE_KEY: {
        DENOMINATOR_KEY: SAMPLE_SURFACE_AREA_CM2_KEY,
        RATIO_NAME_KEY: GDNA_MASS_TO_SAMPLE_SURFACE_AREA_RATIO_KEY}
}

_METATDATA_NAME = "sample info"
_COUNTS_DATA_NAME = "OGU counts data"
_COVERAGE_DATA_NAME = "OGU coverage data"
_LENGTHS_DATA_NAME = "OGU lengths info"


def _generate_ogu_coverages_per_sample_df(
        ogu_coverage_df: pd.DataFrame,
        ogu_counts_per_sample_biom: biom.Table) -> pd.DataFrame:
    """Generates a DataFrame of OGU coverage per sample.

    Parameters
    ----------
    ogu_coverage_df : pd.DataFrame
        A DataFrame containing a column for OGU_ID_KEY and either a column for
        OGU_PERCENT_COVERAGE_KEY (indicating the coverage is the same for all
        samples) or a column for each sample id, which holds either the fraction 
        or the percent coverage of that OGU in that sample. 
        NOTE THAT IT IS UP TO THE USER TO ENSURE THAT THEY KNOW WHICH TYPE OF 
        VALUE (fraction or percent) IS BEING USED AND THAT THEY PROVIDE THE
        APPROPRIATE MIN_COVERAGE PARAMETER (e.g., 0.01 or 1 to drop <1% coverage).
    ogu_counts_per_sample_biom : biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.

    Returns
    -------
    ogu_coverage_per_sample_df : pd.DataFrame
        A DataFrame with an OGU_ID_KEY column and one column for each sample
        id, which holds the coverage (as either a fraction or a percent, based
        on the input) of that OGU in that sample.
    """

    # default assumption: the df is already in the right (per-sample) format
    ogu_coverage_per_sample_df = ogu_coverage_df.copy()
    # BUT if there is an OGU_PERCENT_COVERAGE_KEY column, then the df isn't yet
    # per-sample and we have to copy that percent coverage for all samples
    if OGU_PERCENT_COVERAGE_KEY in ogu_coverage_df.columns:
        sample_ids = get_ids_from_df_or_biom(ogu_counts_per_sample_biom)
        for curr_sample_id in sample_ids:
            if curr_sample_id in ogu_coverage_df.columns:
                # this is an unrecognized format; don't know how to parse it
                raise ValueError(f"OGU coverage data contains both"
                                 f"{OGU_PERCENT_COVERAGE_KEY} and a column "
                                 f"with a sample name: '{curr_sample_id}'.")
            # endif

            ogu_coverage_per_sample_df[curr_sample_id] = \
                ogu_coverage_df[OGU_PERCENT_COVERAGE_KEY]
        # next sample_id

        # extract only the ogu id column and the columns of sample ids
        desired_cols = [OGU_ID_KEY] + sample_ids
        ogu_coverage_per_sample_df = \
            ogu_coverage_per_sample_df.loc[:, desired_cols]
    # endif

    # cast all columns EXCEPT the OGU_ID_KEY column to float
    float_col_names = list(ogu_coverage_per_sample_df.columns)
    float_col_names.remove(OGU_ID_KEY)
    ogu_coverage_per_sample_df = cast_cols(ogu_coverage_per_sample_df, float_col_names, True)

    return ogu_coverage_per_sample_df

def _validate_sample_ids_in_inputs(
        absolute_quant_params_per_sample_df: pd.DataFrame,
        ogu_counts_per_sample_biom: biom.Table,
        ogu_coverage_per_sample_df: pd.DataFrame) -> None:

    """Validates that the sample ids in the inputs are consistent.

    Parameters
    ----------
    absolute_quant_params_per_sample_df: pd.DataFrame
        A Dataframe of metadata parameters for each sample, including a
        SAMPLE_ID_KEY column.
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_coverage_per_sample_df : pd.DataFrame
        A DataFrame containing a column for OGU_ID_KEY and a column for each
        sample id, which holds the coverage (as either a fraction or a percent, based
        on the input) of that OGU in that sample.

    Raises
    ------
    ValueError
        If the sample ids in the absolute quant params per sample,
        ogu counts per sample, and/or ogu percent coverages by sample
        are not consistent.
    """

    # Check if any samples in the reads data are missing from the metadata;
    # Not bothering to report samples that are in metadata but not the reads--
    # maybe those failed the sequencing run.
    _ = validate_id_consistency_between_datasets(
        absolute_quant_params_per_sample_df, ogu_counts_per_sample_biom,
        _METATDATA_NAME, _COUNTS_DATA_NAME,  check_sample_ids=True)

    # Check that every sample in the reads data is also in the ogu coverages.
    # Not worrying about samples that are in coverages but not the reads.
    _ = validate_id_consistency_between_datasets(
        ogu_coverage_per_sample_df, ogu_counts_per_sample_biom,
        _COVERAGE_DATA_NAME, _COUNTS_DATA_NAME, check_sample_ids=True)

    # Not checking that all the samples in the coverages data are in the
    # metadata or vice versa because we don't really care about any of them
    # that aren't *also* in the reads data, and we've already checked those for
    # consistency.


def _validate_ogu_ids_in_inputs(
        ogu_counts_per_sample_biom: biom.Table,
        ogu_coverage_per_sample_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame) -> None:
    """Validates that the OGU ids in the inputs are consistent.

    Parameters
    ----------
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_coverage_per_sample_df : pd.DataFrame
        A DataFrame containing a column for OGU_ID_KEY and a column for each
        sample id, which holds the coverage of that OGU in that sample as either
        a fraction or a percent.
    ogu_lengths_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.

    Raises
    ------
    ValueError
        If the OGU ids in the ogu counts per sample, ogu coverage per
        sample, and/or ogu lengths are not consistent.
    """

    # Check that every ogu in the reads data is also in the ogu lengths;
    # Not bothering to report ogus that are in lengths but not the reads--
    # maybe those just don't exist in these samples.
    _ = validate_id_consistency_between_datasets(
        ogu_lengths_df, ogu_counts_per_sample_biom,
        _LENGTHS_DATA_NAME, _COUNTS_DATA_NAME,  check_sample_ids=False)

    # Check that every ogu in the reads data is also in the ogu coverages;
    # Not bothering to report ogus that are in coverages but not the reads--
    # can imagine having zero-coverage OGUs included there (but not verifying
    # that any OGUs in the coverages data that are missing from the
    # reads data actually have zero coverage, bc one has to stop somewhere :)
    _ = validate_id_consistency_between_datasets(
        ogu_coverage_per_sample_df, ogu_counts_per_sample_biom,
        _COVERAGE_DATA_NAME, _COUNTS_DATA_NAME, check_sample_ids=False)

    # Not checking that all the ogus in the coverages data are in the
    # lengths because we don't really care about any of them that aren't *also*
    # in the reads data, and we've already checked those for consistency.

def _calc_ogu_cell_counts_per_x_of_sample_for_qiita(
        sample_info_df: pd.DataFrame,
        prep_info_df: pd.DataFrame,
        linregress_by_sample_id_fp: str,
        ogu_counts_per_sample_biom: biom.Table,
        ogu_coverage_df: pd.DataFrame,
        ogu_lengths_fp: str,
        output_cell_counts_metric: str,
        min_coverage: float,
        min_rsquared: float = DEFAULT_MIN_RSQUARED,
        syndna_mass_fraction_of_sample: float =
        DEFAULT_SYNDNA_MASS_FRACTION_OF_SAMPLE) \
        -> Dict[str, Union[str, biom.Table]]:

    """Gets # of cells of each OGU/g of sample for samples from Qiita.

    Parameters
    ----------
    sample_info_df: pd.DataFrame
        A Dataframe containing sample info for all samples in the prep,
        including SAMPLE_ID_KEY and SAMPLE_IN_ALIQUOT_MASS_G_KEY
    prep_info_df: pd.DataFrame
        A Dataframe containing prep info for all samples in the prep,
        including SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY, and
        ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY.
    linregress_by_sample_id_fp: str
        String containing the filepath to the yaml file holding the
        dictionary keyed by sample id, containing for each sample a dictionary
        representation of the sample's LinregressResult.
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_coverage_df : pd.DataFrame
        A DataFrame containing a column for OGU_ID_KEY and either a column for
        OGU_PERCENT_COVERAGE_KEY (indicating the coverage is the same for all
        samples) or a column for each sample id, which holds the coverage of 
        that OGU in that sample, expressed as either a fraction or a percent.
        NOTE THAT IT IS UP TO THE USER TO ENSURE THAT THEY KNOW WHICH TYPE OF
        VALUE (fraction or percent) IS BEING USED AND THAT THEY PROVIDE THE
        APPROPRIATE min_coverage PARAMETER (e.g., 0.01 or 1 to drop <1% coverage).
    ogu_lengths_fp : str
        String containing the filepath to a tab-separated, two-column,
        no-header file in which the first column is the OGU id and the
         second is the OGU length in basepairs
    min_coverage : float
        Minimum allowable coverage of an OGU in a sample needed to include
        that OGU/sample in the output, expressed in the same units
        (fraction or percent) as used in ogu_coverage_df.
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.
    syndna_mass_fraction_of_sample: float
        Fraction of the mass of the sample that is added as syndna (usually
        0.05, which is to say 5%).

    Returns
    -------
    output_by_out_type : dict of str or biom.Table
        Dictionary of outputs keyed by their type Currently, the following keys
        are defined:
        CELL_COUNT_RESULT_KEY: biom.Table holding the calculated number of
        cells per gram of sample material for each OGU in each sample.
        CELL_COUNT_LOG_KEY: log of messages from the cell count calc process.
    """

    required_prep_cols = list(
        {INPUT_SYNDNA_POOL_MASS_NG_KEY} | set(REQUIRED_DNA_PREP_INFO_KEYS))
    validate_required_columns_exist(
        prep_info_df, required_prep_cols,
        "prep info is missing required column(s)")

    # Check if any samples in the prep are missing from the sample info;
    # Not bothering to report samples that are in sample info but not the prep
    # --maybe those just weren't included in this prep.
    _ = validate_id_consistency_between_datasets(
        sample_info_df, prep_info_df, "sample info", "prep info", True)

    # cast in case the input comes in as string or something
    syndna_mass_fraction_of_sample = float(syndna_mass_fraction_of_sample)

    # TODO: replace this with just taking in the measured sample gdna mass
    # ensure INPUT_SYNDNA_POOL_MASS_NG_KEY column of prep_info_df is a float,
    # then calculate the mass of gDNA sequenced for each sample.  We have the
    # mass of syndna pool that was added to each sample, and we know that the
    # syndna pool mass is calculated to be a certain percentage of the mass of
    # the sample (added into the library prep in addition to the sample mass).
    # Therefore, if the syndna fraction is 0.05 or 5%, the mass of the sample
    # gDNA put into sequencing is 1/0.05 = 20x the mass of syndna pool added.
    prep_info_df = cast_cols(
        prep_info_df, [INPUT_SYNDNA_POOL_MASS_NG_KEY], True)
    prep_info_df[SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY] = \
        prep_info_df[INPUT_SYNDNA_POOL_MASS_NG_KEY] * \
        (1 / syndna_mass_fraction_of_sample)

    # merge the sample info and prep info dataframes
    absolute_quant_params_per_sample_df = \
        sample_info_df.merge(prep_info_df, on=SAMPLE_ID_KEY, how='left')

    # read in the linregress_by_sample_id yaml file
    with open(linregress_by_sample_id_fp) as f:
        linregress_by_sample_id = yaml.load(f, Loader=yaml.FullLoader)

    # read in the ogu_lengths file
    ogu_lengths_df = pd.read_csv(ogu_lengths_fp, sep='\t', header=None,
                                 names=[OGU_ID_KEY, OGU_LEN_IN_BP_KEY])

    # calculate # cells per x (g, uL, or cm2) of sample material of each OGU
    # in each sample
    output_biom, log_msgs_list = calc_ogu_cell_counts_biom(
        absolute_quant_params_per_sample_df, linregress_by_sample_id,
        ogu_counts_per_sample_biom, ogu_coverage_df, ogu_lengths_df,
        min_coverage, min_rsquared, output_cell_counts_metric)

    out_txt_by_out_type = {
        CELL_COUNT_RESULT_KEY: output_biom,
        CELL_COUNT_LOG_KEY: '\n'.join(log_msgs_list)}

    return out_txt_by_out_type


def _calc_long_format_ogu_cell_counts_df(
        linregress_by_sample_id: Dict[str, Dict[str, float]],
        ogu_counts_per_sample_df: pd.DataFrame,
        ogu_coverage_per_sample_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        per_sample_calc_info_df: pd.DataFrame,
        min_coverage: float,
        min_rsquared: float) -> (Union[pd.DataFrame, None], List[str]):

    """Predicts the # of cells of each OGU in each sample from the read counts.

    Parameters
    ----------
    linregress_by_sample_id : dict[str, dict[str, float]]
        Dictionary keyed by sample id, containing for each sample either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a dictionary
        representation of the sample's LinregressResult.
    ogu_counts_per_sample_df: pd.DataFrame
        A Dataframe with a column for OGU_ID_KEY and then one additional column
        for each sample id, which holds the read counts aligned to that OGU in
        that sample.
    ogu_coverage_per_sample_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and a column for each sample holding the
        coverage for that OGU in that sample as either a fraction or a percent.
        NOTE THAT IT IS UP TO THE USER TO ENSURE THAT THEY KNOW WHICH TYPE OF
        VALUE (fraction or percent) IS BEING USED AND THAT THEY PROVIDE THE
        APPROPRIATE min_coverage PARAMETER (e.g., 0.01 or 1 to drop <1% coverage).
    ogu_lengths_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    per_sample_calc_info_df : pd.DataFrame
        A Dataframe of SAMPLE_ID_KEY, SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY,
        GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY, GDNA_MASS_TO_SAMPLE_VOL_RATIO_KEY,
        and GDNA_MASS_TO_SAMPLE_SURFACE_AREA_RATIO_KEY for each sample. Any or
        all of the ratio columns may be NaN for a given sample.
    min_coverage : float
        Minimum allowable coverage of an OGU needed to include that OGU
        in the output, expressed in the same units (fraction or percent)
        as used in ogu_coverage_per_sample_df.
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.

    Returns
    -------
    ogu_cell_counts_df : pd.DataFrame | None
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        OGU_CELLS_PER_G_OF_GDNA_KEY, and OGU_CELLS_PER_G_OF_GDNA_KEY,
        in addition to various intermediate calculation columns.  Could be
        None if no cell counts were calculated for any sample.
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    log_messages_list = []

    # reformat biom info into a "long format" table with
    # columns needed for per-sample calculation
    working_df, prep_log_messages = _prepare_cell_counts_calc_df(
        ogu_counts_per_sample_df, ogu_coverage_per_sample_df,
        ogu_lengths_df, min_coverage)
    log_messages_list.extend(prep_log_messages)

    # loop through a series of the unique sample ids in the working_df
    cell_counts_df = None
    sample_ids = working_df[SAMPLE_ID_KEY].unique()
    for curr_sample_id in sample_ids:
        # calculate the predicted number of cells of each OGU per gram of
        # gDNA in this sample and also per gram of stool in this sample
        curr_sample_df, curr_log_msgs = _calc_ogu_cell_counts_df_for_sample(
            curr_sample_id, linregress_by_sample_id,
            per_sample_calc_info_df, working_df, min_rsquared)
        log_messages_list.extend(curr_log_msgs)
        if curr_sample_df is None:
            log_messages_list.append(f"No cell counts calculated for "
                                     f"sample {curr_sample_id}")

            # NB: if no cell counts were calculated for this sample,
            # this sample is left out of the final cell_counts_df.
            continue

        # if cell_counts_df does not yet exist, create it from curr_sample_df;
        # otherwise, append curr_sample_df to the existing cell_counts_df
        if cell_counts_df is None:
            cell_counts_df = curr_sample_df
        else:
            # append the current sample's df to the existing cell_counts_df
            cell_counts_df = pd.concat([cell_counts_df, curr_sample_df])
    # next sample_id

    if cell_counts_df is None:
        raise ValueError("No cell counts calculated for any sample")

    return cell_counts_df, log_messages_list


def _prepare_cell_counts_calc_df(
        ogu_counts_per_sample_df: pd.DataFrame,
        ogu_coverage_per_sample_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        min_coverage: float) -> (pd.DataFrame, List[str]):

    """Prepares long-format dataframe containing fields needed for later calcs.

    Parameters
    ----------
    ogu_counts_per_sample_df: pd.DataFrame
        Wide-format dataframe with ogu ids as index and one
        column for each sample id, which holds the read counts
        aligned to that OGU in that sample.
    ogu_coverage_per_sample_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and a column for each sample holding the
        coverage for that OGU in that sample as either a fraction or a percent.
    ogu_lengths_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    min_coverage : float
        Minimum allowable coverage of an OGU across samples needed to include
        that OGU in the output, expressed in the same units (fraction or percent)
        as used in ogu_coverage_per_sample_df.

    Returns
    -------
    working_df : pd.DataFrame
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        OGU_READ_COUNT_KEY, OGU_LEN_IN_BP_KEY, and OGU_AGNOSTIC_COVERAGE_KEY;
        contains rows for OGU/sample combinations with
        OGU coverage >= min_coverage
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    log_messages_list = []

    # cast scalar inputs in case they come in as strings or something
    min_coverage = float(min_coverage)

    # move the ogu ids from the index to a column, bc I hate implicit
    working_df = ogu_counts_per_sample_df.copy()
    working_df = working_df.reset_index(names=[OGU_ID_KEY])

    # reformat biom info into a "long format" table with
    # columns for OGU_ID_KEY, SAMPLE_ID_KEY, OGU_READ_COUNT_KEY
    working_df = working_df.melt(
        id_vars=[OGU_ID_KEY], var_name=SAMPLE_ID_KEY,
        value_name=OGU_READ_COUNT_KEY)

    # reformat the ogu coverage per sample info into a "long format"
    # table w columns for OGU_ID_KEY, SAMPLE_ID_KEY, OGU_AGNOSTIC_COVERAGE_KEY
    working_coverage_df = \
        ogu_coverage_per_sample_df.melt(
            id_vars=[OGU_ID_KEY], var_name=SAMPLE_ID_KEY,
            value_name=OGU_AGNOSTIC_COVERAGE_KEY)
    # merge the working_df with the working_coverage_df to add the
    # coverage of each OGU in each sample
    working_df = working_df.merge(
        working_coverage_df, on=[OGU_ID_KEY, SAMPLE_ID_KEY], how='left')

    # add a column for OGU_LEN_IN_BP_KEY (yes, this will be repeated,
    # but it is convenient to have everything in one table)
    working_df = working_df.merge(ogu_lengths_df, on=OGU_ID_KEY, how='left')

    # drop records for OGUs with coverage < min_coverage
    too_low_cov_mask = working_df[OGU_AGNOSTIC_COVERAGE_KEY] < min_coverage
    too_low_cov_ogus_list = (
        working_df.loc[too_low_cov_mask, OGU_ID_KEY].unique().tolist())
    if len(too_low_cov_ogus_list) > 0:
        log_messages_list.append(f'The following items have coverage lower'
                                 f' than the minimum of {min_coverage}: '
                                 f'{too_low_cov_ogus_list}')
    working_df = working_df[
        working_df[OGU_AGNOSTIC_COVERAGE_KEY] >= min_coverage]
    working_df = working_df.reset_index(drop=True)

    return working_df, log_messages_list


def _calc_ogu_cell_counts_df_for_sample(
        sample_id: str,
        linregress_by_sample_id: Dict[str, Dict[str, float]],
        per_sample_info_df: pd.DataFrame,
        working_df: pd.DataFrame,
        min_rsquared: float,
        is_test: Optional[bool] = False) \
        -> (Union[pd.DataFrame, None], List[str]):

    """Calculates # cells of each OGU per gram of sample material for sample.

    Parameters
    ----------
    sample_id: str
        Sample id for which to calculate cell counts.
    linregress_by_sample_id : dict[str, dict[str: float]]
        Dictionary keyed by sample id, containing for each sample either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a dictionary
        representation of the sample's LinregressResult.
    per_sample_info_df : pd.DataFrame
        A Dataframe of SAMPLE_ID_KEY, SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY,
        GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY, GDNA_MASS_TO_SAMPLE_VOL_RATIO_KEY,
        and GDNA_MASS_TO_SAMPLE_SURFACE_AREA_RATIO_KEY for each sample. Any or
        all of the ratio columns may be NaN for a given sample.
    working_df : pd.DataFrame
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        OGU_READ_COUNT_KEY, and OGU_LEN_IN_BP_KEY
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.
    is_test: Optional[bool]
        Default is False.  If True, the function will use the less-precise
        value of Avogadro's number (6.022*(10^23)) used in
        SynDNA_saliva_samples_analysis.ipynb.  Otherwise, the more precise
        value (6.02214076*(10^23)) will be used.  This is True in testing ONLY.

    Returns
    -------
    sample_df : pd.DataFrame | None
        None if the specified sample id has no linear model or has a model with
        R^2 < min_rsquared.  Otherwise, a long-format dataframe with columns
        for at least OGU_ID_KEY, SAMPLE_ID_KEY, OGU_READ_COUNT_KEY,
        OGU_LEN_IN_BP_KEY,
        OGU_GENOMES_PER_G_OF_GDNA_KEY, OGU_CELLS_PER_G_OF_GDNA_KEY,
        OGU_CELLS_PER_G_OF_SAMPLE_KEY, OGU_CELLS_PER_UL_OF_SAMPLE_KEY, and
        OGU_CELLS_PER_CM2_OF_SAMPLE_KEY
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    log_messages_list = []

    # cast scalar inputs in case they come in as strings or something
    min_rsquared = float(min_rsquared)

    # get the linear regression result for this sample
    linregress_result = linregress_by_sample_id.get(sample_id)
    if linregress_result is None:
        log_messages_list.append(f"No linear regression fitted for sample "
                                 f"{sample_id}")
        return None, log_messages_list

    r_squared = linregress_result["rvalue"]**2
    if r_squared < min_rsquared:
        log_messages_list.append(f"R^2 of linear regression for sample "
                                 f"{sample_id} is {r_squared}, which is less "
                                 f"than the minimum allowed value of "
                                 f"{min_rsquared}.")
        return None, log_messages_list

    # get df of the rows of the working_df specific to this sample
    sample_df = working_df[
        working_df[SAMPLE_ID_KEY] == sample_id].copy()

    # predict mass of each OGU's gDNA in this sample from its counts
    # using the linear model
    ogu_gdna_masses = _calc_ogu_gdna_mass_ng_series_for_sample(
            sample_df, linregress_result[SLOPE_KEY],
            linregress_result[INTERCEPT_KEY])
    sample_df[OGU_GDNA_MASS_NG_KEY] = \
        sample_df[OGU_ID_KEY].map(ogu_gdna_masses)

    # get the mass of gDNA put into sequencing for this sample
    sequenced_sample_gdna_mass_ng = per_sample_info_df.loc[
        per_sample_info_df[SAMPLE_ID_KEY] == sample_id,
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY].values[0]

    # calc the # of genomes of each OGU per gram of gDNA in the sample
    # (normalized by the grams of gDNA in the *sequenced* sample)
    ogu_genomes_per_gdnas = _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
        sample_df, sequenced_sample_gdna_mass_ng, is_test=is_test)
    sample_df[OGU_GENOMES_PER_G_OF_GDNA_KEY] = \
        sample_df[OGU_ID_KEY].map(ogu_genomes_per_gdnas)

    # assume the # of cells of the microbe represented by each OGU
    # per gram of gDNA in this sample is the same as the number of genomes
    # of that OGU per gram of gDNA. This is known to be not quite right
    # (since some cells are dividing and thus have an extra copy, and also
    # because apparently polyploidy is not uncommon among microbes), but
    # that's why these things are called "simplifying assumptions" ...
    sample_df[OGU_CELLS_PER_G_OF_GDNA_KEY] = \
        sample_df[OGU_GENOMES_PER_G_OF_GDNA_KEY]

    # for each potential output metric:
    # 1. multiply the ratio by the genomes of each OGU per g of gDNA in the
    #    sample to get the genomes of each OGU per metric for the sample
    # 2. set the # of cells of the microbe represented by each OGU per gram
    #    of gDNA in this sample to be the same as the number of genomes
    # Don't worry about whether the ratio columns are in the df at this
    # point; earlier on, we put them in as NaN if not there so all these
    # calculations can be done in one place and just be NaN if not relevant.
    for cell_metric_key, metric_info in SAMPLE_LEVEL_METRICS_DICT.items():
        ratio_key = metric_info[RATIO_NAME_KEY]
        ratio_for_sample = per_sample_info_df.loc[
            per_sample_info_df[SAMPLE_ID_KEY] == sample_id,
            ratio_key].values[0]

        # calculate # of cells (i.e., genomes) of each OGU per metric
        # for this sample
        sample_df[cell_metric_key] = (
                sample_df[OGU_GENOMES_PER_G_OF_GDNA_KEY] * ratio_for_sample)

    return sample_df, log_messages_list


def _calc_ogu_gdna_mass_ng_series_for_sample(
        sample_df: pd.DataFrame,
        sample_linregress_slope: float,
        sample_linregress_intercept: float) -> pd.Series:

    """Calculates mass of OGU gDNA in ng for each OGU in a sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        A Dataframe with rows for a single sample, containing at least columns
        for OGU_ID_KEY and OGU_READ_COUNT_KEY.
    sample_linregress_slope: float
        Slope of the linear regression model for the sample.
    sample_linregress_intercept: float
        Intercept of the linear regression model for the sample.

    Returns
    -------
    ogu_genomes_per_g_of_gdna_series : pd.Series
        A Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU per gram of gDNA in the sample.
    """
    working_df = sample_df.copy()

    # NOTE that the linear regressions were originally done as described in
    # the Zaramela et al notebooks, where the log10 of the CPM values were
    # used as the independent variable.  Later scripts by Oriane Moyne
    # showed that this is not necessary and that it is equivalent to simply
    # use log10 of the read counts as the independent variable (as long as it
    # is used for *both* the fit and the prediction, of course!).  Please see
    # documentation on the fit_syndna_models.src._fit_linear_regression_models
    # method for a full description of this change.

    # add column of log10(ogu read counts)
    working_df[LOG_10_OGU_READ_COUNT_KEY] = \
        np.log10(working_df[OGU_READ_COUNT_KEY])

    # calculate log10(ogu gdna mass) of each OGU's gDNA in this sample
    # by multiplying each OGU's log10(ogu read count) by the slope of this
    # sample's regression model and adding the model's intercept.
    # NB: this requires that the linear regression models were derived
    # using synDNA masses *in ng* and not in some other unit.
    working_df[LOG_10_OGU_GDNA_MASS_NG_KEY] = (
            working_df[LOG_10_OGU_READ_COUNT_KEY] *
            sample_linregress_slope +
            sample_linregress_intercept)

    # calculate the actual mass in ng of each OGU's gDNA by raising 10 to the
    # log10(ogu gdna mass) power; set the series index to the OGU_ID_KEY
    ogu_gdna_mass_ng_series = \
        10 ** working_df[LOG_10_OGU_GDNA_MASS_NG_KEY]
    ogu_gdna_mass_ng_series.name = OGU_GDNA_MASS_NG_KEY
    ogu_gdna_mass_ng_series.index = sample_df[OGU_ID_KEY]

    return ogu_gdna_mass_ng_series


def _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
        sample_df: pd.DataFrame,
        total_sample_gdna_mass_ng: float,
        is_test: Optional[bool] = False) -> pd.Series:

    """Calculates # of OGU genomes per gram of gDNA for each OGU in a sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        A Dataframe with rows related to only a single sample, containing
        at least columns for OGU_ID_KEY, OGU_LEN_IN_BP_KEY, and
        OGU_GDNA_MASS_NG_KEY.
    total_sample_gdna_mass_ng: float
        Total mass of gDNA in the sample (across all OGUs) in ng. Note this
        should NOT include the mass of the syndna added to the sample
    is_test: Optional[bool]
        Default is False.  If True, the function will use the less-precise
        value of Avogadro's number (6.022*(10^23)) used in cell [16] of the
        notebook, rather than the more precise value (6.02214076*(10^23))
        calculation used if False.  This is True in testing ONLY.

    Returns
    -------
    ogu_genomes_per_g_of_gdna_series : pd.Series
        A Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU per gram of gDNA of the sample.
    """

    # calculate the number of genomes of each OGU in the sequenced sample
    ogu_genomes_series_for_sample = \
        _calc_ogu_genomes_series_for_sample(sample_df, is_test=is_test)

    # the above is the number of genomes of each OGU that were found in all
    # the sequenced sample gDNA; we want the number of genomes of each OGU
    # per gram of gDNA in the sample, so we divide by the total gDNA mass
    # in the sample (across all OGUs).  Note that both measurements are in ng.
    ogu_genomes_per_ng_of_gdna_series = \
        ogu_genomes_series_for_sample / total_sample_gdna_mass_ng

    # to get the number of genomes per gram of gDNA, we multiply by 1e9
    # (since there are 1e9 ng/gram)
    ogu_genomes_per_g_of_gdna_series = \
        ogu_genomes_per_ng_of_gdna_series * 1e9

    return ogu_genomes_per_g_of_gdna_series


def _calc_ogu_genomes_series_for_sample(
        sample_df: pd.DataFrame,
        is_test: Optional[bool] = False) -> pd.Series:

    """Calculates # of OGU genomes for each OGU in a sequenced sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        A Dataframe with rows related to only a single sample, containing
        at least columns for OGU_ID_KEY, OGU_LEN_IN_BP_KEY, and
        OGU_GDNA_MASS_NG_KEY.
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

    This calculates the total number of genomes for each OGU in the sequenced
    sample by the equation:

        mass of OGU's gDNA in ng * Avogadro's number in genomes/mole
    =	---------------------------------------------------------------
        length of OGU genome in basepairs *
            650 g/mole per basepair (on average) * 10^9 ng/g

    NB: the constant factor on the bottom right MUST CHANGE if the
    units of the OGU gDNA mass are NOT nanograms!

    Avogadro's number is 6.02214076 × 10^23 , and is the number of
    molecules--in this case, genomes--in a mole of a substance.
    """

    ogu_copies_per_g_series = calc_copies_genomic_element_per_g_series(
        sample_df[OGU_LEN_IN_BP_KEY], DNA_BASEPAIR_G_PER_MOLE, is_test=is_test)
    ogu_copies_per_extracted_sample_series = \
        sample_df[OGU_GDNA_MASS_NG_KEY] * \
        ogu_copies_per_g_series / NANOGRAMS_PER_GRAM

    # Set the index of the series to be the OGU_ID_KEY
    ogu_copies_per_extracted_sample_series.index = sample_df[OGU_ID_KEY]
    return ogu_copies_per_extracted_sample_series


def calc_ogu_cell_counts_biom(
        absolute_quant_params_per_sample_df: pd.DataFrame,
        linregress_by_sample_id: Dict[str, Dict[str, float]],
        ogu_counts_per_sample_biom: biom.Table,
        ogu_coverage_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        min_coverage: float,
        min_rsquared: float,
        output_cell_counts_metric: str) -> (biom.Table, List[str]):

    """Calcs input cell count metric for each ogu & sample via linear models.

    Parameters
    ----------
    absolute_quant_params_per_sample_df:  pd.DataFrame
        A Dataframe of at least SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
        ELUTE_VOL_UL_KEY, and SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY for each
        sample. It should also have at least one of SAMPLE_VOLUME_UL_KEY,
        SAMPLE_SURFACE_AREA_CM2_KEY, and/or SAMPLE_IN_ALIQUOT_MASS_G_KEY.
    linregress_by_sample_id : dict[str, dict[str: float]]
        Dictionary keyed by sample id, containing for each sample either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a dictionary
        representation of the sample's LinregressResult.
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_coverage_df : pd.DataFrame
        A DataFrame containing a column for OGU_ID_KEY and either a column for
        OGU_PERCENT_COVERAGE_KEY (indicating the coverage is the same for all
        samples) or a column for each sample id, which holds coverage of 
        that OGU in that sample as EITHER a fraction or a percentage.
        NOTE THAT IT IS UP TO THE USER TO ENSURE THAT THEY KNOW WHICH TYPE OF 
        VALUE (fraction or percent) IS BEING USED AND THAT THEY PROVIDE THE
        APPROPRIATE min_coverage VALUE ACCORDINGLY.
    ogu_lengths_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    min_coverage : float
        Minimum allowable coverage of an OGU across the whole dataset
        required to include that OGU in the output. May represent either 
        a fraction (e.g., 0.01 for 1%) or a percentage (e.g., 1 for 1%), and
        **must be consistent with the type of values in ogu_coverage_df**.
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.
    output_cell_counts_metric : str
        Name of the desired output cell count metric; options are:
        OGU_CELLS_PER_G_OF_GDNA_KEY, OGU_CELLS_PER_G_OF_SAMPLE_KEY,
        OGU_CELLS_PER_UL_OF_SAMPLE_KEY, or OGU_CELLS_PER_CM2_OF_SAMPLE_KEY.

    Returns
    -------
    ogu_cell_counts_biom : biom.Table
        An OGU/sample biom.Table holding the predicted number of cells per gram
        of material, with the material type being defined by the
        output_cell_counts_metric.
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    # check if the inputs all have the required columns
    extra_required = set()
    if output_cell_counts_metric in SAMPLE_LEVEL_METRICS_DICT:
        extra_required = {SAMPLE_LEVEL_METRICS_DICT[
                              output_cell_counts_metric][DENOMINATOR_KEY]}
    required_cols_list = list(
        {SAMPLE_ID_KEY, SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY} |
        set(REQUIRED_DNA_PREP_INFO_KEYS) | extra_required)
    validate_required_columns_exist(
        absolute_quant_params_per_sample_df, required_cols_list,
        "sample info is missing required column(s)")

    validate_required_columns_exist(
        ogu_lengths_df, [OGU_ID_KEY, OGU_LEN_IN_BP_KEY],
        "OGU lengths are missing required column(s)")

    validate_required_columns_exist(
        ogu_coverage_df, [OGU_ID_KEY],
        f"{_COVERAGE_DATA_NAME} is missing required column(s)")

    # handle either original case where OGU coverage is the same for
    # all samples, or the later case where it is different for each sample;
    # from here on in, deal only with per-sample OGU coverages
    ogu_coverage_per_sample_df = \
        _generate_ogu_coverages_per_sample_df(
            ogu_coverage_df, ogu_counts_per_sample_biom)

    _validate_sample_ids_in_inputs(absolute_quant_params_per_sample_df,
                                   ogu_counts_per_sample_biom,
                                   ogu_coverage_per_sample_df)

    _validate_ogu_ids_in_inputs(ogu_counts_per_sample_biom,
                                ogu_coverage_per_sample_df,
                                ogu_lengths_df)

    working_params_df = absolute_quant_params_per_sample_df.copy()

    # cast the GDNA_CONCENTRATION_NG_UL_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
    # ELUTE_VOL_UL_KEY, and SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY columns of
    # params df to float if they aren't already
    float_col_names = [
        GDNA_CONCENTRATION_NG_UL_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
        SAMPLE_SURFACE_AREA_CM2_KEY, SAMPLE_VOLUME_UL_KEY,
        ELUTE_VOL_UL_KEY, SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY]
    working_params_df = cast_cols(working_params_df, float_col_names, True)

    # Remove from input biom any samples that have bad params
    cols_to_filter_on = required_cols_list.copy()
    cols_to_filter_on.remove(SAMPLE_ID_KEY)  # don't filter on sample id
    # TODO: this is a hack; I'm not sure what effect setting the index to
    #  SAMPLE_ID_KEY on the "real" df would have, and I don't have time to
    #  vet it, so I'm just making a copy and setting the index on that.
    filter_params_df = working_params_df.copy()
    filter_params_df.set_index(SAMPLE_ID_KEY, inplace=True)
    filtered_ogu_counts_per_sample_biom, log_msgs_list = \
        filter_data_by_sample_info(
            filter_params_df, ogu_counts_per_sample_biom, cols_to_filter_on)

    # calc GDNA_FROM_ALIQUOT_MASS_G_KEY, the total grams of gDNA that are in
    # the elute after extraction; this is sample-specific
    per_sample_calc_info_df = calc_gs_genomic_element_in_aliquot(
        working_params_df, GDNA_CONCENTRATION_NG_UL_KEY,
        GDNA_FROM_ALIQUOT_MASS_G_KEY)

    for curr_metric_key, curr_metric_dict in SAMPLE_LEVEL_METRICS_DICT.items():
        metric_ratio_col_name = curr_metric_dict[RATIO_NAME_KEY]
        metric_ratio_denom_key = curr_metric_dict[DENOMINATOR_KEY]
        if metric_ratio_denom_key not in per_sample_calc_info_df.columns:
            per_sample_calc_info_df[metric_ratio_denom_key] = np.nan

        per_sample_calc_info_df[metric_ratio_col_name] = \
            per_sample_calc_info_df[GDNA_FROM_ALIQUOT_MASS_G_KEY] / \
            per_sample_calc_info_df[metric_ratio_denom_key]

    # convert input biom table to a dataframe with sparse columns, which
    # should act basically the same as a dense dataframe but use less memory
    ogu_counts_per_sample_df = \
        filtered_ogu_counts_per_sample_biom.to_dataframe(dense=False)

    ogu_cell_counts_long_format_df, calc_log_msgs_list = (
        _calc_long_format_ogu_cell_counts_df(
            linregress_by_sample_id, ogu_counts_per_sample_df,
            ogu_coverage_per_sample_df, ogu_lengths_df,
            per_sample_calc_info_df, min_coverage, min_rsquared))
    log_msgs_list.extend(calc_log_msgs_list)

    ogu_cell_counts_wide_format_df = ogu_cell_counts_long_format_df.pivot(
        index=OGU_ID_KEY, columns=SAMPLE_ID_KEY)[output_cell_counts_metric]

    # replace NaNs with 0s; per Daniel McDonald, much downstream analysis
    # cannot handle NaNs, and it is preferable to set invalid values
    # to 0 and provide a log message saying they are not usable than to leave
    # them as NaNs
    ogu_cell_counts_wide_format_df.fillna(0, inplace=True)

    # convert dataframe to biom table; input params are
    # data (the "output_cell_count_metric"s), observation_ids (the "ogu_id"s),
    # and sample_ids (er, the "sample_id"s)
    ogu_cell_counts_biom = biom.Table(
        ogu_cell_counts_wide_format_df.values,
        ogu_cell_counts_wide_format_df.index,
        ogu_cell_counts_wide_format_df.columns)

    return ogu_cell_counts_biom, log_msgs_list


def calc_ogu_cell_counts_per_g_of_sample_for_qiita(
        sample_info_df: pd.DataFrame,
        prep_info_df: pd.DataFrame,
        linregress_by_sample_id_fp: str,
        ogu_counts_per_sample_biom: biom.Table,
        ogu_coverage_df: pd.DataFrame,
        ogu_lengths_fp: str,
        min_coverage: float,
        min_rsquared: float = DEFAULT_MIN_RSQUARED,
        syndna_mass_fraction_of_sample: float =
        DEFAULT_SYNDNA_MASS_FRACTION_OF_SAMPLE) \
        -> Dict[str, Union[str, biom.Table]]:
    """Calculates the number of cells per gram of sample material.

    Parameters
    ----------
    sample_info_df: pd.DataFrame
        A Dataframe containing sample info for all samples in the prep,
        including SAMPLE_ID_KEY and SAMPLE_IN_ALIQUOT_MASS_G_KEY
    prep_info_df: pd.DataFrame
        A Dataframe containing prep info for all samples in the prep,
        including SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY, and
        ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY.
    linregress_by_sample_id_fp: str
        String containing the filepath to the yaml file holding the
        dictionary keyed by sample id, containing for each sample a dictionary
        representation of the sample's LinregressResult.
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_coverage_df : pd.DataFrame
        A DataFrame containing a column for OGU_ID_KEY and either a column for
        OGU_PERCENT_COVERAGE_KEY (indicating the coverage is the same for all
        samples) or a column for each sample id, which holds the coverage of 
        that OGU in that sample, expressed as either a fraction or a percentage.
        NOTE THAT IT IS UP TO THE USER TO ENSURE THAT THEY KNOW WHICH TYPE OF 
        VALUE (fraction or percent) IS BEING USED AND THAT THEY PROVIDE THE
        APPROPRIATE min_coverage VALUE ACCORDINGLY.
    ogu_lengths_fp : str
        String containing the filepath to a tab-separated, two-column,
        no-header file in which the first column is the OGU id and the
         second is the OGU length in basepairs
    min_coverage : float
        Minimum allowable coverage of an OGU in a sample needed to include
        that OGU/sample in the output, expressed in the same units
        (fraction or percent) as used in ogu_coverage_df.
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.
    syndna_mass_fraction_of_sample: float
        Fraction of the mass of the sample that is added as syndna (usually
        0.05, which is to say 5%).

    Returns
    -------
    output_by_out_type : dict of str or biom.Table
        Dictionary of outputs keyed by their type Currently, the following keys
        are defined:
        CELL_COUNT_RESULT_KEY: biom.Table holding the calculated number of
        cells per gram of sample material for each OGU in each sample.
        CELL_COUNT_LOG_KEY: log of messages from the cell count calc process.
    """

    # check if the inputs all have the required columns
    validate_required_columns_exist(
        sample_info_df, [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY],
        "sample info is missing required column(s)")

    return _calc_ogu_cell_counts_per_x_of_sample_for_qiita(
        sample_info_df, prep_info_df, linregress_by_sample_id_fp,
        ogu_counts_per_sample_biom, ogu_coverage_df, ogu_lengths_fp,
        OGU_CELLS_PER_G_OF_SAMPLE_KEY, min_coverage, min_rsquared,
        syndna_mass_fraction_of_sample)


def calc_ogu_cell_counts_per_cm2_of_sample_for_qiita(
        sample_info_df: pd.DataFrame,
        prep_info_df: pd.DataFrame,
        linregress_by_sample_id_fp: str,
        ogu_counts_per_sample_biom: biom.Table,
        ogu_coverage_df: pd.DataFrame,
        ogu_lengths_fp: str,
        min_coverage: float,
        min_rsquared: float = DEFAULT_MIN_RSQUARED,
        syndna_mass_fraction_of_sample: float =
        DEFAULT_SYNDNA_MASS_FRACTION_OF_SAMPLE) \
        -> Dict[str, Union[str, biom.Table]]:

    # check if the inputs all have the required columns
    validate_required_columns_exist(
        sample_info_df, [SAMPLE_ID_KEY, SAMPLE_SURFACE_AREA_CM2_KEY],
        "sample info is missing required column(s)")

    return _calc_ogu_cell_counts_per_x_of_sample_for_qiita(
        sample_info_df, prep_info_df, linregress_by_sample_id_fp,
        ogu_counts_per_sample_biom, ogu_coverage_df, ogu_lengths_fp,
        OGU_CELLS_PER_CM2_OF_SAMPLE_KEY, min_coverage, min_rsquared,
        syndna_mass_fraction_of_sample)


def calc_ogu_cell_counts_per_ul_of_sample_for_qiita(
        sample_info_df: pd.DataFrame,
        prep_info_df: pd.DataFrame,
        linregress_by_sample_id_fp: str,
        ogu_counts_per_sample_biom: biom.Table,
        ogu_coverage_df: pd.DataFrame,
        ogu_lengths_fp: str,
        min_coverage: float,
        min_rsquared: float = DEFAULT_MIN_RSQUARED,
        syndna_mass_fraction_of_sample: float =
        DEFAULT_SYNDNA_MASS_FRACTION_OF_SAMPLE) \
        -> Dict[str, Union[str, biom.Table]]:

    # check if the inputs all have the required columns
    validate_required_columns_exist(
        sample_info_df, [SAMPLE_ID_KEY, SAMPLE_VOLUME_UL_KEY],
        "sample info is missing required column(s)")

    return _calc_ogu_cell_counts_per_x_of_sample_for_qiita(
        sample_info_df, prep_info_df, linregress_by_sample_id_fp,
        ogu_counts_per_sample_biom, ogu_coverage_df, ogu_lengths_fp,
        OGU_CELLS_PER_UL_OF_SAMPLE_KEY, min_coverage, min_rsquared,
        syndna_mass_fraction_of_sample)
