import biom.table
import pandas
from pysyndna.src.util import calc_copies_per_g_series, \
    calc_g_genomic_element_of_sample_in_aliquot, \
    validate_required_columns_exist, \
    validate_metadata_vs_reads_sample_id_consistency, \
    validate_metadata_vs_prep_sample_id_consistency, SAMPLE_ID_KEY, \
    SAMPLE_IN_ALIQUOT_MASS_G_KEY, ELUTE_VOL_UL_KEY, RNA_BASE_G_PER_MOLE, \
    REQUIRED_SAMPLE_INFO_KEYS

OGU_ORF_ID_KEY = "ogu_orf_id"
OGU_ORF_START_KEY = "ogu_orf_start"
OGU_ORF_END_KEY = "ogu_orf_end"
OGU_ORF_LEN_KEY = "ogu_orf_len"
COPIES_PER_G_OGU_ORF_SSRNA_KEY = "copies_per_g_ogu_orf_ss_rna"
TOTAL_BIOLOGICAL_READS_KEY = "total_biological_reads_r1r2"
SSRNA_CONCENTRATION_NG_UL_KEY = "total_rna_concentration_ng_ul"
SSRNA_FROM_ALIQUOT_MASS_G_KEY = "ssrna_from_aliquot_mass_g"
REQUIRED_RNA_PREP_INFO_KEYS = [SAMPLE_ID_KEY, SSRNA_CONCENTRATION_NG_UL_KEY,
                               ELUTE_VOL_UL_KEY, TOTAL_BIOLOGICAL_READS_KEY]


def _read_ogu_orf_coords_to_df(wol_reannotations_fp: str) -> pandas.DataFrame:
    """Read the OGU+ORF coordinates file into a DataFrame.

    Parameters
    ----------
    wol_reannotations_fp : str
    Filepath to the ORF coordinates file in the wol reannotations format, e.g.:
    >G000005825
    1	816	2168
    2	2348	3490
    3	3744	3959
    4	3971	5086
    5	5098	5373
    6	5432	7372
    7	7399	9966

    Returns
    -------
    ogu_orf_coords_df : pandas.DataFrame
    A DataFrame containing columns for OGU_ORF_ID_KEY, OGU_ORF_START_KEY,
    and OGU_ORF_END_KEY.
    """
    curr_ogu_id, curr_ogu_orf_id = None, None
    curr_ogu_orf_start, curr_ogu_orf_end = None, None
    ogu_orf_ids, ogu_orf_starts, ogu_orf_ends = [], [], []

    with open(wol_reannotations_fp, "r") as fh:
        for line in fh.readlines():
            line = line.strip()
            if line.startswith(">G"):
                curr_ogu_id = line.replace(">", "")
            else:
                line_pieces = line.split("\t")
                curr_orf_id = line_pieces[0]
                curr_ogu_orf_start = int(line_pieces[1])
                curr_ogu_orf_end = int(line_pieces[2])
                curr_ogu_orf_id = curr_ogu_id + "_" + curr_orf_id
                ogu_orf_ids.append(curr_ogu_orf_id)
                ogu_orf_starts.append(curr_ogu_orf_start)
                ogu_orf_ends.append(curr_ogu_orf_end)
            # endif what to do with this line
        # next line

    ogu_orf_coords_dict = {
        OGU_ORF_ID_KEY: ogu_orf_ids,
        OGU_ORF_START_KEY: ogu_orf_starts,
        OGU_ORF_END_KEY: ogu_orf_ends
    }
    coords_df = pandas.DataFrame(ogu_orf_coords_dict)
    return coords_df


def _calc_ogu_orf_copies_per_g_from_coords(
        ogu_orf_coords_df: pandas.DataFrame) -> pandas.DataFrame:
    """Calculate the copies per gram of each OGU+ORF ssRNA.

    Note that this not (necessarily) the same as the copies per gram of the
    ssRNA *transcript* containing each OGU+ORF, since the latter might also
    contain other OGU+ORFs and thus be heavier.
    Parameters
    ----------
    ogu_orf_coords_df : pandas.DataFrame
        A DataFrame with columns for OGU_ORF_ID_KEY, OGU_ORF_START_KEY, and
        OGU_ORF_END_KEY.

    Returns
    -------
    ogu_orf_copies_per_g_df: pandas.DataFrame
        A DataFrame with columns for OGU_ORF_ID_KEY and
        COPIES_PER_G_OGU_ORF_SSRNA_KEY.
    """

    output_df = ogu_orf_coords_df.copy()

    # calculate the length of each OGU+ORF ssRNA:
    # abs(ogu_orf_end - ogu_orf_start) + 1
    # abs because sometimes the start is greater than the end,
    # +1 because the length is inclusive
    output_df[OGU_ORF_LEN_KEY] = \
        output_df[OGU_ORF_END_KEY] - \
        output_df[OGU_ORF_START_KEY]
    output_df[OGU_ORF_LEN_KEY] = \
        output_df[OGU_ORF_LEN_KEY].abs()
    output_df[OGU_ORF_LEN_KEY] = \
        output_df[OGU_ORF_LEN_KEY] + 1

    # calculate the copies per gram of each OGU+ORF ssRNA
    ogu_orf_copies_per_g_series = calc_copies_per_g_series(
        output_df[OGU_ORF_LEN_KEY], RNA_BASE_G_PER_MOLE)

    output_df[COPIES_PER_G_OGU_ORF_SSRNA_KEY] = \
        ogu_orf_copies_per_g_series
    output_df.index = output_df[OGU_ORF_ID_KEY]

    return output_df


def _calc_copies_of_ogu_orf_ssrna_per_g_sample(
        quant_params_per_sample_df: pandas.DataFrame,
        reads_per_ogu_orf_per_sample_biom: biom.Table,
        ogu_orf_copies_per_g_ssrna_df: pandas.DataFrame) -> biom.Table:

    """Calculate the copies of each OGU+ORF ssRNA per gram of sample.

    Parameters
    ----------
    quant_params_per_sample_df : pandas.DataFrame
        A DataFrame containing at least SAMPLE_ID_KEY,
        SAMPLE_IN_ALIQUOT_MASS_G_KEY, SSRNA_CONCENTRATION_NG_UL_KEY,
        ELUTE_VOL_UL_KEY, and TOTAL_BIOLOGICAL_READS_KEY.
    reads_per_ogu_orf_per_sample_biom : biom.Table
        A biom.Table with the number of reads per OGU+ORF per sample, such
        as that output by woltka.
    ogu_orf_copies_per_g_ssrna_df: pandas.DataFrame
        A DataFrame with columns for OGU_ORF_ID_KEY and
        COPIES_PER_G_OGU_ORF_SSRNA_KEY.

    Returns
    -------
    copies_of_ogu_orf_ssrna_per_g_sample : biom.Table
        A biom.Table with the copies of each OGU+ORF ssRNA per gram of sample.
    """

    # turn REQUIRED_SAMPLE_INFO_KEYS and REQUIRED_RNA_PREP_INFO_KEYS into sets
    # and combine them into a single set, then turn it back into a list
    required_cols_list = list(
        set(REQUIRED_SAMPLE_INFO_KEYS) | set(REQUIRED_RNA_PREP_INFO_KEYS))
    validate_required_columns_exist(
        quant_params_per_sample_df, required_cols_list,
        "parameters dataframe is missing required column(s)")

    # validate that the sample ids in the quant_params_per_sample_df match the
    # sample ids in the reads_per_ogu_orf_per_sample_biom. Ignore sample ids
    # in the quant_params_per_sample_df that are not in the biom table; those
    # could just be samples that failed sequencing/etc.
    _ = validate_metadata_vs_reads_sample_id_consistency(
        quant_params_per_sample_df, reads_per_ogu_orf_per_sample_biom)

    # Set index on quant_params_per_sample_df to be SAMPLE_ID_KEY for easy
    # lookup of values by sample id during biom lambda functions
    quant_params_per_sample_df.index = \
        quant_params_per_sample_df[SAMPLE_ID_KEY]

    # Calculate the grams of total ssRNA from each sample that are in the elute
    # after extraction
    g_total_ssrna_per_sample_df = calc_g_genomic_element_of_sample_in_aliquot(
        quant_params_per_sample_df, SSRNA_CONCENTRATION_NG_UL_KEY,
        SSRNA_FROM_ALIQUOT_MASS_G_KEY)

    # step 1 of OGU+ORF quantitation is upstream of this function:
    # Run woltka to get the reads_per_ogu_orf_per_sample_biom.
    # Calculations below are done directly on biom tables, since they are
    # expected to be very large and very sparse.

    # step 2:
    # Calculate fraction of total biological reads per OGU+ORF per sample:
    # Divide every value in reads_per_ogu_orf_per_sample_biom by the
    # value of the TOTAL_BIOLOGICAL_READS_KEY for that value's OGU_ORF_ID_KEY
    # in quant_params_per_sample_df.
    # See https://biom-format.org/documentation/generated/biom.table.Table.transform.html
    # for details of how to write and use a function for biom.transform().
    def get_fraction_of_sample_reads(data, id_, _):
        # df.at[] is fast to get a single value by a row/column label pair
        return data / quant_params_per_sample_df.at[id_, TOTAL_BIOLOGICAL_READS_KEY]
    fraction_of_sample_reads_per_sample_biom = \
        reads_per_ogu_orf_per_sample_biom.transform(
            f=get_fraction_of_sample_reads, axis='sample', inplace=False)

    # step 3:
    # Calculate grams of ssRNA per OGU+ORF per sample:
    # Multiply the fraction of total biological reads per OGU+ORF per sample
    # by the total grams of ssRNA from each sample that are in the elute after
    # extraction.
    def get_ogu_orf_ssrna_g_in_sample(data, id_, _):
        return data * g_total_ssrna_per_sample_df.at[id_, SSRNA_FROM_ALIQUOT_MASS_G_KEY]
    g_ssrna_per_ogu_orf_per_sample_biom = \
        fraction_of_sample_reads_per_sample_biom.transform(
            f=get_ogu_orf_ssrna_g_in_sample, axis='sample', inplace=False)

    # step 4:
    # Calculate copies per OGU+ORF per sample
    # Multiply the grams of ssRNA of each OGU+ORF per sample by the copies per
    # gram of each OGU+ORF ssRNA.
    # This gives you the copies of each OGU+ORF ssRNA present in the whole
    # extracted sample.
    def get_copies_per_ogu_orf_per_sample(data, id_, _):
        return data * ogu_orf_copies_per_g_ssrna_df.at[id_, COPIES_PER_G_OGU_ORF_SSRNA_KEY]
    copies_per_ogu_orf_per_sample_biom = \
        g_ssrna_per_ogu_orf_per_sample_biom.transform(
            f=get_copies_per_ogu_orf_per_sample, axis='observation', inplace=False)

    # Step 5:
    # Calculate the copies of each OGU+ORF ssRNA per gram of sample material
    # Divide the copies per OGU+ORF in each extracted sample by the grams of
    # sample material put into the extraction for the relevant sample
    def get_copies_per_g_sample(data, id_, _):
        return data / quant_params_per_sample_df.at[id_, SAMPLE_IN_ALIQUOT_MASS_G_KEY]
    copies_of_ogu_orf_ssrna_per_g_sample_biom = \
        copies_per_ogu_orf_per_sample_biom.transform(
            f=get_copies_per_g_sample, axis='sample', inplace=False)

    return copies_of_ogu_orf_ssrna_per_g_sample_biom


def calc_copies_of_ogu_orf_ssrna_per_g_sample(
        quant_params_per_sample_df: pandas.DataFrame,
        reads_per_ogu_orf_per_sample_biom: biom.Table,
        ogu_orf_coords_fp: str) -> biom.Table:
    """Calculate the copies of each OGU+ORF ssRNA per gram of sample.

    Parameters
    ----------
    quant_params_per_sample_df : pandas.DataFrame
        A DataFrame containing at least SAMPLE_ID_KEY,
        SAMPLE_IN_ALIQUOT_MASS_G_KEY, SSRNA_CONCENTRATION_NG_UL_KEY,
        ELUTE_VOL_UL_KEY, and TOTAL_BIOLOGICAL_READS_KEY.
    reads_per_ogu_orf_per_sample_biom : biom.Table
        A biom.Table with the number of reads per OGU+ORF per sample, such
        as that output by woltka.
    ogu_orf_coords_fp : str
        Filepath to the OGU+ORF coordinates file, such as the coords.txt
        file used by woltka, in the format shown below:
        >G000005825
        1	816	2168
        2	2348	3490
        3	3744	3959
        4	3971	5086
        5	5098	5373
        6	5432	7372
        7	7399	9966
        <etc.>

    Returns
    -------
    copies_of_ogu_orf_ssrna_per_g_sample : biom.Table
        A biom.Table with the copies of each OGU+ORF ssRNA per gram of sample.
    """

    # Calculate the copies per gram of each OGU+ORF ssRNA
    ogu_orf_coords_df = _read_ogu_orf_coords_to_df(ogu_orf_coords_fp)
    ogu_orf_copies_per_g_ssrna_df = _calc_ogu_orf_copies_per_g_from_coords(
        ogu_orf_coords_df)

    copies_of_ogu_orf_ssrna_per_g_sample_biom = \
        _calc_copies_of_ogu_orf_ssrna_per_g_sample(
            quant_params_per_sample_df, reads_per_ogu_orf_per_sample_biom,
            ogu_orf_copies_per_g_ssrna_df)

    return copies_of_ogu_orf_ssrna_per_g_sample_biom


def calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita(
        sample_info_df: pandas.DataFrame,
        prep_info_df: pandas.DataFrame,
        reads_per_ogu_orf_per_sample_biom: biom.Table,
        ogu_orf_coords_fp: str) -> biom.Table:

    """Calculate the copies of each OGU+ORF ssRNA per gram of sample for Qiita.

    Parameters
    ----------
    sample_info_df : pandas.DataFrame
        A DataFrame containing sample info for all samples in the prep,
        including SAMPLE_ID_KEY and SAMPLE_IN_ALIQUOT_MASS_G_KEY
    prep_info_df : pandas.DataFrame
        A DataFrame containing prep info for all samples in the prep,
        including SAMPLE_ID_KEY, SSRNA_CONCENTRATION_NG_UL_KEY,
        ELUTE_VOL_UL_KEY, and TOTAL_BIOLOGICAL_READS_KEY.
    reads_per_ogu_orf_per_sample_biom : biom.Table
        A biom.Table with the number of reads per OGU+ORF per sample, such
        as that output by woltka.
    ogu_orf_coords_fp : str
        Filepath to the OGU+ORF coordinates file, such as the coords.txt
        file used by woltka, in the format shown below:
        >G000005825
        1	816	2168
        2	2348	3490
        3	3744	3959
        4	3971	5086
        5	5098	5373
        6	5432	7372
        7	7399	9966
        <etc.>

    Returns
    -------
    copies_of_ogu_orf_ssrna_per_g_sample : biom.Table
        A biom.Table with the copies of each OGU+ORF ssRNA per gram of sample.
    """

    # check if the inputs all have the required columns
    validate_required_columns_exist(
        sample_info_df, REQUIRED_SAMPLE_INFO_KEYS,
        "sample info is missing required column(s)")

    validate_required_columns_exist(
        prep_info_df, REQUIRED_RNA_PREP_INFO_KEYS,
        "prep info is missing required column(s)")

    # validate that the sample ids in the sample_info_df match the sample ids
    # in the prep_info_df. Ignore sample ids in sample_info_df that are not in
    # the prep_info_df; these could just not be included in this prep.
    _ = validate_metadata_vs_prep_sample_id_consistency(
        sample_info_df, prep_info_df)

    quant_params_per_sample_df = prep_info_df.merge(
        sample_info_df, on=SAMPLE_ID_KEY, how="inner")

    copies_of_ogu_orf_ssrna_per_g_sample_biom = \
        calc_copies_of_ogu_orf_ssrna_per_g_sample(
            quant_params_per_sample_df, reads_per_ogu_orf_per_sample_biom,
            ogu_orf_coords_fp)

    return copies_of_ogu_orf_ssrna_per_g_sample_biom
